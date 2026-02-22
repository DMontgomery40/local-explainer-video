from __future__ import annotations

import json
from pathlib import Path

from scripts.tdd_loop import STATUS_BLOCKED, TDDLoop


def _create_loop(tmp_path: Path) -> TDDLoop:
    state_path = tmp_path / "loop_state.json"
    learning_path = tmp_path / "LEARNING.md"
    approval_path = tmp_path / "approval.json"
    return TDDLoop(
        state_path=state_path,
        learning_path=learning_path,
        approval_path=approval_path,
    )


def test_failed_gate_creates_open_learning_entry(tmp_path: Path) -> None:
    loop = _create_loop(tmp_path)
    entry_id = loop.record_gate_failure(
        phase=1,
        gate="unit-fast",
        signal="pytest tests/unit -q",
        observed_error="AssertionError: mismatch",
    )
    doc = loop.learning.load()
    entry = doc.entry_by_id(entry_id)
    assert entry is not None
    assert entry.status == "OPEN"
    assert entry.severity == "blocker"


def test_fix_resolves_entry_with_regression_reference(tmp_path: Path) -> None:
    loop = _create_loop(tmp_path)
    loop.record_gate_failure(
        phase=1,
        gate="unit-fast",
        signal="pytest tests/unit -q",
        observed_error="AssertionError: mismatch",
    )
    entry_id = loop.record_gate_fix(
        phase=1,
        gate="unit-fast",
        root_cause="selector fallback was too broad",
        fix_applied="narrowed manifest selector and added hard fail",
        prevention_rule="never allow generic fallback in production mode",
        regression_test_added=f"{(tmp_path / 'tests' / 'contract' / 'test_no_generic.py').resolve()}::test_no_generic",
    )
    entry = loop.learning.load().entry_by_id(entry_id)
    assert entry is not None
    assert entry.status == "RESOLVED"
    assert "::test_no_generic" in entry.regression_test_added


def test_phase_advance_blocked_on_unresolved_blockers(tmp_path: Path) -> None:
    loop = _create_loop(tmp_path)
    loop.record_gate_failure(
        phase=2,
        gate="phase-gate-medium",
        signal="pytest tests/integration -q",
        observed_error="TimeoutError: gate exceeded",
        severity="blocker",
        learning_type="timeout",
    )
    ok, reason = loop.attempt_phase_advance(3)
    assert not ok
    assert "unresolved blocker" in reason
    state = json.loads(loop.state_path.read_text(encoding="utf-8"))
    assert state["status"] == STATUS_BLOCKED


def test_phase_gate_summary_written_on_gate_execution(tmp_path: Path) -> None:
    loop = _create_loop(tmp_path)
    result = loop.run_gate(
        phase=1,
        gate="phase-gate-fast",
        command="python3 -c 'print(\"ok\")'",
    )
    assert result.ok
    state = json.loads(loop.state_path.read_text(encoding="utf-8"))
    summaries = state.get("phase_gate_summaries", [])
    assert summaries
    assert summaries[-1]["phase"] == 1
    assert summaries[-1]["passed"] is True
