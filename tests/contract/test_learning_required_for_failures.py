from __future__ import annotations

import json
from pathlib import Path

from scripts.tdd_loop import TDDLoop


def _loop(tmp_path: Path) -> TDDLoop:
    return TDDLoop(
        state_path=tmp_path / "loop_state.json",
        learning_path=tmp_path / "LEARNING.md",
        approval_path=tmp_path / "approval.json",
    )


def test_failure_events_must_reference_learning_entries(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    entry_id = loop.record_gate_failure(
        phase=1,
        gate="fast-gate",
        signal="pytest tests/unit -q",
        observed_error="AssertionError",
    )
    ok, missing = loop.validate_failure_event_links()
    assert ok
    assert missing == []
    assert entry_id


def test_missing_learning_entry_blocks_phase_advance(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    state = json.loads(loop.state_path.read_text(encoding="utf-8"))
    state["failed_gate_events"] = [
        {
            "timestamp_utc": "2026-02-22T00:00:00Z",
            "phase": 1,
            "gate": "fast-gate",
            "signal": "pytest tests/unit -q",
            "learning_entry_id": "L-20260222-000000-deadbe",
        }
    ]
    loop.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    ok, reason = loop.can_advance_phase()
    assert not ok
    assert "missing learning entries" in reason

