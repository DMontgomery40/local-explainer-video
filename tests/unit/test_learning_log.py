from __future__ import annotations

from pathlib import Path

import pytest

from core.template_pipeline.learning_log import LearningLog, parse_learning_markdown


def _regression_ref(path: Path) -> str:
    return f"{path.resolve()}::test_case"


def test_entry_serialization_and_append(tmp_path: Path) -> None:
    learning_path = tmp_path / "LEARNING.md"
    log = LearningLog(learning_path)

    entry = log.create_open_entry(
        phase=1,
        severity="blocker",
        learning_type="test_failure",
        signal="pytest tests/unit -q",
        observed_error="AssertionError: expected 1 got 0",
        artifacts=[str((tmp_path / "artifact.json").resolve())],
    )
    assert entry.status == "OPEN"

    doc = parse_learning_markdown(learning_path)
    loaded = doc.entry_by_id(entry.entry_id)
    assert loaded is not None
    assert loaded.signal == "pytest tests/unit -q"
    assert loaded.artifacts


def test_required_field_enforcement(tmp_path: Path) -> None:
    learning_path = tmp_path / "LEARNING.md"
    log = LearningLog(learning_path)
    with pytest.raises(ValueError, match="Phase must be in"):
        log.create_open_entry(
            phase=9,
            severity="blocker",
            learning_type="test_failure",
            signal="pytest tests/unit -q",
            observed_error="boom",
        )

    entry = log.create_open_entry(
        phase=1,
        severity="high",
        learning_type="runtime_error",
        signal="python script.py",
        observed_error="timeout",
    )
    with pytest.raises(ValueError, match="absolute path"):
        log.resolve_entry(
            entry.entry_id,
            root_cause="bad timeout config",
            fix_applied="set timeout=300",
            prevention_rule="always pin timeout by gate class",
            regression_test_added="tests/unit/test_timeout.py::test_timeout",
        )


def test_status_transition_and_recurrence(tmp_path: Path) -> None:
    learning_path = tmp_path / "LEARNING.md"
    log = LearningLog(learning_path)
    entry = log.create_open_entry(
        phase=2,
        severity="blocker",
        learning_type="policy_drift",
        signal="phase2-parity-gate",
        observed_error="generic fallback used in production mode",
    )

    resolved = log.resolve_entry(
        entry.entry_id,
        root_cause="manifest catch-all was production-enabled",
        fix_applied="marked generic template dev-only and tightened selector",
        prevention_rule="production mode fails when only generic template matches",
        regression_test_added=_regression_ref(tmp_path / "tests" / "contract" / "test_no_generic.py"),
    )
    assert resolved.status == "RESOLVED"

    reopened = log.record_recurrence(
        previous_entry_id=entry.entry_id,
        phase=2,
        severity="blocker",
        learning_type="policy_drift",
        signal="phase2-parity-gate",
        observed_error="generic fallback reappeared after refactor",
    )
    assert reopened.status == "OPEN"

    doc = parse_learning_markdown(learning_path)
    previous = doc.entry_by_id(entry.entry_id)
    assert previous is not None
    assert previous.status == "REGRESSION"


def test_guardrails_refresh_after_five_resolved(tmp_path: Path) -> None:
    learning_path = tmp_path / "LEARNING.md"
    log = LearningLog(learning_path)
    test_ref = _regression_ref(tmp_path / "tests" / "unit" / "test_guardrails.py")

    for i in range(5):
        entry = log.create_open_entry(
            phase=1,
            severity="high",
            learning_type="test_failure",
            signal=f"pytest gate_{i}",
            observed_error=f"failure_{i}",
        )
        log.resolve_entry(
            entry.entry_id,
            root_cause=f"root cause {i}",
            fix_applied=f"fix {i}",
            prevention_rule=f"prevent rule {i}",
            regression_test_added=test_ref,
        )
    changed = log.maybe_refresh_guardrails()
    assert changed
    doc = parse_learning_markdown(learning_path)
    assert any("prevent rule" in rule for rule in doc.guardrails)
