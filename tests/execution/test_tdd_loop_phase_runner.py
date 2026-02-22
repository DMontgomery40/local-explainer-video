from __future__ import annotations

import json
from pathlib import Path

from scripts.tdd_loop import STATUS_COMPLETED, STATUS_WAITING_APPROVAL, TDDLoop


def _create_loop(tmp_path: Path) -> TDDLoop:
    return TDDLoop(
        state_path=tmp_path / "loop_state.json",
        learning_path=tmp_path / "LEARNING.md",
        approval_path=tmp_path / "approval.json",
    )


def _write_plan(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _regression_ref(tmp_path: Path) -> str:
    return f"{(tmp_path / 'tests' / 'execution' / 'test_loop.py').resolve()}::test_loop"


def test_run_plan_stops_only_at_phase2_waiting_for_approval(tmp_path: Path) -> None:
    loop = _create_loop(tmp_path)
    plan_path = tmp_path / "plan.json"
    reg = _regression_ref(tmp_path)
    _write_plan(
        plan_path,
        {
            "final_phase": 3,
            "phases": {
                "1": {"steps": [{"gate": "p1", "command": "python3 -c 'print(\"p1\")'", "regression_test": reg}]},
                "2": {"steps": [{"gate": "p2", "command": "python3 -c 'print(\"p2\")'", "regression_test": reg}]},
                "3": {"steps": [{"gate": "p3", "command": "python3 -c 'print(\"p3\")'", "regression_test": reg}]},
            },
        },
    )

    result = loop.run_plan(plan_path=plan_path, target_phase=3, cwd=tmp_path)
    assert result["ok"] is False
    assert result["status"] == STATUS_WAITING_APPROVAL
    assert result["phase"] == 2


def test_run_plan_resumes_after_human_approval(tmp_path: Path) -> None:
    loop = _create_loop(tmp_path)
    plan_path = tmp_path / "plan.json"
    reg = _regression_ref(tmp_path)
    _write_plan(
        plan_path,
        {
            "final_phase": 3,
            "phases": {
                "1": {"steps": [{"gate": "p1", "command": "python3 -c 'print(\"p1\")'", "regression_test": reg}]},
                "2": {"steps": [{"gate": "p2", "command": "python3 -c 'print(\"p2\")'", "regression_test": reg}]},
                "3": {"steps": [{"gate": "p3", "command": "python3 -c 'print(\"p3\")'", "regression_test": reg}]},
            },
        },
    )

    first = loop.run_plan(plan_path=plan_path, target_phase=3, cwd=tmp_path)
    assert first["status"] == STATUS_WAITING_APPROVAL

    loop.write_approval_artifact(approved=True, approved_by="human")
    second = loop.run_plan(plan_path=plan_path, target_phase=3, cwd=tmp_path)
    assert second["ok"] is True
    assert second["status"] == STATUS_COMPLETED
    assert second["phase"] == 3


def test_run_plan_auto_resolves_retried_gate_failure(tmp_path: Path) -> None:
    loop = _create_loop(tmp_path)
    plan_path = tmp_path / "plan.json"
    reg = _regression_ref(tmp_path)

    _write_plan(
        plan_path,
        {
            "final_phase": 1,
            "phases": {
                "1": {
                    "steps": [
                        {
                            "gate": "p1",
                            "command": "python3 -c 'import sys; sys.exit(2)'",
                            "regression_test": reg,
                        }
                    ]
                }
            },
        },
    )

    first = loop.run_plan(plan_path=plan_path, target_phase=1, cwd=tmp_path)
    assert first["ok"] is False

    _write_plan(
        plan_path,
        {
            "final_phase": 1,
            "phases": {
                "1": {
                    "steps": [
                        {
                            "gate": "p1",
                            "command": "python3 -c 'print(\"ok\")'",
                            "regression_test": reg,
                        }
                    ]
                }
            },
        },
    )

    second = loop.run_plan(plan_path=plan_path, target_phase=1, cwd=tmp_path)
    assert second["ok"] is True
    assert second["status"] == STATUS_COMPLETED
    doc = loop.learning.load()
    assert any(entry.status == "RESOLVED" for entry in doc.entries)


def test_run_gate_timeout_creates_timeout_learning_entry(tmp_path: Path) -> None:
    loop = _create_loop(tmp_path)
    result = loop.run_gate(
        phase=1,
        gate="phase-gate-timeout",
        command="python3 -c 'import time; time.sleep(2)'",
        learning_type="timeout",
        timeout_seconds=1,
    )
    assert result.ok is False
    assert result.timed_out is True
    assert result.returncode == 124
    doc = loop.learning.load()
    assert any(
        entry.status == "OPEN"
        and entry.learning_type == "timeout"
        and entry.signal == "python3 -c 'import time; time.sleep(2)'"
        for entry in doc.entries
    )
