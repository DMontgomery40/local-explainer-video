#!/usr/bin/env python3
"""Long-running deterministic TDD loop with learning-ledger enforcement."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.template_pipeline.learning_log import LearningLog


DEFAULT_LEARNING_PATH = REPO_ROOT / ".codex" / "LEARNING.md"
DEFAULT_STATE_PATH = REPO_ROOT / ".codex" / "loop_state" / "template_pipeline_tdd_state.json"
DEFAULT_APPROVAL_PATH = REPO_ROOT / ".codex" / "template_approval" / "approval.json"
DEFAULT_PLAN_PATH = REPO_ROOT / ".codex" / "loop_state" / "phase_plan.json"

STATUS_RUNNING = "RUNNING"
STATUS_WAITING_APPROVAL = "WAITING_TEMPLATE_APPROVAL"
STATUS_BLOCKED = "BLOCKED"
STATUS_COMPLETED = "COMPLETED"


def _utc_now_string() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_regression_test_ref() -> str:
    return (
        f"{(REPO_ROOT / 'tests' / 'execution' / 'test_tdd_loop_learning_hooks.py').resolve()}"
        "::test_phase_gate_summary_written_on_gate_execution"
    )


def default_phase_plan() -> dict[str, Any]:
    reg = _default_regression_test_ref()
    return {
        "final_phase": 5,
        "phases": {
            "1": {
                "steps": [
                    {
                        "gate": "phase-gate-fast",
                        "command": "python3 -m pytest tests/unit tests/contract -q --maxfail=1",
                        "timeout_seconds": 120,
                        "severity": "blocker",
                        "type": "test_failure",
                        "regression_test": reg,
                    },
                    {
                        "gate": "phase-gate-execution",
                        "command": "python3 -m pytest tests/execution -q",
                        "timeout_seconds": 600,
                        "severity": "blocker",
                        "type": "test_failure",
                        "regression_test": reg,
                    },
                    {
                        "gate": "phase-gate-integration-dev",
                        "command": (
                            "python3 scripts/run_template_pipeline_e2e.py "
                            "--source-project 09-05-1954-0 --provider openai "
                            "--template-mode development --name integration_gate"
                        ),
                        "timeout_seconds": 2400,
                        "severity": "blocker",
                        "type": "runtime_error",
                        "regression_test": reg,
                    },
                ]
            },
            "2": {
                "steps": [
                    {
                        "gate": "phase2-generate-qwen-templates",
                        "command": "python3 scripts/generate_qwen_templates.py --live --all-archetypes",
                        "timeout_seconds": 3600,
                        "severity": "blocker",
                        "type": "external",
                        "regression_test": reg,
                    },
                    {
                        "gate": "phase2-build-anchors",
                        "command": "python3 scripts/build_template_anchors.py --all",
                        "timeout_seconds": 300,
                        "severity": "blocker",
                        "type": "test_failure",
                        "regression_test": reg,
                    },
                    {
                        "gate": "phase2-build-approval-packet",
                        "command": "python3 scripts/build_template_approval_packet.py",
                        "timeout_seconds": 300,
                        "severity": "blocker",
                        "type": "test_failure",
                        "regression_test": reg,
                    },
                ]
            },
            "3": {
                "steps": [
                    {
                        "gate": "phase3-contract",
                        "command": (
                            "python3 -m pytest tests/contract/test_no_generic_prod_fallback.py "
                            "tests/contract/test_manifest_contract.py -q"
                        ),
                        "timeout_seconds": 300,
                        "severity": "blocker",
                        "type": "test_failure",
                        "regression_test": reg,
                    },
                    {
                        "gate": "phase3-production-integration",
                        "command": (
                            "python3 scripts/run_template_pipeline_e2e.py "
                            "--source-project 09-05-1954-0 --provider openai "
                            "--template-mode production --name production_gate"
                        ),
                        "timeout_seconds": 2400,
                        "severity": "blocker",
                        "type": "runtime_error",
                        "regression_test": reg,
                    },
                ]
            },
            "4": {
                "steps": [
                    {
                        "gate": "phase4-director-structured-first",
                        "command": "python3 -m pytest tests/unit/test_director_structured_first.py -q",
                        "timeout_seconds": 300,
                        "severity": "blocker",
                        "type": "test_failure",
                        "regression_test": reg,
                    },
                    {
                        "gate": "phase4-scene-schema-contract",
                        "command": "python3 -m pytest tests/unit/template_pipeline/test_scene_schema_contract.py -q",
                        "timeout_seconds": 300,
                        "severity": "blocker",
                        "type": "test_failure",
                        "regression_test": reg,
                    },
                ]
            },
            "5": {
                "steps": [
                    {
                        "gate": "phase5-parity-eval",
                        "command": "python3 scripts/eval_parity.py --require-confidence 84.7",
                        "timeout_seconds": 120,
                        "severity": "blocker",
                        "type": "quality_regression",
                        "regression_test": reg,
                    },
                    {
                        "gate": "phase5-final-production-e2e",
                        "command": (
                            "python3 scripts/run_template_pipeline_e2e.py "
                            "--source-project 09-05-1954-0 --provider openai "
                            "--template-mode production --name final_release_gate"
                        ),
                        "timeout_seconds": 2400,
                        "severity": "blocker",
                        "type": "runtime_error",
                        "regression_test": reg,
                    },
                ]
            },
        },
    }


def _normalize_state(raw: dict[str, Any]) -> dict[str, Any]:
    state = dict(raw)
    defaults: dict[str, Any] = {
        "phase": 1,
        "cycle": 0,
        "status": STATUS_RUNNING,
        "open_failures_by_gate": {},
        "last_resolved_by_gate": {},
        "failed_gate_events": [],
        "successful_gate_events": [],
        "phase_gate_summaries": [],
        "phase_execution_events": [],
        "completed_utc": None,
        "last_updated_utc": _utc_now_string(),
    }
    for key, value in defaults.items():
        if key not in state:
            if isinstance(value, dict):
                state[key] = {}
            elif isinstance(value, list):
                state[key] = []
            else:
                state[key] = value
    return state


def _ensure_state_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _normalize_state({})
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    _ensure_state_file(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Loop state file must be a JSON object")
    return _normalize_state(raw)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    payload = _normalize_state(payload)
    payload["last_updated_utc"] = _utc_now_string()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _truncate_error(stderr: str, stdout: str, limit: int = 260) -> str:
    text = (stderr or "").strip() or (stdout or "").strip()
    if not text:
        text = "command failed with no output"
    compact = " ".join(text.split())
    return compact[: limit - 1] + "…" if len(compact) > limit else compact


@dataclass(slots=True)
class GateRunResult:
    ok: bool
    returncode: int
    signal: str
    stdout: str
    stderr: str
    learning_entry_id: str | None = None
    timed_out: bool = False


class TDDLoop:
    """Stateful TDD loop controller with mandatory learning hooks."""

    def __init__(
        self,
        *,
        state_path: Path = DEFAULT_STATE_PATH,
        learning_path: Path = DEFAULT_LEARNING_PATH,
        approval_path: Path = DEFAULT_APPROVAL_PATH,
    ) -> None:
        self.state_path = state_path
        self.approval_path = approval_path
        self.learning = LearningLog(learning_path)
        _ensure_state_file(self.state_path)

    def _state(self) -> dict[str, Any]:
        return _read_json(self.state_path)

    def _save_state(self, state: dict[str, Any]) -> None:
        _write_json(self.state_path, state)

    @staticmethod
    def _gate_key(phase: int, gate: str) -> str:
        return f"{phase}:{gate}"

    def increment_cycle(self) -> int:
        state = self._state()
        state["cycle"] = int(state.get("cycle", 0)) + 1
        self._save_state(state)
        return int(state["cycle"])

    def _append_phase_execution_event(
        self,
        *,
        phase: int,
        gate: str,
        command: str,
        ok: bool,
        learning_entry_id: str | None,
    ) -> None:
        state = self._state()
        events = list(state.get("phase_execution_events", []))
        events.append(
            {
                "timestamp_utc": _utc_now_string(),
                "phase": phase,
                "gate": gate,
                "command": command,
                "ok": bool(ok),
                "learning_entry_id": learning_entry_id,
            }
        )
        state["phase_execution_events"] = events
        self._save_state(state)

    def record_gate_failure(
        self,
        *,
        phase: int,
        gate: str,
        signal: str,
        observed_error: str,
        severity: str = "blocker",
        learning_type: str = "test_failure",
        artifacts: list[str] | None = None,
    ) -> str:
        state = self._state()
        key = self._gate_key(phase, gate)
        open_by_gate = dict(state.get("open_failures_by_gate", {}))
        resolved_by_gate = dict(state.get("last_resolved_by_gate", {}))
        prev_entry = resolved_by_gate.get(key)
        if key in open_by_gate:
            entry_id = open_by_gate[key]
        elif prev_entry:
            entry = self.learning.record_recurrence(
                previous_entry_id=prev_entry,
                phase=phase,
                severity=severity,
                learning_type=learning_type,
                signal=signal,
                observed_error=observed_error,
                artifacts=artifacts or [],
            )
            entry_id = entry.entry_id
            open_by_gate[key] = entry_id
        else:
            entry = self.learning.create_open_entry(
                phase=phase,
                severity=severity,
                learning_type=learning_type,
                signal=signal,
                observed_error=observed_error,
                artifacts=artifacts or [],
            )
            entry_id = entry.entry_id
            open_by_gate[key] = entry_id

        failed_events = list(state.get("failed_gate_events", []))
        failed_events.append(
            {
                "timestamp_utc": _utc_now_string(),
                "phase": phase,
                "gate": gate,
                "signal": signal,
                "learning_entry_id": entry_id,
            }
        )
        state["failed_gate_events"] = failed_events
        state["open_failures_by_gate"] = open_by_gate
        state["status"] = STATUS_RUNNING
        self._save_state(state)
        return entry_id

    def record_gate_fix(
        self,
        *,
        phase: int,
        gate: str,
        root_cause: str,
        fix_applied: str,
        prevention_rule: str,
        regression_test_added: str,
        linked_change: str = "none",
        artifacts: list[str] | None = None,
        emit_success_event: bool = True,
    ) -> str:
        state = self._state()
        key = self._gate_key(phase, gate)
        open_by_gate = dict(state.get("open_failures_by_gate", {}))
        entry_id = open_by_gate.get(key)
        if not entry_id:
            raise ValueError(f"No OPEN failure entry tracked for gate {key}")

        self.learning.resolve_entry(
            entry_id,
            root_cause=root_cause,
            fix_applied=fix_applied,
            prevention_rule=prevention_rule,
            regression_test_added=regression_test_added,
            linked_change=linked_change,
            artifacts=artifacts or [],
        )
        self.learning.maybe_refresh_guardrails()

        open_by_gate.pop(key, None)
        resolved_by_gate = dict(state.get("last_resolved_by_gate", {}))
        resolved_by_gate[key] = entry_id
        state["open_failures_by_gate"] = open_by_gate
        state["last_resolved_by_gate"] = resolved_by_gate
        if emit_success_event:
            success_events = list(state.get("successful_gate_events", []))
            success_events.append(
                {
                    "timestamp_utc": _utc_now_string(),
                    "phase": phase,
                    "gate": gate,
                    "learning_entry_id": entry_id,
                    "regression_test_added": regression_test_added,
                }
            )
            state["successful_gate_events"] = success_events
        state["status"] = STATUS_RUNNING
        self._save_state(state)
        return entry_id

    def record_phase_gate_summary(
        self,
        *,
        phase: int,
        passed: bool,
        summary: str,
        artifacts: list[str] | None = None,
    ) -> None:
        state = self._state()
        rows = list(state.get("phase_gate_summaries", []))
        rows.append(
            {
                "timestamp_utc": _utc_now_string(),
                "phase": phase,
                "passed": bool(passed),
                "summary": summary,
                "artifacts": artifacts or [],
            }
        )
        state["phase_gate_summaries"] = rows
        self._save_state(state)

    def validate_failure_event_links(self) -> tuple[bool, list[str]]:
        state = self._state()
        failures = list(state.get("failed_gate_events", []))
        doc = self.learning.load()
        ids = {e.entry_id for e in doc.entries}
        missing: list[str] = []
        for row in failures:
            eid = str(row.get("learning_entry_id", "")).strip()
            if not eid:
                missing.append(f"{row.get('phase')}:{row.get('gate')} missing learning_entry_id")
                continue
            if eid not in ids:
                missing.append(f"{row.get('phase')}:{row.get('gate')} references unknown {eid}")
        return (len(missing) == 0, missing)

    def has_unresolved_blockers(self) -> bool:
        return self.learning.has_open_blockers()

    def can_advance_phase(self) -> tuple[bool, str]:
        ok, missing = self.validate_failure_event_links()
        if not ok:
            return False, "phase blocked: failed gate event(s) missing learning entries"
        if self.has_unresolved_blockers():
            return False, "phase blocked: unresolved blocker learnings exist"
        return True, "ok"

    def attempt_phase_advance(self, target_phase: int) -> tuple[bool, str]:
        state = self._state()
        current = int(state.get("phase", 1))
        if target_phase <= current:
            return False, f"target phase must be greater than current phase ({current})"
        allowed, reason = self.can_advance_phase()
        if not allowed:
            state["status"] = STATUS_BLOCKED
            self._save_state(state)
            return False, reason
        if current == 2 and target_phase >= 3 and not self._template_approval_present():
            state["status"] = STATUS_WAITING_APPROVAL
            self._save_state(state)
            return False, "waiting for template approval artifact after phase 2"
        state["phase"] = target_phase
        state["status"] = STATUS_RUNNING
        self._save_state(state)
        return True, f"advanced to phase {target_phase}"

    def mark_completed(self) -> None:
        state = self._state()
        state["status"] = STATUS_COMPLETED
        state["completed_utc"] = _utc_now_string()
        self._save_state(state)

    def _template_approval_present(self) -> bool:
        if not self.approval_path.exists():
            return False
        try:
            payload = json.loads(self.approval_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        return bool(payload.get("approved") is True)

    def write_approval_artifact(
        self,
        *,
        approved: bool,
        approved_by: str = "human",
        notes: str = "",
        packet_dir: str = "",
    ) -> None:
        payload = {
            "approved": bool(approved),
            "approved_by": approved_by,
            "notes": notes,
            "packet_dir": packet_dir,
            "updated_utc": _utc_now_string(),
        }
        self.approval_path.parent.mkdir(parents=True, exist_ok=True)
        self.approval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run_gate(
        self,
        *,
        phase: int,
        gate: str,
        command: str,
        severity: str = "blocker",
        learning_type: str = "test_failure",
        artifacts: list[str] | None = None,
        cwd: Path = REPO_ROOT,
        timeout_seconds: int | None = None,
        auto_resolve_on_success: bool = True,
        regression_test_added: str | None = None,
        root_cause: str | None = None,
        fix_applied: str | None = None,
        prevention_rule: str | None = None,
    ) -> GateRunResult:
        signal = command
        proc: subprocess.CompletedProcess[str] | None = None
        timed_out = False
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            timeout_label = timeout_seconds if timeout_seconds is not None else "unknown"
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            error = f"command timed out after {timeout_label}s"
            if stderr.strip() or stdout.strip():
                error = f"{error}; {_truncate_error(stderr, stdout)}"
            entry_id = self.record_gate_failure(
                phase=phase,
                gate=gate,
                signal=signal,
                observed_error=error,
                severity=severity,
                learning_type=learning_type,
                artifacts=artifacts or [],
            )
            if gate.startswith("phase-gate"):
                self.record_phase_gate_summary(
                    phase=phase,
                    passed=False,
                    summary=f"{gate} timed out ({timeout_label}s)",
                    artifacts=artifacts or [],
                )
            self._append_phase_execution_event(
                phase=phase,
                gate=gate,
                command=command,
                ok=False,
                learning_entry_id=entry_id,
            )
            return GateRunResult(
                ok=False,
                returncode=124,
                signal=signal,
                stdout=stdout,
                stderr=stderr or error,
                learning_entry_id=entry_id,
                timed_out=timed_out,
            )

        assert proc is not None
        key = self._gate_key(phase, gate)
        state = self._state()
        open_by_gate = dict(state.get("open_failures_by_gate", {}))
        open_entry_id = open_by_gate.get(key)

        if proc.returncode != 0:
            entry_id = self.record_gate_failure(
                phase=phase,
                gate=gate,
                signal=signal,
                observed_error=_truncate_error(proc.stderr, proc.stdout),
                severity=severity,
                learning_type=learning_type,
                artifacts=artifacts or [],
            )
            if gate.startswith("phase-gate"):
                self.record_phase_gate_summary(
                    phase=phase,
                    passed=False,
                    summary=f"{gate} failed ({proc.returncode})",
                    artifacts=artifacts or [],
                )
            self._append_phase_execution_event(
                phase=phase,
                gate=gate,
                command=command,
                ok=False,
                learning_entry_id=entry_id,
            )
            return GateRunResult(
                ok=False,
                returncode=proc.returncode,
                signal=signal,
                stdout=proc.stdout,
                stderr=proc.stderr,
                learning_entry_id=entry_id,
                timed_out=timed_out,
            )

        resolved_entry_id: str | None = None
        if open_entry_id and auto_resolve_on_success:
            resolved_entry_id = self.record_gate_fix(
                phase=phase,
                gate=gate,
                root_cause=root_cause or f"{gate} failure condition no longer present after code changes",
                fix_applied=fix_applied or f"{gate} command now passes in TDD green step",
                prevention_rule=prevention_rule
                or "Keep command under automated loop coverage to prevent recurrence",
                regression_test_added=regression_test_added or _default_regression_test_ref(),
                artifacts=artifacts or [],
                emit_success_event=False,
            )

        state = self._state()
        success_events = list(state.get("successful_gate_events", []))
        success_events.append(
            {
                "timestamp_utc": _utc_now_string(),
                "phase": phase,
                "gate": gate,
                "learning_entry_id": resolved_entry_id,
                "signal": signal,
                "regression_test_added": regression_test_added or _default_regression_test_ref(),
            }
        )
        state["successful_gate_events"] = success_events
        state["status"] = STATUS_RUNNING
        self._save_state(state)
        if gate.startswith("phase-gate"):
            self.record_phase_gate_summary(
                phase=phase,
                passed=True,
                summary=f"{gate} passed",
                artifacts=artifacts or [],
            )
        self._append_phase_execution_event(
            phase=phase,
            gate=gate,
            command=command,
            ok=True,
            learning_entry_id=resolved_entry_id,
        )
        return GateRunResult(
            ok=True,
            returncode=0,
            signal=signal,
            stdout=proc.stdout,
            stderr=proc.stderr,
            learning_entry_id=resolved_entry_id,
            timed_out=timed_out,
        )

    def load_phase_plan(self, plan_path: Path | None = None) -> dict[str, Any]:
        candidate = plan_path or DEFAULT_PLAN_PATH
        payload: dict[str, Any]
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Phase plan must be a JSON object")
            payload = data
        else:
            payload = default_phase_plan()

        final_phase = int(payload.get("final_phase", 5))
        raw_phases = payload.get("phases")
        if not isinstance(raw_phases, dict):
            raise ValueError("Phase plan missing `phases` object")

        phases: dict[str, Any] = {}
        for phase_key, phase_spec in raw_phases.items():
            key = str(phase_key)
            if not isinstance(phase_spec, dict):
                raise ValueError(f"Phase spec for {key} must be an object")
            steps = phase_spec.get("steps")
            if steps is None:
                steps = []
            if not isinstance(steps, list):
                raise ValueError(f"Phase {key} steps must be an array")
            normalized_steps: list[dict[str, Any]] = []
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    raise ValueError(f"Phase {key} step {i+1} must be an object")
                gate = str(step.get("gate") or f"phase{key}-step{i+1}")
                command = str(step.get("command") or "").strip()
                if not command:
                    raise ValueError(f"Phase {key} step {gate} missing command")
                normalized_steps.append(
                    {
                        "gate": gate,
                        "command": command,
                        "timeout_seconds": int(step.get("timeout_seconds") or 0),
                        "severity": str(step.get("severity") or "blocker"),
                        "type": str(step.get("type") or "test_failure"),
                        "artifacts": step.get("artifacts") if isinstance(step.get("artifacts"), list) else [],
                        "regression_test": str(step.get("regression_test") or _default_regression_test_ref()),
                        "root_cause": str(step.get("root_cause") or ""),
                        "fix_applied": str(step.get("fix_applied") or ""),
                        "prevention_rule": str(step.get("prevention_rule") or ""),
                    }
                )
            phases[key] = {"steps": normalized_steps}
        return {"final_phase": final_phase, "phases": phases}

    def write_default_plan(self, path: Path = DEFAULT_PLAN_PATH, *, force: bool = False) -> Path:
        if path.exists() and not force:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default_phase_plan(), indent=2), encoding="utf-8")
        return path

    def run_plan(
        self,
        *,
        plan_path: Path | None = None,
        target_phase: int | None = None,
        cwd: Path = REPO_ROOT,
    ) -> dict[str, Any]:
        plan = self.load_phase_plan(plan_path)
        final_phase = int(target_phase or plan.get("final_phase", 5))
        state = self._state()
        current_phase = int(state.get("phase", 1))

        while current_phase <= final_phase:
            phase_spec = (plan.get("phases") or {}).get(str(current_phase), {})
            steps = phase_spec.get("steps") if isinstance(phase_spec, dict) else []
            if not isinstance(steps, list):
                steps = []

            for step in steps:
                result = self.run_gate(
                    phase=current_phase,
                    gate=str(step.get("gate") or f"phase{current_phase}-step"),
                    command=str(step.get("command") or ""),
                    severity=str(step.get("severity") or "blocker"),
                    learning_type=str(step.get("type") or "test_failure"),
                    artifacts=[str(a) for a in (step.get("artifacts") or [])],
                    cwd=cwd,
                    timeout_seconds=int(step.get("timeout_seconds") or 0) or None,
                    auto_resolve_on_success=True,
                    regression_test_added=str(step.get("regression_test") or _default_regression_test_ref()),
                    root_cause=str(step.get("root_cause") or ""),
                    fix_applied=str(step.get("fix_applied") or ""),
                    prevention_rule=str(step.get("prevention_rule") or ""),
                )
                if not result.ok:
                    blocked = self._state()
                    blocked["status"] = STATUS_BLOCKED
                    self._save_state(blocked)
                    self.record_phase_gate_summary(
                        phase=current_phase,
                        passed=False,
                        summary=f"phase {current_phase} blocked on {step.get('gate')}",
                        artifacts=[str(a) for a in (step.get("artifacts") or [])],
                    )
                    return {
                        "ok": False,
                        "status": STATUS_BLOCKED,
                        "phase": current_phase,
                        "failed_gate": step.get("gate"),
                        "returncode": result.returncode,
                    }

            self.record_phase_gate_summary(
                phase=current_phase,
                passed=True,
                summary=f"phase {current_phase} completed",
            )

            if current_phase >= final_phase:
                self.mark_completed()
                return {
                    "ok": True,
                    "status": STATUS_COMPLETED,
                    "phase": current_phase,
                    "message": "target phase reached",
                }

            advanced, reason = self.attempt_phase_advance(current_phase + 1)
            if not advanced:
                current_state = self._state()
                return {
                    "ok": False,
                    "status": current_state.get("status", STATUS_BLOCKED),
                    "phase": current_phase,
                    "message": reason,
                }
            current_phase = int(self._state().get("phase", current_phase + 1))

        self.mark_completed()
        return {
            "ok": True,
            "status": STATUS_COMPLETED,
            "phase": int(self._state().get("phase", final_phase)),
            "message": "plan complete",
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic TDD loop controller.")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--learning-path", default=str(DEFAULT_LEARNING_PATH))
    parser.add_argument("--approval-path", default=str(DEFAULT_APPROVAL_PATH))
    sub = parser.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status")
    status.set_defaults(cmd="status")

    cycle = sub.add_parser("cycle")
    cycle.set_defaults(cmd="cycle")

    fail = sub.add_parser("fail")
    fail.add_argument("--phase", type=int, required=True)
    fail.add_argument("--gate", required=True)
    fail.add_argument("--signal", required=True)
    fail.add_argument("--observed-error", required=True)
    fail.add_argument("--severity", default="blocker")
    fail.add_argument("--type", default="test_failure")
    fail.add_argument("--artifacts", nargs="*", default=[])

    fix = sub.add_parser("fix")
    fix.add_argument("--phase", type=int, required=True)
    fix.add_argument("--gate", required=True)
    fix.add_argument("--root-cause", required=True)
    fix.add_argument("--fix-applied", required=True)
    fix.add_argument("--prevention-rule", required=True)
    fix.add_argument("--regression-test", required=True)
    fix.add_argument("--linked-change", default="none")
    fix.add_argument("--artifacts", nargs="*", default=[])

    gate = sub.add_parser("run-gate")
    gate.add_argument("--phase", type=int, required=True)
    gate.add_argument("--gate", required=True)
    gate.add_argument("--command", required=True)
    gate.add_argument("--severity", default="blocker")
    gate.add_argument("--type", default="test_failure")
    gate.add_argument("--regression-test", default=_default_regression_test_ref())
    gate.add_argument("--timeout-seconds", type=int)
    gate.add_argument("--artifacts", nargs="*", default=[])

    adv = sub.add_parser("advance")
    adv.add_argument("--target-phase", type=int, required=True)

    init_plan = sub.add_parser("init-plan")
    init_plan.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    init_plan.add_argument("--force", action="store_true")

    show_plan = sub.add_parser("show-plan")
    show_plan.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))

    run_plan = sub.add_parser("run-plan")
    run_plan.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    run_plan.add_argument("--target-phase", type=int)

    set_approval = sub.add_parser("set-approval")
    set_approval.add_argument("--approved", choices=["true", "false"], required=True)
    set_approval.add_argument("--approved-by", default="human")
    set_approval.add_argument("--notes", default="")
    set_approval.add_argument("--packet-dir", default="")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    loop = TDDLoop(
        state_path=Path(args.state_path),
        learning_path=Path(args.learning_path),
        approval_path=Path(args.approval_path),
    )
    if args.cmd == "status":
        state = loop._state()
        ok, message = loop.can_advance_phase()
        print(json.dumps({"state": state, "phase_advance_ok": ok, "reason": message}, indent=2))
        return 0
    if args.cmd == "cycle":
        value = loop.increment_cycle()
        print(json.dumps({"cycle": value}, indent=2))
        return 0
    if args.cmd == "fail":
        entry_id = loop.record_gate_failure(
            phase=args.phase,
            gate=args.gate,
            signal=args.signal,
            observed_error=args.observed_error,
            severity=args.severity,
            learning_type=args.type,
            artifacts=args.artifacts,
        )
        print(json.dumps({"entry_id": entry_id}, indent=2))
        return 0
    if args.cmd == "fix":
        entry_id = loop.record_gate_fix(
            phase=args.phase,
            gate=args.gate,
            root_cause=args.root_cause,
            fix_applied=args.fix_applied,
            prevention_rule=args.prevention_rule,
            regression_test_added=args.regression_test,
            linked_change=args.linked_change,
            artifacts=args.artifacts,
        )
        print(json.dumps({"entry_id": entry_id}, indent=2))
        return 0
    if args.cmd == "run-gate":
        result = loop.run_gate(
            phase=args.phase,
            gate=args.gate,
            command=args.command,
            severity=args.severity,
            learning_type=args.type,
            artifacts=args.artifacts,
            timeout_seconds=args.timeout_seconds,
            regression_test_added=args.regression_test,
        )
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "returncode": result.returncode,
                    "learning_entry_id": result.learning_entry_id,
                    "timed_out": result.timed_out,
                },
                indent=2,
            )
        )
        return 0 if result.ok else result.returncode
    if args.cmd == "advance":
        ok, reason = loop.attempt_phase_advance(args.target_phase)
        print(json.dumps({"ok": ok, "reason": reason}, indent=2))
        return 0 if ok else 1
    if args.cmd == "init-plan":
        path = loop.write_default_plan(Path(args.plan_path), force=args.force)
        print(json.dumps({"plan_path": str(path)}, indent=2))
        return 0
    if args.cmd == "show-plan":
        plan = loop.load_phase_plan(Path(args.plan_path))
        print(json.dumps(plan, indent=2))
        return 0
    if args.cmd == "run-plan":
        plan_path = Path(args.plan_path)
        result = loop.run_plan(
            plan_path=plan_path if plan_path.exists() else None,
            target_phase=args.target_phase,
        )
        print(json.dumps(result, indent=2))
        status = str(result.get("status") or "")
        if status == STATUS_COMPLETED:
            return 0
        if status == STATUS_WAITING_APPROVAL:
            return 3
        return 1
    if args.cmd == "set-approval":
        loop.write_approval_artifact(
            approved=args.approved == "true",
            approved_by=args.approved_by,
            notes=args.notes,
            packet_dir=args.packet_dir,
        )
        print(json.dumps({"approval_path": str(loop.approval_path), "approved": args.approved == "true"}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
