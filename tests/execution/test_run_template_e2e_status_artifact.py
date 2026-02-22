from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import scripts.run_template_pipeline_e2e as e2e


def _write_source_project(tmp_path: Path, project_name: str) -> Path:
    project_dir = tmp_path / "projects" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "meta": {
            "input_text": "input text",
            "llm_provider": "openai",
        },
        "scenes": [],
    }
    (project_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return project_dir


def test_e2e_writes_failed_status_artifact_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source_project(tmp_path, "source-1")
    (tmp_path / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(e2e, "generate_storyboard", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced boom")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_template_pipeline_e2e.py",
            "--source-project",
            str(source),
        ],
    )
    e2e.RUN_STATE["status"] = "not_started"
    e2e.RUN_STATE["status_artifact"] = None
    e2e.RUN_STATE["project_dir"] = None

    with pytest.raises(RuntimeError, match="forced boom"):
        e2e.main()

    created = sorted((tmp_path / "projects").glob("source-1__template_e2e_*"))
    assert created, "Expected output project directory to be created"
    status_artifact = created[-1] / "artifacts" / "template_e2e_status.json"
    assert status_artifact.exists(), "Expected failed run status artifact"
    payload = json.loads(status_artifact.read_text(encoding="utf-8"))
    assert payload.get("status") == "failed"
