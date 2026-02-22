from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.template_pipeline import TemplateRenderError
from core.template_pipeline.renderer import classify_render_exception, render_scene_to_image
from core.template_pipeline.scene_schemas import SceneValidationError
from core.template_pipeline.selector import TemplateSelectionError


def test_classify_render_exception_mapping() -> None:
    assert classify_render_exception(SceneValidationError("bad scene")) == "validation"
    assert classify_render_exception(TemplateSelectionError("no template")) == "selection"
    assert classify_render_exception(ValueError("bad value")) == "value_error"


def test_render_failure_writes_audit_and_error_category(tmp_path: Path) -> None:
    scene = {
        "id": 99,
        "uid": "u99",
        "title": "Bad",
        "narration": "Bad",
        "scene_type": "bar_volume_chart",
        "structured_data": "not-an-object",
    }
    with pytest.raises(TemplateRenderError) as exc:
        render_scene_to_image(scene, project_dir=tmp_path)
    assert exc.value.category == "validation"
    assert scene.get("render_error", {}).get("category") == "validation"

    audit_path = tmp_path / "artifacts" / "template_render_audit.jsonl"
    assert audit_path.exists()
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row.get("event") == "render_failure" for row in rows)


def test_render_production_mode_hard_fails_when_not_covered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMPLATE_PIPELINE_MODE", "production")
    scene = {
        "id": 1,
        "uid": "u1",
        "title": "Trend",
        "narration": "Trend",
        "scene_type": "multi_session_trend",
        "structured_data": {"metric": "M", "points": []},
    }
    with pytest.raises(TemplateRenderError) as exc:
        render_scene_to_image(scene, project_dir=tmp_path)
    assert exc.value.category == "selection"

