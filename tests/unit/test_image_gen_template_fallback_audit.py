from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.image_gen as image_gen
import core.template_pipeline as template_pipeline


def _scene() -> dict:
    return {
        "id": 3,
        "uid": "u3",
        "title": "T",
        "narration": "N",
        "visual_prompt": "Render this",
        "scene_type": "bar_volume_chart",
        "structured_data": {},
    }


def test_template_failure_logs_fallback_audit_and_uses_qwen_when_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "proj"
    (project_dir / "images").mkdir(parents=True, exist_ok=True)
    scene = _scene()

    monkeypatch.setenv("USE_TEMPLATE_PIPELINE", "true")
    monkeypatch.setenv("ALLOW_TEMPLATE_FALLBACK_TO_QWEN", "true")
    monkeypatch.setattr(template_pipeline, "render_scene_to_image", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("template failed")))
    monkeypatch.setattr(template_pipeline, "scene_eligible_for_template", lambda _scene: True)
    monkeypatch.setattr(template_pipeline, "use_template_pipeline", lambda: True)
    monkeypatch.setattr(template_pipeline, "allow_qwen_fallback", lambda: True)

    def _fake_generate(prompt: str, output_path: Path, **kwargs: object) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")
        return output_path

    monkeypatch.setattr(image_gen, "generate_image", _fake_generate)
    out = image_gen.generate_scene_image(scene, project_dir, model="qwen/qwen-image-2512")
    assert out.exists()
    assert scene.get("fallback_to_qwen") is True

    audit_path = project_dir / "artifacts" / "template_fallback_audit.jsonl"
    assert audit_path.exists()
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(r.get("event") == "template_render_fallback_to_qwen" for r in rows)


def test_template_failure_raises_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path / "proj2"
    (project_dir / "images").mkdir(parents=True, exist_ok=True)
    scene = _scene()

    monkeypatch.setenv("USE_TEMPLATE_PIPELINE", "true")
    monkeypatch.setenv("ALLOW_TEMPLATE_FALLBACK_TO_QWEN", "false")
    monkeypatch.setattr(template_pipeline, "render_scene_to_image", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("template failed")))
    monkeypatch.setattr(template_pipeline, "scene_eligible_for_template", lambda _scene: True)
    monkeypatch.setattr(template_pipeline, "use_template_pipeline", lambda: True)
    monkeypatch.setattr(template_pipeline, "allow_qwen_fallback", lambda: False)

    with pytest.raises(RuntimeError, match="template failed"):
        image_gen.generate_scene_image(scene, project_dir, model="qwen/qwen-image-2512")
    assert scene.get("fallback_to_qwen") is False
    audit_path = project_dir / "artifacts" / "template_fallback_audit.jsonl"
    assert audit_path.exists()

