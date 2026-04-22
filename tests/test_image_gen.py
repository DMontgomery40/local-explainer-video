from __future__ import annotations

from pathlib import Path

import pytest

from core.image_gen import build_codex_image_prompt, generate_scene_image


def test_build_codex_image_prompt_uses_gpt_image_2_and_target_constraints(tmp_path: Path):
    output_path = tmp_path / "images" / "scene_000.png"

    prompt = build_codex_image_prompt(
        prompt='Warm explainer slide with exact text "LUMIT"',
        output_path=output_path,
        title="Signal Timing",
    )

    assert "gpt-image-2" in prompt
    assert "landscape 16:9" in prompt
    assert "1664x928" in prompt
    assert '"LUMIT"' in prompt
    assert str(output_path) in prompt


def test_generate_scene_image_prefers_prompt_based_codex_generation(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_generate_image(prompt, output_path, **kwargs):
        captured["prompt"] = prompt
        captured["output_path"] = Path(output_path)
        captured["kwargs"] = kwargs
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr("core.image_gen.generate_image", fake_generate_image)

    scene = {"id": 2, "title": "Executive Function", "visual_prompt": "Exact prompt"}
    result = generate_scene_image(scene, tmp_path)

    assert result == tmp_path / "images" / "scene_002.png"
    assert captured["prompt"] == "Exact prompt"
    assert captured["kwargs"]["title"] == "Executive Function"
    assert scene["image_path"] == str(result)


def test_generate_scene_image_falls_back_to_remotion_for_promptless_scene(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_render_scene_still(*, family, props, output_path):
        captured["family"] = family
        captured["props"] = props
        captured["output_path"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return output_path

    monkeypatch.setattr("core.image_gen._render_scene_still", fake_render_scene_still)

    scene = {"id": 1, "title": "Fallback Title", "visual_prompt": ""}
    result = generate_scene_image(scene, tmp_path)

    assert captured["family"] == "narration_slide"
    assert captured["props"] == {"headline": "Fallback Title", "body": ""}
    assert result == tmp_path / "images" / "scene_001.png"
    assert scene["image_path"] == str(result)


def test_generate_scene_image_rejects_motion_scene(tmp_path: Path):
    with pytest.raises(ValueError, match="motion/template scene"):
        generate_scene_image({"id": 7, "scene_type": "motion", "visual_prompt": "Prompt"}, tmp_path)
