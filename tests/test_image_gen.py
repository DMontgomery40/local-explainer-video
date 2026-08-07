from __future__ import annotations

from pathlib import Path

import pytest

from core import image_gen
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


# The AN_04-08-1986 explainer shipped 14 slides carrying the patient identifier
# followed by the visual prompt's own style sentence ("Premium medical illustration,
# metaphor-driven but data-grounded."), each under a different invented crest. The
# style words are art direction and the crest was never requested; both reached the
# picture because nothing told the image model where on-screen text stops. These pin
# that instruction into both generation paths, since either one can render a slide.
def test_codex_prompt_forbids_painting_style_words_and_inventing_logos(tmp_path: Path):
    prompt = build_codex_image_prompt(
        prompt=(
            'Dark navy background. Lower-left: "AN_04-08-1986" '
            "Premium medical illustration, metaphor-driven but data-grounded."
        ),
        output_path=tmp_path / "images" / "scene_008.png",
    )

    assert "only the words this prompt places inside quotation marks" in prompt
    assert "never draw them as letters" in prompt
    assert "invent none" in prompt


def test_openai_image_path_carries_the_same_text_discipline(monkeypatch, tmp_path: Path):
    import base64

    sent: dict[str, str] = {}
    pixel = base64.b64encode(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d4944415478da6364f8cf000000030101002d0d0a2d0000000049454e44ae426082"
        )
    ).decode()

    class _Result:
        data = [type("D", (), {"b64_json": pixel})()]

    class _Images:
        def generate(self, **kwargs):
            sent["prompt"] = kwargs["prompt"]
            return _Result()

    class _Client:
        images = _Images()

    monkeypatch.setattr(image_gen, "_normalize_image_to_target", lambda *a, **k: (1664, 928))
    monkeypatch.setitem(__import__("sys").modules, "openai", type("M", (), {"OpenAI": lambda **k: _Client()}))
    monkeypatch.setenv("OPENAI_IMAGE_API_KEY", "test-key")

    image_gen._generate_image_openai(
        prompt='Lower-left: "AN_04-08-1986" Premium clinical infographic.',
        output_path=tmp_path / "scene_000.png",
        model="gpt-image-2",
        target_width=1664,
        target_height=928,
    )

    assert "only the words this prompt places inside quotation marks" in sent["prompt"]
    assert "invent none" in sent["prompt"]
