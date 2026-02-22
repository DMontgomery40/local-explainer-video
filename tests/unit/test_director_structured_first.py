from __future__ import annotations

import pytest

import core.director as director


def test_director_uses_structured_first_without_scene_typer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(director, "use_template_pipeline", lambda: True)
    monkeypatch.setattr(director, "load_prompt", lambda _name: "prompt")

    def _fake_generate(system_prompt: str, user_prompt: str, *, require_visual_prompt: bool) -> list[dict]:
        assert require_visual_prompt is False
        return [
            {
                "id": 0,
                "uid": "u0",
                "title": "T",
                "narration": "N",
                "visual_prompt": "",
                "scene_type": "bar_volume_chart",
                "structured_data": {},
            }
        ]

    monkeypatch.setattr(director, "_generate_with_openai", _fake_generate)
    monkeypatch.setattr(director, "annotate_scenes_with_types", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("should not call fallback typer")))
    out = director.generate_storyboard("input text", provider="openai")
    assert out[0]["scene_type"] == "bar_volume_chart"


def test_director_blocks_when_structured_missing_and_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(director, "use_template_pipeline", lambda: True)
    monkeypatch.setattr(director, "load_prompt", lambda _name: "prompt")
    monkeypatch.setenv("ALLOW_DOWNSTREAM_SCENE_TYPER_FALLBACK", "false")

    def _fake_generate(system_prompt: str, user_prompt: str, *, require_visual_prompt: bool) -> list[dict]:
        return [
            {
                "id": 0,
                "uid": "u0",
                "title": "T",
                "narration": "N",
                "visual_prompt": "V",
            }
        ]

    monkeypatch.setattr(director, "_generate_with_openai", _fake_generate)
    with pytest.raises(ValueError, match="requires `scene_type \\+ structured_data`"):
        director.generate_storyboard("input text", provider="openai")

