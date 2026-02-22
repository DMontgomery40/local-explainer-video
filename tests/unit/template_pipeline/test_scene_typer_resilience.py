from __future__ import annotations

from typing import Any

import pytest

from core.template_pipeline import scene_typer


def _base_scenes() -> list[dict[str, Any]]:
    return [
        {
            "id": 0,
            "title": "T",
            "narration": "N",
            "visual_prompt": "V",
        }
    ]


def test_scene_typer_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    monkeypatch.setenv("SCENE_TYPER_MAX_RETRIES", "2")
    monkeypatch.setenv("SCENE_TYPER_RETRY_BACKOFF_SECONDS", "0")

    def _fake_call_provider_once(provider: str, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise scene_typer.SceneTyperProviderError("transient failure")
        return {
            "scenes": [
                {
                    "id": 0,
                    "scene_type": "bar_volume_chart",
                    "structured_data": {},
                }
            ]
        }

    monkeypatch.setattr(scene_typer, "_call_provider_once", _fake_call_provider_once)
    monkeypatch.setattr(scene_typer, "_load_prompt", lambda _name: "prompt")

    out = scene_typer.annotate_scenes_with_types(
        scenes=_base_scenes(),
        input_text="input",
        provider="openai",
    )
    assert calls["n"] == 2
    assert out[0]["scene_type"] == "bar_volume_chart"


def test_scene_typer_raises_validation_error_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCENE_TYPER_MAX_RETRIES", "1")
    monkeypatch.setenv("SCENE_TYPER_RETRY_BACKOFF_SECONDS", "0")

    def _always_invalid(provider: str, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {"unexpected": []}

    monkeypatch.setattr(scene_typer, "_call_provider_once", _always_invalid)
    monkeypatch.setattr(scene_typer, "_load_prompt", lambda _name: "prompt")

    with pytest.raises(scene_typer.SceneTyperValidationError, match="invalid output"):
        scene_typer.annotate_scenes_with_types(
            scenes=_base_scenes(),
            input_text="input",
            provider="openai",
        )

