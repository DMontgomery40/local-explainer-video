"""Downstream LLM scene typing for deterministic template pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Literal

import anthropic
import openai

from .scene_schemas import validate_scene


_PROMPTS: dict[str, str] = {}
_OPENAI_CLIENT: openai.OpenAI | None = None
_ANTHROPIC_CLIENT: anthropic.Anthropic | None = None


class SceneTyperError(RuntimeError):
    """Base class for typed-scene generation failures."""


class SceneTyperTimeoutError(SceneTyperError):
    """Raised when provider call times out repeatedly."""


class SceneTyperProviderError(SceneTyperError):
    """Raised on provider/API failures."""


class SceneTyperValidationError(SceneTyperError):
    """Raised when provider output is malformed or schema-invalid."""


def _load_prompt(name: str) -> str:
    if name not in _PROMPTS:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / f"{name}.txt"
        _PROMPTS[name] = prompt_path.read_text(encoding="utf-8")
    return _PROMPTS[name]


def _openai_client() -> openai.OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = openai.OpenAI()
    return _OPENAI_CLIENT


def _anthropic_client() -> anthropic.Anthropic:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = anthropic.Anthropic()
    return _ANTHROPIC_CLIENT


def _scene_typer_request_timeout_seconds() -> float:
    raw = str(os.getenv("SCENE_TYPER_REQUEST_TIMEOUT_SECONDS", "120")).strip()
    try:
        value = float(raw)
    except Exception:
        value = 120.0
    return max(5.0, value)


def _scene_typer_max_retries() -> int:
    raw = str(os.getenv("SCENE_TYPER_MAX_RETRIES", "2")).strip()
    try:
        value = int(raw)
    except Exception:
        value = 2
    return max(0, value)


def _scene_typer_backoff_seconds() -> float:
    raw = str(os.getenv("SCENE_TYPER_RETRY_BACKOFF_SECONDS", "1.5")).strip()
    try:
        value = float(raw)
    except Exception:
        value = 1.5
    return max(0.0, value)


def _extract_json(content: str) -> dict[str, Any]:
    raw = (content or "").strip()
    if "```json" in raw:
        start = raw.index("```json") + 7
        end = raw.index("```", start)
        raw = raw[start:end].strip()
    elif "```" in raw:
        start = raw.index("```") + 3
        end = raw.index("```", start)
        raw = raw[start:end].strip()
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise SceneTyperValidationError(f"Failed to parse scene typer JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SceneTyperValidationError("Scene typer output must be a JSON object")
    return parsed


def _call_openai(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    try:
        resp = _openai_client().responses.create(
            model="gpt-5.1",
            instructions=system_prompt,
            input=user_prompt,
            text={"format": {"type": "json_object"}},
            temperature=0.1,
            timeout=_scene_typer_request_timeout_seconds(),
        )
        return _extract_json(resp.output_text)
    except SceneTyperValidationError:
        raise
    except TimeoutError as exc:
        raise SceneTyperTimeoutError(str(exc)) from exc
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if "timeout" in text.lower():
            raise SceneTyperTimeoutError(text) from exc
        raise SceneTyperProviderError(text) from exc


def _call_anthropic(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    try:
        resp = _anthropic_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_block = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        return _extract_json(text_block)
    except SceneTyperValidationError:
        raise
    except TimeoutError as exc:
        raise SceneTyperTimeoutError(str(exc)) from exc
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if "timeout" in text.lower():
            raise SceneTyperTimeoutError(text) from exc
        raise SceneTyperProviderError(text) from exc


def _merge_annotations(scenes: list[dict[str, Any]], typed_payload: dict[str, Any]) -> list[dict[str, Any]]:
    typed_scenes = typed_payload.get("scenes")
    if not isinstance(typed_scenes, list):
        raise SceneTyperValidationError("Scene typer output missing scenes array")

    by_id: dict[int, dict[str, Any]] = {}
    for ts in typed_scenes:
        if not isinstance(ts, dict):
            continue
        sid = ts.get("id")
        if isinstance(sid, int):
            by_id[sid] = ts

    merged: list[dict[str, Any]] = []
    for idx, scene in enumerate(scenes):
        target = by_id.get(int(scene.get("id", idx)))
        if target is None and idx < len(typed_scenes) and isinstance(typed_scenes[idx], dict):
            target = typed_scenes[idx]
        if target is None:
            raise SceneTyperValidationError(f"Missing scene typer output for scene index={idx}")

        merged_scene = dict(scene)
        merged_scene["scene_type"] = target.get("scene_type")
        merged_scene["structured_data"] = target.get("structured_data")
        if "visual_prompt" in target and target.get("visual_prompt"):
            merged_scene["visual_prompt"] = target.get("visual_prompt")
        try:
            merged.append(validate_scene(merged_scene))
        except Exception as exc:
            raise SceneTyperValidationError(
                f"Scene typer produced invalid scene for id={merged_scene.get('id')}: {exc}"
            ) from exc
    return merged


def _call_provider_once(
    provider: Literal["openai", "anthropic"],
    *,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    if provider == "openai":
        return _call_openai(system_prompt, user_prompt)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt)
    raise SceneTyperProviderError(f"Unknown provider: {provider}")


def annotate_scenes_with_types(
    *,
    scenes: list[dict[str, Any]],
    input_text: str,
    provider: Literal["openai", "anthropic"] = "openai",
) -> list[dict[str, Any]]:
    """Attach deterministic `scene_type` + `structured_data` via downstream LLM pass."""

    system_prompt = _load_prompt("scene_typer_system")
    compact = [
        {
            "id": s.get("id", i),
            "title": s.get("title"),
            "narration": s.get("narration"),
            "visual_prompt": s.get("visual_prompt"),
        }
        for i, s in enumerate(scenes)
        if isinstance(s, dict)
    ]
    user_prompt = (
        "Convert storyboard scenes into deterministic scene typing.\n"
        "Use ONLY facts present in input text and scene text.\n\n"
        "INPUT_TEXT:\n"
        "-----\n"
        f"{input_text}\n"
        "-----\n\n"
        "SCENES:\n"
        "-----\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n"
        "-----\n\n"
        "Return JSON object with `scenes` array."
    )

    attempts = 1 + _scene_typer_max_retries()
    backoff = _scene_typer_backoff_seconds()
    errors: list[str] = []
    for idx in range(attempts):
        try:
            typed = _call_provider_once(
                provider=provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return _merge_annotations(scenes, typed)
        except (SceneTyperTimeoutError, SceneTyperProviderError, SceneTyperValidationError) as exc:
            err = f"attempt={idx + 1}/{attempts} {type(exc).__name__}: {exc}"
            errors.append(err)
            if idx >= attempts - 1:
                chain = " | ".join(errors)
                if isinstance(exc, SceneTyperTimeoutError):
                    raise SceneTyperTimeoutError(
                        f"Scene typer timed out after {attempts} attempts: {chain}"
                    ) from exc
                if isinstance(exc, SceneTyperValidationError):
                    raise SceneTyperValidationError(
                        f"Scene typer returned invalid output after {attempts} attempts: {chain}"
                    ) from exc
                raise SceneTyperProviderError(
                    f"Scene typer provider failed after {attempts} attempts: {chain}"
                ) from exc
            if backoff > 0:
                # Exponential-ish retry pause with small linear growth.
                time.sleep(backoff * (idx + 1))
        except Exception as exc:
            stack = traceback.format_exc(limit=2)
            raise SceneTyperProviderError(
                f"Unexpected scene typer failure ({type(exc).__name__}): {exc}\n{stack}"
            ) from exc

    raise SceneTyperError("Scene typer failed without yielding a typed payload")
