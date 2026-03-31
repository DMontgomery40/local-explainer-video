"""Storyboard generation for qEEG explainer videos.

Supports two paths:
  1. Local agent runners (codex/claude CLI) via local_planner — the original path.
  2. Direct Anthropic API calls with composition-based output — the Remotion path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

_PROMPTS: dict[str, str] = {}

StoryboardProvider = Literal["codex", "claude", "openai", "anthropic", "api"]


def _resolve_prompt_path(name: str) -> Path:
    """Resolve prompt path, supporting versioned director_system experiments."""
    prompts_dir = Path(__file__).parent.parent / "prompts"

    if name == "director_system":
        version = (os.getenv("DIRECTOR_SYSTEM_VERSION") or "").strip()
        if version:
            versioned_path = prompts_dir / "director_system_versions" / version / "director_system.txt"
            if not versioned_path.exists():
                raise FileNotFoundError(
                    f"DIRECTOR_SYSTEM_VERSION={version!r} not found at {versioned_path}"
                )
            return versioned_path

        override_path = (os.getenv("DIRECTOR_SYSTEM_PROMPT_PATH") or "").strip()
        if override_path:
            path = Path(override_path)
            if not path.is_absolute():
                path = Path(__file__).parent.parent / override_path
            if not path.exists():
                raise FileNotFoundError(f"DIRECTOR_SYSTEM_PROMPT_PATH not found: {path}")
            return path

    return prompts_dir / f"{name}.txt"


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory (cached)."""
    prompt_path = _resolve_prompt_path(name)
    cache_key = f"{name}:{prompt_path.resolve()}"
    if cache_key not in _PROMPTS:
        _PROMPTS[cache_key] = prompt_path.read_text()
    return _PROMPTS[cache_key]


def _log(msg: str) -> None:
    print(f"[DIRECTOR] {msg}", file=sys.stderr, flush=True)


# ── Direct Anthropic API path ──────────────────────────────────────────────

ANTHROPIC_MODEL = os.getenv("DIRECTOR_ANTHROPIC_MODEL", "claude-sonnet-4-6")
REMOTION_SKILL_ID = os.getenv("REMOTION_SKILL_ID", "")


def generate_storyboard_api(
    input_text: str,
    *,
    model: str | None = None,
    skill_id: str | None = None,
) -> list[dict[str, Any]]:
    """Generate a storyboard with per-scene Remotion code via the Anthropic Messages API.

    Requires ANTHROPIC_API_KEY in the environment.
    Each scene includes narration + scene_code (Remotion React component body).

    Returns:
        List of scene dicts, each with "narration" and "scene_code".
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic  — required for API-based storyboard generation")

    client = anthropic.Anthropic()
    system_prompt = load_prompt("director_system")
    chosen_model = model or ANTHROPIC_MODEL

    betas: list[str] = []
    container: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    sid = skill_id or REMOTION_SKILL_ID

    if sid:
        betas = ["code-execution-2025-08-25", "skills-2025-10-02"]
        container = {
            "skills": [{"type": "custom", "skill_id": sid, "version": "latest"}],
        }
        tools = [{"type": "code_execution_20250825", "name": "code_execution"}]

    _log(f"Calling {chosen_model}" + (f" with skill {sid}" if sid else ""))

    kwargs: dict[str, Any] = {
        "model": chosen_model,
        "max_tokens": 128000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": input_text}],
    }
    if betas:
        kwargs["betas"] = betas
    if container:
        kwargs["container"] = container
    if tools:
        kwargs["tools"] = tools

    _log("Streaming response...")
    text = ""
    if betas:
        with client.beta.messages.stream(**kwargs) as stream:
            for event in stream:
                if hasattr(event, "type") and event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        text += event.delta.text
    else:
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if hasattr(event, "type") and event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        text += event.delta.text

    _log(f"Received {len(text)} chars of output")
    scenes = _parse_scenes_json(text)
    _log(f"Parsed {len(scenes)} scenes")
    return scenes


def _parse_scenes_json(raw: str) -> list[dict[str, Any]]:
    """Extract scene list from model output (handles both {scenes:[...]} and bare [...])."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "scenes" in parsed:
        return parsed["scenes"]
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"Unexpected storyboard JSON shape: {type(parsed)}")


def available_storyboard_runners() -> list[str]:
    """Return list of available local storyboard runner names."""
    try:
        from .local_planner import available_storyboard_runners as _available
        return _available()
    except Exception:
        return []


# ── Legacy local-runner path ───────────────────────────────────────────────

def generate_storyboard(
    input_text: str,
    provider: StoryboardProvider = "codex",
    *,
    project_dir: str | Path | None = None,
) -> list[dict]:
    """Generate a storyboard using either API or local agent runner."""
    if provider == "api":
        return generate_storyboard_api(input_text)

    from .local_planner import (
        generate_cathode_ready_storyboard,
        normalize_storyboard_runner,
    )
    system_prompt = load_prompt("director_system")
    return generate_cathode_ready_storyboard(
        input_text=input_text,
        system_prompt=system_prompt,
        runner=normalize_storyboard_runner(provider),
        project_dir=project_dir,
    )


def refine_prompt(
    original_prompt: str,
    feedback: str,
    narration: str = "",
    provider: StoryboardProvider = "codex",
    *,
    project_dir: str | Path | None = None,
) -> str:
    """Refine a composition prop or text via the local runner."""
    from .local_planner import (
        refine_text_with_local_runner,
        normalize_storyboard_runner,
    )
    system_prompt = load_prompt("refiner_system")
    return refine_text_with_local_runner(
        field_name="visual_prompt",
        original_text=original_prompt,
        feedback=feedback,
        narration=narration,
        system_prompt=system_prompt,
        runner=normalize_storyboard_runner(provider),
        project_dir=project_dir,
    )


def refine_narration(
    original_narration: str,
    feedback: str,
    provider: StoryboardProvider = "codex",
    *,
    project_dir: str | Path | None = None,
) -> str:
    """Refine narration via the local runner."""
    from .local_planner import (
        refine_text_with_local_runner,
        normalize_storyboard_runner,
    )
    system_prompt = load_prompt("refiner_narration_system")
    return refine_text_with_local_runner(
        field_name="narration",
        original_text=original_narration,
        feedback=feedback,
        system_prompt=system_prompt,
        runner=normalize_storyboard_runner(provider),
        project_dir=project_dir,
    )


__all__ = [
    "available_storyboard_runners",
    "generate_storyboard",
    "generate_storyboard_api",
    "load_prompt",
    "refine_narration",
    "refine_prompt",
]
