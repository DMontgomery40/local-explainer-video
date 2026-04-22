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

from dotenv import load_dotenv

_PROMPTS: dict[str, str] = {}
REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

StoryboardProvider = Literal["codex", "claude", "openai", "anthropic", "api"]

REMOTION_SKILL_ID_ENV_NAMES = (
    "REMOTION_SKILL_ID",
    "DIRECTOR_REMOTION_SKILL_ID",
    "ANTHROPIC_REMOTION_SKILL_ID",
)
REMOTION_SKILL_VERSION_ENV_NAMES = (
    "REMOTION_SKILL_VERSION",
    "DIRECTOR_REMOTION_SKILL_VERSION",
    "ANTHROPIC_REMOTION_SKILL_VERSION",
)
DIRECTOR_ANTHROPIC_MAX_TOKENS_ENV_NAMES = (
    "DIRECTOR_ANTHROPIC_MAX_TOKENS",
    "ANTHROPIC_MAX_TOKENS",
)
DIRECTOR_ANTHROPIC_EFFORT_ENV_NAMES = (
    "DIRECTOR_ANTHROPIC_EFFORT",
    "ANTHROPIC_EFFORT",
)
DIRECTOR_ANTHROPIC_THINKING_TYPE_ENV_NAMES = (
    "DIRECTOR_ANTHROPIC_THINKING_TYPE",
    "ANTHROPIC_THINKING_TYPE",
)
DIRECTOR_ANTHROPIC_SPEED_ENV_NAMES = (
    "DIRECTOR_ANTHROPIC_SPEED",
    "ANTHROPIC_SPEED",
)
DIRECTOR_ANTHROPIC_ENABLE_COMPACTION_ENV_NAMES = (
    "DIRECTOR_ANTHROPIC_ENABLE_COMPACTION",
    "ANTHROPIC_ENABLE_COMPACTION",
)
DIRECTOR_ANTHROPIC_BATCH_MAX_TOKENS_ENV_NAMES = (
    "DIRECTOR_ANTHROPIC_BATCH_MAX_TOKENS",
    "ANTHROPIC_BATCH_MAX_TOKENS",
)
DIRECTOR_REFERENCE_IMAGE_FILE_IDS_ENV_NAMES = (
    "DIRECTOR_REFERENCE_IMAGE_FILE_IDS",
    "DIRECTOR_CONTACT_SHEET_FILE_IDS",
)


def _resolve_prompt_path(name: str) -> Path:
    """Resolve prompt path, supporting versioned director_system experiments."""
    prompts_dir = REPO_ROOT / "prompts"

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
                path = REPO_ROOT / override_path
            if not path.exists():
                raise FileNotFoundError(f"DIRECTOR_SYSTEM_PROMPT_PATH not found: {path}")
            return path

    return prompts_dir / f"{name}.txt"


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory (cached)."""
    prompt_path = _resolve_prompt_path(name)
    cache_key = f"{name}:{prompt_path.resolve()}"
    if cache_key not in _PROMPTS:
        text = prompt_path.read_text()
        if name == "director_system":
            reference_plan = (REPO_ROOT / "prompts" / "reference_successful_plan_04-08-1997-0.json").read_text()
            text = text.replace("{{REFERENCE_SUCCESSFUL_PLAN_04_08_1997_0}}", reference_plan)
        _PROMPTS[cache_key] = text
    return _PROMPTS[cache_key]


def _log(msg: str) -> None:
    print(f"[DIRECTOR] {msg}", file=sys.stderr, flush=True)


# ── Direct Anthropic API path ──────────────────────────────────────────────

ANTHROPIC_MODEL = os.getenv("DIRECTOR_ANTHROPIC_MODEL", "claude-sonnet-4-6")
FILES_API_BETA = "files-api-2025-04-14"
FAST_MODE_BETA = "fast-mode-2026-02-01"
MESSAGE_BATCHES_BETA = "message-batches-2024-09-24"
OUTPUT_300K_BETA = "output-300k-2026-03-24"
COMPACTION_BETA = "compact-2026-01-12"
DEFAULT_THINKING_TYPE = "adaptive"
DEFAULT_EFFORT = "high"


def _env_first(*names: str) -> str:
    """Return the first non-empty environment variable among the given names."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _env_int(*names: str) -> int | None:
    value = _env_first(*names)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Expected integer for one of {names}, got {value!r}")


def _env_bool(*names: str) -> bool | None:
    value = _env_first(*names)
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean for one of {names}, got {value!r}")


def resolve_remotion_skill_config(
    *,
    skill_id: str | None = None,
    skill_version: str | None = None,
) -> tuple[str | None, str]:
    """Resolve the Anthropic custom skill id/version for the Remotion skill."""
    resolved_id = (skill_id or _env_first(*REMOTION_SKILL_ID_ENV_NAMES)).strip() or None
    resolved_version = (skill_version or _env_first(*REMOTION_SKILL_VERSION_ENV_NAMES)).strip() or "latest"
    return resolved_id, resolved_version


def _model_output_cap(model: str) -> int:
    normalized = model.strip().lower()
    if normalized == "claude-opus-4-6":
        return 128000
    if normalized == "claude-sonnet-4-6":
        return 64000
    return 16000


def _normalize_effort_for_model(model: str, effort: str) -> str:
    normalized_model = model.strip().lower()
    normalized_effort = (effort or DEFAULT_EFFORT).strip().lower()
    allowed = {"low", "medium", "high", "max"}
    if normalized_effort not in allowed:
        raise ValueError(f"Unsupported Anthropic effort {effort!r}")
    if normalized_effort == "max" and normalized_model != "claude-opus-4-6":
        _log(
            f"Requested effort='max' for {model}; Anthropic limits max effort to Claude Opus 4.6. "
            "Falling back to effort='high'."
        )
        return "high"
    return normalized_effort


def resolve_anthropic_request_config(
    *,
    model: str,
    max_tokens: int | None = None,
    effort: str | None = None,
    thinking_type: str | None = None,
    speed: str | None = None,
    enable_compaction: bool | None = None,
) -> dict[str, Any]:
    """Resolve request config for Claude 4.6 family features."""
    resolved_max_tokens = max_tokens or _env_int(*DIRECTOR_ANTHROPIC_MAX_TOKENS_ENV_NAMES) or _model_output_cap(model)
    output_cap = _model_output_cap(model)
    if resolved_max_tokens > output_cap:
        _log(
            f"Requested max_tokens={resolved_max_tokens} for {model}, but the documented cap is {output_cap}. "
            f"Clamping to {output_cap}."
        )
        resolved_max_tokens = output_cap

    resolved_thinking_type = (thinking_type or _env_first(*DIRECTOR_ANTHROPIC_THINKING_TYPE_ENV_NAMES) or DEFAULT_THINKING_TYPE).strip().lower()
    if resolved_thinking_type not in {"adaptive", "disabled"}:
        raise ValueError(f"Unsupported Anthropic thinking type {resolved_thinking_type!r}")

    resolved_effort = _normalize_effort_for_model(
        model,
        effort or _env_first(*DIRECTOR_ANTHROPIC_EFFORT_ENV_NAMES) or DEFAULT_EFFORT,
    )
    resolved_speed = (speed or _env_first(*DIRECTOR_ANTHROPIC_SPEED_ENV_NAMES)).strip().lower()
    resolved_enable_compaction = (
        enable_compaction
        if enable_compaction is not None
        else _env_bool(*DIRECTOR_ANTHROPIC_ENABLE_COMPACTION_ENV_NAMES)
    )
    if resolved_enable_compaction is None:
        resolved_enable_compaction = True

    request: dict[str, Any] = {
        "max_tokens": resolved_max_tokens,
        "output_config": {"effort": resolved_effort},
    }
    if resolved_thinking_type == "adaptive":
        request["thinking"] = {"type": "adaptive"}
    if resolved_speed:
        if resolved_speed != "fast":
            raise ValueError(f"Unsupported Anthropic speed {resolved_speed!r}")
        if model.strip().lower() != "claude-opus-4-6":
            _log(f"Ignoring speed='fast' for {model}; fast mode is Opus-only.")
        else:
            request["speed"] = "fast"
            request.setdefault("extra_betas", []).append(FAST_MODE_BETA)
    if resolved_enable_compaction:
        request.setdefault("extra_betas", []).append(COMPACTION_BETA)
    return request


def resolve_director_reference_file_ids(
    *,
    image_file_ids: list[str] | None = None,
) -> list[str]:
    """Resolve always-attached reference image file ids for the director prompt."""
    if image_file_ids is None:
        resolved_images = _split_csv(_env_first(*DIRECTOR_REFERENCE_IMAGE_FILE_IDS_ENV_NAMES))
    else:
        resolved_images = image_file_ids
    return _unique(resolved_images)


def _build_storyboard_user_content(
    input_text: str,
    *,
    image_file_ids: list[str],
) -> str | list[dict[str, Any]]:
    if not image_file_ids:
        return input_text

    content: list[dict[str, Any]] = []
    for file_id in image_file_ids:
        content.append(
            {
                "type": "image",
                "source": {"type": "file", "file_id": file_id},
            }
        )
    content.append({"type": "text", "text": input_text})
    return content


def build_storyboard_api_kwargs(
    input_text: str,
    *,
    system_prompt: str,
    model: str,
    skill_id: str | None = None,
    skill_version: str | None = None,
    max_tokens: int | None = None,
    effort: str | None = None,
    thinking_type: str | None = None,
    speed: str | None = None,
    enable_compaction: bool | None = None,
    reference_image_file_ids: list[str] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Build Anthropic Messages API kwargs for storyboard generation."""
    resolved_skill_id, resolved_skill_version = resolve_remotion_skill_config(
        skill_id=skill_id,
        skill_version=skill_version,
    )
    resolved_reference_images = resolve_director_reference_file_ids(
        image_file_ids=reference_image_file_ids,
    )
    user_content = _build_storyboard_user_content(
        input_text,
        image_file_ids=resolved_reference_images,
    )
    request_config = resolve_anthropic_request_config(
        model=model,
        max_tokens=max_tokens,
        effort=effort,
        thinking_type=thinking_type,
        speed=speed,
        enable_compaction=enable_compaction,
    )
    extra_betas = list(request_config.pop("extra_betas", []))

    kwargs: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
        **request_config,
    }
    betas: list[str] = []
    betas.extend(extra_betas)
    if resolved_reference_images:
        betas.append(FILES_API_BETA)
    if resolved_skill_id:
        betas.extend(["code-execution-2025-08-25", "skills-2025-10-02"])
        kwargs["container"] = {
            "skills": [
                {
                    "type": "custom",
                    "skill_id": resolved_skill_id,
                    "version": resolved_skill_version,
                }
            ],
        }
        kwargs["tools"] = [{"type": "code_execution_20250825", "name": "code_execution"}]
    if betas:
        kwargs["betas"] = _unique(betas)

    return kwargs, resolved_skill_id


def build_storyboard_batch_request(
    input_text: str,
    *,
    system_prompt: str,
    model: str,
    custom_id: str,
    skill_id: str | None = None,
    skill_version: str | None = None,
    max_tokens: int | None = None,
    effort: str | None = None,
    thinking_type: str | None = None,
    speed: str | None = None,
    reference_image_file_ids: list[str] | None = None,
    enable_compaction: bool | None = None,
) -> dict[str, Any]:
    """Build a single Message Batches API request with optional 300k output beta."""
    batch_max_tokens = max_tokens or _env_int(*DIRECTOR_ANTHROPIC_BATCH_MAX_TOKENS_ENV_NAMES) or 300000
    kwargs, _ = build_storyboard_api_kwargs(
        input_text,
        system_prompt=system_prompt,
        model=model,
        skill_id=skill_id,
        skill_version=skill_version,
        effort=effort,
        thinking_type=thinking_type,
        speed=speed,
        enable_compaction=enable_compaction,
        reference_image_file_ids=reference_image_file_ids,
    )
    betas = _unique(list(kwargs.get("betas", [])) + [MESSAGE_BATCHES_BETA, OUTPUT_300K_BETA])
    kwargs["betas"] = betas
    kwargs["max_tokens"] = min(batch_max_tokens, 300000)
    return {
        "custom_id": custom_id,
        "params": kwargs,
    }


def generate_storyboard_api(
    input_text: str,
    *,
    model: str | None = None,
    skill_id: str | None = None,
    skill_version: str | None = None,
    max_tokens: int | None = None,
    effort: str | None = None,
    thinking_type: str | None = None,
    speed: str | None = None,
    enable_compaction: bool | None = None,
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
    reference_image_file_ids = resolve_director_reference_file_ids()
    kwargs, resolved_skill_id = build_storyboard_api_kwargs(
        input_text,
        system_prompt=system_prompt,
        model=chosen_model,
        skill_id=skill_id,
        skill_version=skill_version,
        max_tokens=max_tokens,
        effort=effort,
        thinking_type=thinking_type,
        speed=speed,
        enable_compaction=enable_compaction,
        reference_image_file_ids=reference_image_file_ids,
    )

    if resolved_skill_id:
        _log(f"Calling {chosen_model} with remotion skill {resolved_skill_id}")
    else:
        _log(
            "Calling "
            f"{chosen_model} without a remotion skill. "
            f"Set one of {', '.join(REMOTION_SKILL_ID_ENV_NAMES)} to attach your Anthropic custom skill."
        )
    if reference_image_file_ids:
        _log(
            "Attaching "
            f"{len(reference_image_file_ids)} reference image(s) to every request"
        )

    _log("Streaming response...")
    text = ""
    if "betas" in kwargs:
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


def resolve_refinement_runner(provider: StoryboardProvider | str | None) -> str:
    """Route refinement to a local runner, preferring Codex for API-created plans."""
    from .local_planner import (
        available_storyboard_runners as _available_storyboard_runners,
        normalize_storyboard_runner,
    )

    normalized = str(provider or "").strip().lower()
    if normalized != "api":
        return normalize_storyboard_runner(provider)

    available = _available_storyboard_runners()
    for candidate in ("codex", "claude"):
        if candidate in available:
            return candidate

    raise ValueError(
        "Refinement for API-created projects requires a local runner, but neither "
        "`codex` nor `claude` is available on this machine."
    )


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
    from .local_planner import refine_text_with_local_runner
    system_prompt = load_prompt("refiner_system")
    return refine_text_with_local_runner(
        field_name="visual_prompt",
        original_text=original_prompt,
        feedback=feedback,
        narration=narration,
        system_prompt=system_prompt,
        runner=resolve_refinement_runner(provider),
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
    from .local_planner import refine_text_with_local_runner
    system_prompt = load_prompt("refiner_narration_system")
    return refine_text_with_local_runner(
        field_name="narration",
        original_text=original_narration,
        feedback=feedback,
        system_prompt=system_prompt,
        runner=resolve_refinement_runner(provider),
        project_dir=project_dir,
    )


__all__ = [
    "available_storyboard_runners",
    "build_storyboard_api_kwargs",
    "build_storyboard_batch_request",
    "generate_storyboard",
    "generate_storyboard_api",
    "load_prompt",
    "refine_narration",
    "refine_prompt",
    "resolve_refinement_runner",
    "resolve_anthropic_request_config",
    "resolve_director_reference_file_ids",
    "resolve_remotion_skill_config",
]
