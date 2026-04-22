"""Separate local Claude Code print-mode storyboard runner."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .director import REPO_ROOT, _model_output_cap, _parse_scenes_json, load_prompt

load_dotenv(REPO_ROOT / ".env", override=False)

DEFAULT_CLAUDE_BINARY = Path(
    os.getenv("CLAUDE_CODE_BINARY") or "/Users/davidmontgomery/.local/bin/claude"
).expanduser()
DEFAULT_CLAUDE_MODEL = (
    os.getenv("CLAUDE_LOCAL_REMOTION_MODEL") or "claude-opus-4-6"
)
DEFAULT_CLAUDE_TOOLS = (
    os.getenv("CLAUDE_LOCAL_REMOTION_TOOLS") or "Read,Grep,Glob,Bash"
)
CLAUDE_LOCAL_MAX_OUTPUT_TOKENS_ENV_NAMES = (
    "CLAUDE_LOCAL_REMOTION_MAX_OUTPUT_TOKENS",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "DIRECTOR_ANTHROPIC_MAX_TOKENS",
)
CLAUDE_LOCAL_USE_API_KEY_ENV_NAMES = (
    "CLAUDE_LOCAL_REMOTION_USE_API_KEY",
    "CLAUDE_CODE_USE_API_KEY",
)


def _env_first_int(*names: str) -> int | None:
    for name in names:
        value = os.getenv(name)
        if not value:
            continue
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc
    return None


def _env_flag(*names: str) -> bool | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Environment variable {name} must be boolean-like, got {value!r}")
    return None


def resolve_claude_code_max_output_tokens(model: str = DEFAULT_CLAUDE_MODEL) -> int:
    configured = _env_first_int(*CLAUDE_LOCAL_MAX_OUTPUT_TOKENS_ENV_NAMES)
    cap = _model_output_cap(model)
    if configured is None:
        return cap
    return min(configured, cap)


def local_reference_contact_sheet_paths() -> list[Path]:
    base = REPO_ROOT / "prompts" / "reference_contact_sheets"
    candidates = [
        base / "04-08-1997-0.png",
        base / "01-19-1966-0.png",
        base / "09-05-1954-0__integration_gate_20260222_073928.png",
    ]
    return [path for path in candidates if path.exists()]


def build_local_storyboard_prompt(input_text: str) -> str:
    system_prompt = load_prompt("director_system")
    sheet_paths = local_reference_contact_sheet_paths()
    contact_sheet_block = "\n".join(f"- {path}" for path in sheet_paths) or "- No local contact sheets found."

    return f"""{system_prompt}

LOCAL CLAUDE CODE PRINT-MODE CONTEXT:
- You are running inside Claude Code print mode via `claude -p`, not the Anthropic Messages API.
- The successful reference qEEG plan is already embedded above in the system prompt text.
- Local reference contact sheets are available at these absolute paths:
{contact_sheet_block}
- If you need to understand the contact sheets more concretely, you may inspect local files and use tools.
- Return exactly one JSON object with a top-level `"scenes"` array and no markdown fences.

SOURCE TEXT:
--- BEGIN INPUT ---
{input_text}
--- END INPUT ---
"""


def storyboard_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "array",
                "items": {"type": "object"},
            }
        },
        "required": ["scenes"],
        "additionalProperties": False,
    }


def build_claude_print_command(schema_json: str) -> list[str]:
    return [
        str(DEFAULT_CLAUDE_BINARY),
        "-p",
        "--model",
        DEFAULT_CLAUDE_MODEL,
        "--output-format",
        "json",
        "--json-schema",
        schema_json,
        "--tools",
        DEFAULT_CLAUDE_TOOLS,
        "--add-dir",
        str(REPO_ROOT),
        "--dangerously-skip-permissions",
        "--no-session-persistence",
    ]


def build_claude_print_env(model: str = DEFAULT_CLAUDE_MODEL) -> dict[str, str]:
    env = os.environ.copy()
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(resolve_claude_code_max_output_tokens(model))
    use_api_key = _env_flag(*CLAUDE_LOCAL_USE_API_KEY_ENV_NAMES)
    if not use_api_key:
        env.pop("ANTHROPIC_API_KEY", None)
    return env


def generate_storyboard_claude_local(
    input_text: str,
    *,
    project_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    prompt = build_local_storyboard_prompt(input_text)
    schema_json = json.dumps(storyboard_output_schema(), indent=2)
    command = build_claude_print_command(schema_json)
    env = build_claude_print_env(DEFAULT_CLAUDE_MODEL)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
    )

    if project_dir:
        artifact_dir = Path(project_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "claude_local_storyboard_prompt.txt").write_text(prompt, encoding="utf-8")
        (artifact_dir / "claude_local_storyboard_command.txt").write_text(
            shlex.join(command),
            encoding="utf-8",
        )
        (artifact_dir / "claude_local_storyboard_env.txt").write_text(
            f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={env['CLAUDE_CODE_MAX_OUTPUT_TOKENS']}\n",
            encoding="utf-8",
        )
        (artifact_dir / "claude_local_storyboard_stdout.json").write_text(completed.stdout, encoding="utf-8")
        (artifact_dir / "claude_local_storyboard_stderr.log").write_text(completed.stderr, encoding="utf-8")

    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(
            "Local Claude storyboard runner returned no JSON output. "
            "Inspect claude_local_storyboard_stderr.log for details."
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Local Claude storyboard runner returned invalid JSON output. "
            "Inspect claude_local_storyboard_stdout.json for details."
        ) from exc

    if completed.returncode != 0 or payload.get("is_error"):
        message = str(payload.get("result") or completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Local Claude storyboard runner failed: {message}")

    structured = payload.get("structured_output")
    if not isinstance(structured, dict):
        raise ValueError("Claude local runner did not return structured_output")

    scenes = structured.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Claude local runner returned no scenes array")
    return _parse_scenes_json(json.dumps(structured))


__all__ = [
    "build_claude_print_command",
    "build_claude_print_env",
    "build_local_storyboard_prompt",
    "generate_storyboard_claude_local",
    "local_reference_contact_sheet_paths",
    "resolve_claude_code_max_output_tokens",
    "storyboard_output_schema",
]
