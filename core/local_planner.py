"""Local agent-runner planner for Cathode-ready qEEG scene JSON."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATHODE_ROOT = Path(
    os.getenv("CATHODE_REPO_PATH") or "/Users/davidmontgomery/cathode"
).expanduser()
DEFAULT_CLAUDE_BINARY = Path(
    os.getenv("CLAUDE_CODE_BINARY") or "/Users/davidmontgomery/.local/bin/claude"
).expanduser()
DEFAULT_CLAUDE_MODEL = (
    os.getenv("CLAUDE_LOCAL_PLANNER_MODEL") or "claude-opus-4-6[1m]"
)
DEFAULT_CLAUDE_TOOLS = (
    os.getenv("CLAUDE_LOCAL_PLANNER_TOOLS") or "Read,Grep,Glob"
)
DEFAULT_CODEX_BINARY = os.getenv("CODEX_BINARY") or "codex"

SUPPORTED_RUNNERS = ("codex", "claude")
LEGACY_RUNNER_ALIASES = {
    "openai": "codex",
    "anthropic": "claude",
}

ALLOWED_SCENE_TYPES = {"image", "video", "motion"}
ALLOWED_COMPOSITION_MODES = {"none", "overlay", "native"}
ALLOWED_FAMILIES = {
    "static_media",
    "media_pan",
    "software_demo_focus",
    "kinetic_statements",
    "kinetic_title",
    "bullet_stack",
    "quote_focus",
    "three_data_stage",
    "surreal_tableau_3d",
    "cover_hook",
    "orientation",
    "synthesis_summary",
    "closing_cta",
    "clinical_explanation",
    "metric_improvement",
    "brain_region_focus",
    "metric_comparison",
    "timeline_progression",
    "analogy_metaphor",
}
NATIVE_FAMILIES = {
    "cover_hook",
    "orientation",
    "synthesis_summary",
    "closing_cta",
    "clinical_explanation",
    "metric_improvement",
    "brain_region_focus",
    "metric_comparison",
    "timeline_progression",
    "analogy_metaphor",
    "three_data_stage",
    "kinetic_statements",
    "kinetic_title",
    "bullet_stack",
    "quote_focus",
    "software_demo_focus",
    "surreal_tableau_3d",
}


@dataclass(frozen=True)
class CathodeResourceMap:
    cathode_root: Path
    template_backgrounds_dir: Path
    text_zones_json: Path
    template_layout_map_ts: Path
    clinical_template_prompt: Path
    scene_family_contracts: Path
    remotion_architecture: Path
    codex_skill: Path
    claude_skill: Path
    quality_bar_paths: tuple[Path, ...]


def normalize_storyboard_runner(provider: str | None) -> str:
    value = str(provider or "").strip().lower()
    value = LEGACY_RUNNER_ALIASES.get(value, value)
    if value not in SUPPORTED_RUNNERS:
        raise ValueError(
            f"Unknown local storyboard runner: {provider!r}. "
            f"Choose one of {', '.join(SUPPORTED_RUNNERS)}."
        )
    return value


def _command_exists(command: str | Path) -> bool:
    if isinstance(command, Path):
        return command.exists()
    if os.sep in str(command):
        return Path(command).expanduser().exists()
    return shutil.which(str(command)) is not None


def available_storyboard_runners() -> list[str]:
    runners: list[str] = []
    if _command_exists(DEFAULT_CODEX_BINARY):
        runners.append("codex")
    if _command_exists(DEFAULT_CLAUDE_BINARY):
        runners.append("claude")
    return runners


def build_cathode_resource_map() -> CathodeResourceMap:
    cathode_root = DEFAULT_CATHODE_ROOT.resolve()
    if not cathode_root.exists():
        raise FileNotFoundError(f"Cathode repo not found at {cathode_root}")

    quality_candidates = [
        REPO_ROOT / "projects/09-05-1954-0/.v1-videos",
        REPO_ROOT / "projects/01-01-1983-0/final_video.mp4",
        cathode_root / "projects/07-14-2008-0/plan.json",
    ]
    quality_bar_paths = tuple(path for path in quality_candidates if path.exists())

    return CathodeResourceMap(
        cathode_root=cathode_root,
        template_backgrounds_dir=cathode_root / "template_deck/backgrounds",
        text_zones_json=cathode_root / "template_deck/text_zones.json",
        template_layout_map_ts=cathode_root / "frontend/src/remotion/templateLayoutMap.ts",
        clinical_template_prompt=cathode_root / "prompts/director_clinical_template_system_prompt.txt",
        scene_family_contracts=(
            cathode_root
            / "skills/cathode-remotion-development/references/scene-family-contracts.md"
        ),
        remotion_architecture=(
            cathode_root
            / "skills/cathode-remotion-development/references/cathode-remotion-architecture.md"
        ),
        codex_skill=Path("/Users/davidmontgomery/.codex/skills/remotion/SKILL.md"),
        claude_skill=Path("/Users/davidmontgomery/.claude/skills/remotion/SKILL.md"),
        quality_bar_paths=quality_bar_paths,
    )


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required planner context file: {path}")
    return path.read_text()


def _background_ids(resource_map: CathodeResourceMap) -> list[str]:
    return sorted(
        path.stem
        for path in resource_map.template_backgrounds_dir.glob("*.png")
        if path.is_file()
    )


def _storyboard_output_schema() -> dict[str, Any]:
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


def _refinement_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }


def build_storyboard_prompt(input_text: str, system_prompt: str) -> str:
    resource_map = build_cathode_resource_map()
    background_ids = _background_ids(resource_map)
    quality_refs = "\n".join(
        f"- {path}" for path in resource_map.quality_bar_paths
    ) or "- No local quality-bar reference files were found."

    return f"""{system_prompt}

You are not writing the old local-explainer-video scene schema anymore.
You are authoring the final Cathode-ready scene JSON directly.

This spike is qEEG-only and art-first.
Do not flatten the video into generic charts or generic data cards.
If a chart scene is appropriate, it still needs to feel beautiful, intentional, and emotionally legible.
One of the quality-bar references was an artistic Qwen-generated chart that worked extremely well.

Important planning rules for this spike:
- The returned plan is the truth. Do not rely on downstream code to reroute families, repair props, or guess backgrounds later.
- Use the real Cathode template families, background ids, text-zone/layout resources, and Remotion context below.
- Use tools to inspect those files directly if you need more detail.
- Use the installed `remotion` skill if helpful.
- Use web search if you truly need outside context, but the local files are the main source of truth.
- If Claude agent teams or Codex parallel work would help, you may use them, but the final answer must still be one JSON object only.

Return exactly one JSON object with a top-level `"scenes"` array and no markdown fences.

Required scene fields:
- `title`
- `narration`
- `scene_type`
- `visual_prompt`
- `on_screen_text`
- `composition.family`
- `composition.mode`
- `composition.props`

Rules for scene structure:
- `scene_type` must be one of: {sorted(ALLOWED_SCENE_TYPES)}
- `composition.mode` must be one of: {sorted(ALLOWED_COMPOSITION_MODES)}
- Allowed composition families: {sorted(ALLOWED_FAMILIES)}
- For native/template scenes, use `scene_type: "motion"` and `composition.mode: "native"`.
- For native/template scenes, set `composition.manifestation` to `"native_remotion"`.
- For native/template scenes, `visual_prompt` should be `null` or an empty string.
- For image scenes, `visual_prompt` must be a complete standalone Qwen prompt.
- Keep `on_screen_text` aligned with what is actually visible in the scene.
- Use exact Cathode `background_id` values only from the background catalog below.

Do not recreate Cathode's current failure mode:
- no family rerouting heuristics
- no background auto-picking later
- no prop enrichers fixing missing structure after the fact
- no chart monoculture

Clinic note that applies only to chart choice, not aesthetic ambition:
"I agree with line graphs with target range highlights. Bar graphs can do the same job because there are only 3 timepoints. As we go along and encourage people to get more treatment and more qEEGs, then line graphs will have more scientific validity."

Interpret that note correctly:
- it does NOT lower the art bar
- it only means bar vs line is flexible for three timepoints
- reference/target range should remain visible when chart scenes are used

Local explainer-video source text:
--- BEGIN INPUT ---
{input_text}
--- END INPUT ---

Absolute Cathode resource paths to inspect:
- Cathode repo root: {resource_map.cathode_root}
- Template backgrounds: {resource_map.template_backgrounds_dir}
- Template text zones: {resource_map.text_zones_json}
- Template layout map: {resource_map.template_layout_map_ts}
- Clinical template prompt: {resource_map.clinical_template_prompt}
- Scene family contracts: {resource_map.scene_family_contracts}
- Remotion architecture notes: {resource_map.remotion_architecture}
- Codex Remotion skill: {resource_map.codex_skill}
- Claude Remotion skill: {resource_map.claude_skill}

Quality-bar reference paths:
{quality_refs}

Available Cathode background ids:
{json.dumps(background_ids, indent=2)}

Cathode scene family contracts:
--- BEGIN SCENE FAMILY CONTRACTS ---
{_read_text(resource_map.scene_family_contracts)}
--- END SCENE FAMILY CONTRACTS ---

Cathode clinical template prompt:
--- BEGIN CLINICAL TEMPLATE PROMPT ---
{_read_text(resource_map.clinical_template_prompt)}
--- END CLINICAL TEMPLATE PROMPT ---

Cathode Remotion architecture notes:
--- BEGIN REMOTION ARCHITECTURE ---
{_read_text(resource_map.remotion_architecture)}
--- END REMOTION ARCHITECTURE ---

Cathode text zones:
--- BEGIN TEXT ZONES JSON ---
{_read_text(resource_map.text_zones_json)}
--- END TEXT ZONES JSON ---

Cathode template layout map:
--- BEGIN TEMPLATE LAYOUT MAP ---
{_read_text(resource_map.template_layout_map_ts)}
--- END TEMPLATE LAYOUT MAP ---
"""


def build_refinement_prompt(
    *,
    system_prompt: str,
    field_name: str,
    original_text: str,
    feedback: str,
    narration: str = "",
) -> str:
    narration_block = (
        f"\nScene narration for context:\n{narration.strip()}\n"
        if narration.strip()
        else ""
    )
    return f"""{system_prompt}

You are running inside a local agent CLI, not an API wrapper.
Return exactly one JSON object with a single `text` field and no markdown fences.

Field being refined: {field_name}
Original text:
{original_text}
{narration_block}
User feedback:
{feedback}
"""


def generate_cathode_ready_storyboard(
    *,
    input_text: str,
    system_prompt: str,
    runner: str,
    project_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    prompt = build_storyboard_prompt(input_text, system_prompt)
    result = run_structured_local_prompt(
        prompt=prompt,
        runner=runner,
        schema=_storyboard_output_schema(),
        project_dir=project_dir,
        artifact_stem="storyboard",
    )
    return validate_cathode_ready_scenes(result)


def refine_text_with_local_runner(
    *,
    field_name: str,
    original_text: str,
    feedback: str,
    system_prompt: str,
    runner: str,
    narration: str = "",
    project_dir: str | Path | None = None,
) -> str:
    prompt = build_refinement_prompt(
        system_prompt=system_prompt,
        field_name=field_name,
        original_text=original_text,
        feedback=feedback,
        narration=narration,
    )
    result = run_structured_local_prompt(
        prompt=prompt,
        runner=runner,
        schema=_refinement_output_schema(),
        project_dir=project_dir,
        artifact_stem=f"refine_{field_name}",
    )
    text = str((result or {}).get("text") or "").strip()
    if not text:
        raise ValueError(f"{field_name} refinement returned empty text")
    return text


def run_structured_local_prompt(
    *,
    prompt: str,
    runner: str,
    schema: dict[str, Any],
    project_dir: str | Path | None = None,
    artifact_stem: str = "planner",
) -> dict[str, Any]:
    normalized_runner = normalize_storyboard_runner(runner)
    artifact_dir = _ensure_artifact_dir(project_dir)
    schema_json = json.dumps(schema, indent=2)

    _write_artifact(artifact_dir, f"{artifact_stem}_prompt.txt", prompt)
    _write_artifact(artifact_dir, f"{artifact_stem}_schema.json", schema_json)

    if normalized_runner == "codex":
        return _run_codex_prompt(
            prompt=prompt,
            schema_json=schema_json,
            artifact_dir=artifact_dir,
            artifact_stem=artifact_stem,
        )
    return _run_claude_prompt(
        prompt=prompt,
        schema_json=schema_json,
        artifact_dir=artifact_dir,
        artifact_stem=artifact_stem,
    )


def build_codex_command(schema_path: Path, output_path: Path, cathode_root: Path) -> list[str]:
    return [
        DEFAULT_CODEX_BINARY,
        "--search",
        "-a",
        "never",
        "-s",
        "read-only",
        "exec",
        "-C",
        str(REPO_ROOT),
        "--add-dir",
        str(cathode_root),
        "--skip-git-repo-check",
        "--ephemeral",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
    ]


def build_claude_command(schema_json: str, cathode_root: Path) -> list[str]:
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
        str(cathode_root),
        "--dangerously-skip-permissions",
        "--no-session-persistence",
    ]


def _run_codex_prompt(
    *,
    prompt: str,
    schema_json: str,
    artifact_dir: Path | None,
    artifact_stem: str,
) -> dict[str, Any]:
    cathode_root = build_cathode_resource_map().cathode_root
    with tempfile.TemporaryDirectory(prefix="lev_codex_planner_") as tmpdir:
        tmp_path = Path(tmpdir)
        schema_path = tmp_path / "schema.json"
        output_path = tmp_path / "last_message.json"
        schema_path.write_text(schema_json)

        command = build_codex_command(schema_path, output_path, cathode_root)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
        )

        _write_artifact(artifact_dir, f"{artifact_stem}_codex_stdout.log", completed.stdout)
        _write_artifact(artifact_dir, f"{artifact_stem}_codex_stderr.log", completed.stderr)
        _write_artifact(
            artifact_dir,
            f"{artifact_stem}_codex_command.txt",
            " ".join(command),
        )

        raw_output = output_path.read_text().strip() if output_path.exists() else ""
        if not raw_output and completed.returncode == 0:
            raise ValueError("Codex returned success but no structured output file was written")
        if completed.returncode != 0 and not raw_output:
            raise RuntimeError(
                f"Codex runner failed with exit code {completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()}"
            )

        _write_artifact(artifact_dir, f"{artifact_stem}_codex_output.json", raw_output)
        return _parse_json_object(raw_output, runner="codex")


def _run_claude_prompt(
    *,
    prompt: str,
    schema_json: str,
    artifact_dir: Path | None,
    artifact_stem: str,
) -> dict[str, Any]:
    cathode_root = build_cathode_resource_map().cathode_root
    command = build_claude_command(schema_json, cathode_root)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
    )

    _write_artifact(artifact_dir, f"{artifact_stem}_claude_stdout.json", completed.stdout)
    _write_artifact(artifact_dir, f"{artifact_stem}_claude_stderr.log", completed.stderr)
    _write_artifact(
        artifact_dir,
        f"{artifact_stem}_claude_command.txt",
        " ".join(command),
    )

    payload = _parse_json_object(completed.stdout.strip(), runner="claude")
    if completed.returncode != 0 or payload.get("is_error"):
        message = str(payload.get("result") or completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Claude runner failed: {message}")

    structured = payload.get("structured_output")
    if not isinstance(structured, dict):
        raise ValueError("Claude runner did not return structured_output")
    _write_artifact(
        artifact_dir,
        f"{artifact_stem}_claude_structured_output.json",
        json.dumps(structured, indent=2),
    )
    return structured


def _parse_json_object(raw_text: str, *, runner: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError(f"{runner} returned empty output")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{runner} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{runner} returned JSON that is not an object")
    return payload


def validate_cathode_ready_scenes(raw_payload: Any) -> list[dict[str, Any]]:
    if isinstance(raw_payload, dict):
        raw_scenes = raw_payload.get("scenes")
    else:
        raw_scenes = raw_payload
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("Planner output must contain a non-empty scenes array")

    allowed_background_ids = set(_background_ids(build_cathode_resource_map()))
    validated: list[dict[str, Any]] = []

    for index, raw_scene in enumerate(raw_scenes):
        if not isinstance(raw_scene, dict):
            raise ValueError(f"Scene {index + 1} is not an object")

        scene = dict(raw_scene)
        title = str(scene.get("title") or f"Scene {index + 1}").strip()
        narration = str(scene.get("narration") or "").strip()
        if not narration:
            raise ValueError(f"Scene {index + 1} has empty narration")

        composition = scene.get("composition")
        if composition is not None and not isinstance(composition, dict):
            raise ValueError(f"Scene {index + 1} composition must be an object")

        scene_type = str(scene.get("scene_type") or "").strip().lower()
        if not scene_type and isinstance(composition, dict):
            family = str(composition.get("family") or "").strip()
            mode = str(composition.get("mode") or "").strip().lower()
            if family in NATIVE_FAMILIES or mode == "native":
                scene_type = "motion"
        if not scene_type:
            scene_type = "image"
        if scene_type not in ALLOWED_SCENE_TYPES:
            raise ValueError(f"Scene {index + 1} has unsupported scene_type {scene_type!r}")

        visual_prompt = scene.get("visual_prompt")
        if visual_prompt in (None, ""):
            visual_prompt_text = ""
        else:
            visual_prompt_text = str(visual_prompt).strip()
        if scene_type != "motion" and not visual_prompt_text:
            raise ValueError(f"Scene {index + 1} needs a visual_prompt for scene_type={scene_type}")

        on_screen_text = scene.get("on_screen_text")
        if isinstance(on_screen_text, list):
            normalized_text = [str(item).strip() for item in on_screen_text if str(item).strip()]
        else:
            normalized_text = []

        normalized_composition = None
        if isinstance(composition, dict):
            normalized_composition = _normalize_composition(
                composition,
                scene_index=index,
                allowed_background_ids=allowed_background_ids,
            )
        if scene_type == "motion" and not normalized_composition:
            raise ValueError(f"Scene {index + 1} is motion but has no composition object")

        scene["id"] = scene.get("id", index)
        scene["uid"] = str(scene.get("uid") or uuid.uuid4())[:8]
        scene["title"] = title
        scene["narration"] = narration
        scene["scene_type"] = scene_type
        scene["visual_prompt"] = visual_prompt_text
        scene["on_screen_text"] = normalized_text
        scene["refinement_history"] = (
            scene.get("refinement_history")
            if isinstance(scene.get("refinement_history"), list)
            else []
        )

        if normalized_composition is not None:
            scene["composition"] = normalized_composition
            if scene_type == "motion":
                scene["motion"] = {
                    "template_id": normalized_composition["family"],
                    "props": normalized_composition["props"],
                    "render_path": normalized_composition.get("render_path"),
                    "preview_path": normalized_composition.get("preview_path"),
                    "rationale": normalized_composition.get("rationale", ""),
                }

        validated.append(scene)

    return validated


def _normalize_composition(
    composition: dict[str, Any],
    *,
    scene_index: int,
    allowed_background_ids: set[str],
) -> dict[str, Any]:
    family = str(composition.get("family") or "").strip()
    if not family:
        raise ValueError(f"Scene {scene_index + 1} composition is missing family")
    if family not in ALLOWED_FAMILIES:
        raise ValueError(f"Scene {scene_index + 1} uses unsupported family {family!r}")

    mode = str(composition.get("mode") or ("native" if family in NATIVE_FAMILIES else "none")).strip().lower()
    if mode not in ALLOWED_COMPOSITION_MODES:
        raise ValueError(f"Scene {scene_index + 1} composition has unsupported mode {mode!r}")

    props = composition.get("props")
    if props is None:
        props = {}
    if not isinstance(props, dict):
        raise ValueError(f"Scene {scene_index + 1} composition props must be an object")

    background_id = str(props.get("background_id") or "").strip()
    if background_id and background_id not in allowed_background_ids:
        raise ValueError(
            f"Scene {scene_index + 1} uses unknown Cathode background_id {background_id!r}"
        )

    transition_after = composition.get("transition_after")
    if transition_after is not None and not isinstance(transition_after, dict):
        raise ValueError(
            f"Scene {scene_index + 1} transition_after must be an object or null"
        )

    data = composition.get("data")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Scene {scene_index + 1} composition data must be an object")

    normalized = dict(composition)
    normalized["family"] = family
    normalized["mode"] = mode
    normalized["props"] = props
    normalized["data"] = data
    normalized["transition_after"] = transition_after
    if mode == "native" and not normalized.get("manifestation"):
        normalized["manifestation"] = "native_remotion"
    if "rationale" in normalized:
        normalized["rationale"] = str(normalized.get("rationale") or "").strip()
    return normalized


def _ensure_artifact_dir(project_dir: str | Path | None) -> Path | None:
    if not project_dir:
        return None
    artifact_dir = Path(project_dir) / "planner_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _write_artifact(artifact_dir: Path | None, name: str, content: str) -> None:
    if artifact_dir is None:
        return
    artifact_path = artifact_dir / name
    artifact_path.write_text(content)
