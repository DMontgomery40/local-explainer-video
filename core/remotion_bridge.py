"""Bridge between Python pipeline and Remotion renderer.

Reads plan.json scenes, writes per-scene props, calls `npx remotion render`
for each scene, and collects the rendered MP4/PNG outputs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REMOTION_DIR = Path(__file__).resolve().parent.parent / "remotion"
FPS = 30
WIDTH = 1664
HEIGHT = 928

VALID_FAMILIES = {
    "cover_hook",
    "brain_region_focus",
    "metric_card",
    "metric_comparison",
    "timeline_progression",
    "bullet_stack",
    "data_stage",
    "analogy_split",
    "closing_cta",
    "narration_slide",
}


def _to_remotion_id(family: str) -> str:
    """Convert snake_case family name to kebab-case Remotion composition ID."""
    return family.replace("_", "-")


def _log(msg: str) -> None:
    print(f"[REMOTION] {msg}", file=sys.stderr, flush=True)


def _find_node() -> str:
    """Locate Node.js binary."""
    for candidate in ("/opt/homebrew/bin/node", "node"):
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError("Node.js not found — install Node 18+ to use Remotion rendering")


def _find_npx() -> str:
    """Locate npx binary."""
    for candidate in ("/opt/homebrew/bin/npx", "npx"):
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError("npx not found — install Node 18+ to use Remotion rendering")


def _audio_duration_seconds(audio_path: Path) -> float | None:
    """Get audio duration via ffprobe, or None if unavailable."""
    ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def duration_frames(audio_path: Path | None, default_seconds: float = 5.0) -> int:
    """Compute scene duration in frames from audio, with fallback."""
    if audio_path and audio_path.exists():
        dur = _audio_duration_seconds(audio_path)
        if dur and dur > 0:
            return max(int(dur * FPS) + FPS, FPS)
    return int(default_seconds * FPS)


def render_scene(
    *,
    family: str,
    props: dict[str, Any],
    output_path: Path,
    duration_in_frames: int = 150,
    format: str = "mp4",
) -> Path:
    """Render a single Remotion composition to a file.

    Args:
        family: Composition ID (must be in VALID_FAMILIES).
        props: JSON-serializable props dict for the composition.
        output_path: Where to write the rendered file.
        duration_in_frames: How many frames to render.
        format: "mp4" or "png" (single frame).

    Returns:
        Path to the rendered output file.

    Raises:
        ValueError: If family is unknown.
        RuntimeError: If Remotion render fails.
    """
    if family not in VALID_FAMILIES:
        raise ValueError(f"Unknown composition family: {family!r}. Must be one of {VALID_FAMILIES}")

    composition_id = _to_remotion_id(family)
    npx = _find_npx()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    props_json = json.dumps(props, ensure_ascii=False)

    if format == "png":
        cmd = [
            npx, "remotion", "still",
            composition_id,
            "--output", str(output_path),
            "--props", props_json,
        ]
    else:
        cmd = [
            npx, "remotion", "render",
            composition_id,
            "--output", str(output_path),
            "--props", props_json,
            "--frames", f"0-{duration_in_frames - 1}",
            "--codec", "h264",
        ]

    _log(f"Rendering {family} → {output_path.name} ({duration_in_frames} frames)")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REMOTION_DIR),
    )

    if result.returncode != 0:
        _log(f"Render FAILED: {result.stderr[-500:]}")
        raise RuntimeError(
            f"Remotion render failed for {family}: {result.stderr[-300:]}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Render produced empty output: {output_path}")

    _log(f"  OK: {output_path.name} ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path


def render_plan_scenes(
    plan: dict[str, Any],
    project_dir: Path,
    *,
    render_dir_name: str = "remotion_renders",
) -> list[dict[str, Any]]:
    """Render all scenes from a plan.json into per-scene MP4s.

    Args:
        plan: Parsed plan.json dict (must have "scenes" key).
        project_dir: Project directory (audio files resolved relative to this).
        render_dir_name: Subdirectory under project_dir for rendered outputs.

    Returns:
        List of scene dicts augmented with "clip_path" pointing to the rendered MP4.
    """
    scenes = plan.get("scenes", [])
    if not scenes:
        raise ValueError("Plan has no scenes")

    render_dir = Path(project_dir) / render_dir_name
    render_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for scene in scenes:
        scene_id = scene.get("id", len(results))
        composition = scene.get("composition", {})
        family = composition.get("family", "narration_slide")
        props = composition.get("props", {})

        if family not in VALID_FAMILIES:
            _log(f"Scene {scene_id}: unknown family {family!r}, falling back to narration_slide")
            family = "narration_slide"
            props = {"headline": scene.get("title", ""), "body": ""}

        audio_path = scene.get("audio_path")
        if audio_path:
            audio_path = Path(audio_path)
            if not audio_path.is_absolute():
                audio_path = Path(project_dir) / audio_path

        frames = duration_frames(audio_path)

        output_file = render_dir / f"scene_{scene_id:03d}.mp4"

        try:
            render_scene(
                family=family,
                props=props,
                output_path=output_file,
                duration_in_frames=frames,
            )
            scene_result = dict(scene)
            scene_result["clip_path"] = str(output_file)
            results.append(scene_result)
        except Exception as exc:
            _log(f"Scene {scene_id} render failed: {exc}")
            scene_result = dict(scene)
            scene_result["clip_path"] = None
            scene_result["render_error"] = str(exc)
            results.append(scene_result)

    rendered = sum(1 for r in results if r.get("clip_path"))
    _log(f"Rendered {rendered}/{len(scenes)} scenes")
    return results


def render_scene_still(
    *,
    family: str,
    props: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render a single frame PNG for preview purposes."""
    return render_scene(
        family=family,
        props=props,
        output_path=output_path,
        duration_in_frames=1,
        format="png",
    )


def render_dynamic_scene(
    *,
    scene_code: str,
    output_path: Path,
    duration_in_frames: int = 150,
    format: str = "mp4",
) -> Path:
    """Render a scene from generated Remotion component code.

    The scene_code is the body of a React functional component.
    It gets passed as a prop to the 'dynamic-scene' composition which
    compiles and renders it at runtime.
    """
    npx = _find_npx()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    props_json = json.dumps({
        "code": scene_code,
        "durationInFrames": duration_in_frames,
    }, ensure_ascii=False)

    if format == "png":
        cmd = [
            npx, "remotion", "still",
            "dynamic-scene",
            "--output", str(output_path),
            "--props", props_json,
        ]
    else:
        cmd = [
            npx, "remotion", "render",
            "dynamic-scene",
            "--output", str(output_path),
            "--props", props_json,
            "--codec", "h264",
        ]

    _log(f"Rendering dynamic scene → {output_path.name} ({duration_in_frames} frames)")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REMOTION_DIR),
    )

    if result.returncode != 0:
        _log(f"Dynamic render FAILED: {result.stderr[-500:]}")
        raise RuntimeError(f"Dynamic scene render failed: {result.stderr[-300:]}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Render produced empty output: {output_path}")

    _log(f"  OK: {output_path.name} ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path


def render_plan_scenes_dynamic(
    plan: dict[str, Any],
    project_dir: Path,
    *,
    render_dir_name: str = "remotion_renders",
) -> list[dict[str, Any]]:
    """Render all scenes from a plan with per-scene generated code.

    Each scene should have a "scene_code" field containing the React component body.
    Falls back to the template-based render_scene if scene_code is absent.
    """
    scenes = plan.get("scenes", [])
    if not scenes:
        raise ValueError("Plan has no scenes")

    render_dir = Path(project_dir) / render_dir_name
    render_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for scene in scenes:
        scene_id = scene.get("id", len(results))
        scene_code = scene.get("scene_code", "")
        output_file = render_dir / f"scene_{scene_id:03d}.mp4"

        audio_path = scene.get("audio_path")
        if audio_path:
            audio_p = Path(audio_path)
            if not audio_p.is_absolute():
                audio_p = Path(project_dir) / audio_path
        else:
            audio_p = None
        frames = duration_frames(audio_p)

        try:
            if scene_code.strip():
                render_dynamic_scene(
                    scene_code=scene_code,
                    output_path=output_file,
                    duration_in_frames=frames,
                )
            else:
                composition = scene.get("composition", {})
                family = composition.get("family", "narration_slide")
                props = composition.get("props", {"headline": scene.get("title", "")})
                if family not in VALID_FAMILIES:
                    family = "narration_slide"
                render_scene(
                    family=family, props=props,
                    output_path=output_file, duration_in_frames=frames,
                )
            scene_result = dict(scene)
            scene_result["clip_path"] = str(output_file)
            results.append(scene_result)
        except Exception as exc:
            _log(f"Scene {scene_id} render failed: {exc}")
            scene_result = dict(scene)
            scene_result["clip_path"] = None
            scene_result["render_error"] = str(exc)
            results.append(scene_result)

    rendered = sum(1 for r in results if r.get("clip_path"))
    _log(f"Rendered {rendered}/{len(scenes)} scenes")
    return results
