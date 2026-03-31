"""Scene image generation — now delegates to Remotion render.

Legacy callers (app.py, batch_regenerate.py) call generate_scene_image()
which renders a single-frame PNG via the Remotion composition specified
in the scene's `composition` field. If no composition is present,
falls back to narration_slide with the scene title.

edit_image() is a no-op stub — image editing is not needed when visuals
are deterministic Remotion renders.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

TARGET_WIDTH = 1664
TARGET_HEIGHT = 928


def _log(msg: str) -> None:
    print(f"[IMAGE_GEN] {msg}", file=sys.stderr, flush=True)


def generate_image(prompt: str, output_path: str | Path, **kwargs: Any) -> Path:
    """Render a narration_slide still from a text prompt (legacy compat).

    New code should use core.remotion_bridge.render_scene_still() directly.
    """
    from .remotion_bridge import render_scene_still

    output_path = Path(output_path)
    return render_scene_still(
        family="narration_slide",
        props={"headline": prompt[:80], "body": prompt[80:200] if len(prompt) > 80 else ""},
        output_path=output_path.with_suffix(".png"),
    )


def edit_image(
    image_path: str | Path,
    edit_prompt: str,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """No-op stub — Remotion scenes are deterministic; re-render instead of editing."""
    _log("edit_image() is a no-op in the Remotion pipeline. Re-render the scene instead.")
    return Path(image_path)


def generate_scene_image(
    scene: dict[str, Any],
    project_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Render a single-frame PNG for a scene via Remotion.

    Reads `scene["composition"]` for the family and props.
    Falls back to narration_slide with the scene title if no composition is set.
    """
    from .remotion_bridge import render_scene_still

    project_dir = Path(project_dir)
    scene_id = scene.get("id", 0)

    composition = scene.get("composition", {})
    family = composition.get("family", "narration_slide")
    props = composition.get("props", {})

    if not props:
        props = {
            "headline": scene.get("title", ""),
            "body": "",
        }

    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_path = images_dir / f"scene_{scene_id:03d}.png"

    _log(f"Rendering scene {scene_id} ({family}) → {output_path.name}")

    try:
        result = render_scene_still(
            family=family,
            props=props,
            output_path=output_path,
        )
        scene["image_path"] = str(result)
        _log(f"  OK: {result.name}")
        return result
    except Exception as exc:
        _log(f"  FAILED: {exc}")
        raise
