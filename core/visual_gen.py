"""Scene visual generation router (AI image backend + Blender backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.blender_gen import BlenderRenderConfig, render_blender_scene
from core.image_gen import generate_scene_image


BLENDER_QEEG_MARKER = "[[BLENDER_QEEG]]"


def is_blender_scene(scene: Mapping[str, Any] | dict[str, Any]) -> bool:
    """Return True when scene should be rendered via Blender."""
    backend = str(scene.get("render_backend") or "").strip().lower()
    if backend == "blender":
        return True
    visual_prompt = str(scene.get("visual_prompt") or "")
    return BLENDER_QEEG_MARKER in visual_prompt


def generate_scene_visual(
    scene: dict[str, Any],
    project_dir: Path,
    *,
    data_pack: Mapping[str, Any] | dict[str, Any] | None = None,
    blender_config: BlenderRenderConfig | None = None,
    force_blender_render: bool = False,
    log: Any = None,
    **image_kwargs: Any,
) -> Path:
    """
    Generate a scene visual using Blender for qEEG scenes, else AI image generation.

    Non-Blender behavior intentionally stays identical to generate_scene_image().
    """
    if is_blender_scene(scene):
        return render_blender_scene(
            scene,
            project_dir,
            data_pack=data_pack,
            config=blender_config,
            force=force_blender_render,
            log=log,
        )
    return generate_scene_image(scene, project_dir, **image_kwargs)
