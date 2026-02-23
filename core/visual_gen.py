"""Scene visual generation router (AI image backend + Blender backend)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from core.blender_gen import BlenderRenderConfig, render_blender_scene
from core.image_gen import generate_scene_image
from core.template_pack import render_non_brain_scene


BLENDER_QEEG_MARKER = "[[BLENDER_QEEG]]"
ALLOW_NON_BRAIN_AI_FALLBACK_ENV = "ALLOW_NON_BRAIN_AI_FALLBACK"


def _allow_non_brain_ai_fallback() -> bool:
    value = str(os.getenv(ALLOW_NON_BRAIN_AI_FALLBACK_ENV) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
    Generate a scene visual using Blender for qEEG scenes, else deterministic template-pack rendering.

    Emergency non-brain AI fallback can be enabled via ALLOW_NON_BRAIN_AI_FALLBACK.
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

    manifest_override = image_kwargs.pop("template_manifest_path", None)
    fallback_mode = image_kwargs.pop("non_brain_fallback_mode", None)

    try:
        return render_non_brain_scene(
            scene,
            project_dir,
            manifest_path=Path(manifest_override) if manifest_override else None,
            fallback_mode=str(fallback_mode) if fallback_mode else None,
        )
    except Exception:
        if _allow_non_brain_ai_fallback():
            return generate_scene_image(scene, project_dir, **image_kwargs)
        raise
