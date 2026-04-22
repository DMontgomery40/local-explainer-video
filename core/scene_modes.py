"""Helpers for classifying scene execution mode."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def scene_is_cathode_motion(scene: Mapping[str, Any] | None) -> bool:
    """Return whether a scene depends on Cathode native/template motion rendering."""
    if not isinstance(scene, Mapping):
        return False

    scene_type = str(scene.get("scene_type") or "").strip().lower()
    if scene_type == "motion":
        return True

    composition = scene.get("composition")
    if not isinstance(composition, Mapping):
        return False

    mode = str(composition.get("mode") or "").strip().lower()
    manifestation = str(composition.get("manifestation") or "").strip().lower()
    return mode == "native" or manifestation == "native_remotion"


def plan_has_cathode_motion_scenes(plan_or_scenes: Any) -> bool:
    """Return whether any scene in a plan/list is Cathode native motion."""
    scenes: Iterable[Any]
    if isinstance(plan_or_scenes, Mapping):
        raw_scenes = plan_or_scenes.get("scenes")
        scenes = raw_scenes if isinstance(raw_scenes, Iterable) else ()
    elif isinstance(plan_or_scenes, Iterable) and not isinstance(plan_or_scenes, (str, bytes)):
        scenes = plan_or_scenes
    else:
        scenes = ()

    return any(scene_is_cathode_motion(scene) for scene in scenes)
