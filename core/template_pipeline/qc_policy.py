"""QC policy helpers for deterministic/template fallback behavior."""

from __future__ import annotations

from typing import Any


def detect_qwen_fallback_scene_ids(plan: dict[str, Any]) -> list[int]:
    scenes = plan.get("scenes")
    if not isinstance(scenes, list):
        return []
    out: list[int] = []
    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue
        sid = int(scene.get("id", i))
        if bool(scene.get("fallback_to_qwen")) or str(scene.get("render_mode") or "").strip() == "qwen_fallback":
            out.append(sid)
    return sorted(set(out))

