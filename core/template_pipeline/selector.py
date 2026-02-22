"""Deterministic heuristic template selector."""

from __future__ import annotations

from typing import Any

from .manifest import TemplateManifest, TemplateSpec, selector_probe
from .scene_schemas import normalize_scene_type


class TemplateSelectionError(ValueError):
    """Raised when no template can be selected for a scene."""


def _matches_selector(spec: TemplateSpec, probe: dict[str, Any]) -> bool:
    selector = spec.selector
    if not selector:
        return True

    for key, raw_expected in selector.items():
        if key == "trend":
            expected = str(raw_expected).strip().lower()
            actual = str(probe.get("trend") or "").strip().lower()
            if expected and expected != actual:
                return False
            continue
        if key.endswith("_min"):
            base = key.removesuffix("_min")
            actual = probe.get(base)
            if actual is None or float(actual) < float(raw_expected):
                return False
            continue
        if key.endswith("_max"):
            base = key.removesuffix("_max")
            actual = probe.get(base)
            if actual is None or float(actual) > float(raw_expected):
                return False
            continue
        if key.endswith("_eq"):
            base = key.removesuffix("_eq")
            actual = probe.get(base)
            if actual is None or float(actual) != float(raw_expected):
                return False
            continue
        if key.endswith("_in"):
            base = key.removesuffix("_in")
            actual = probe.get(base)
            if isinstance(raw_expected, list):
                if actual not in raw_expected:
                    return False
            else:
                return False
            continue

        actual = probe.get(key)
        if actual != raw_expected:
            return False

    return True


def _candidates_for_scene(manifest: TemplateManifest, scene_type: str) -> list[TemplateSpec]:
    return [
        spec
        for spec in manifest.templates
        if scene_type in spec.scene_types
    ]


def _production_candidates(candidates: list[TemplateSpec]) -> list[TemplateSpec]:
    return [
        spec
        for spec in candidates
        if spec.operational_status == "production" and spec.production_ready
    ]


def select_template(
    scene: dict[str, Any],
    manifest: TemplateManifest,
    *,
    production_only: bool = False,
) -> TemplateSpec:
    scene_type = normalize_scene_type(scene.get("scene_type"))
    if not scene_type:
        raise TemplateSelectionError(f"Scene {scene.get('id')} missing scene_type")

    candidates = _candidates_for_scene(manifest, scene_type)
    if not candidates:
        raise TemplateSelectionError(f"No templates registered for scene_type={scene_type}")
    if production_only:
        candidates = _production_candidates(candidates)
        if not candidates:
            raise TemplateSelectionError(
                f"No production-ready templates registered for scene_type={scene_type}"
            )

    probe = selector_probe(scene)
    filtered = [spec for spec in candidates if _matches_selector(spec, probe)]
    if not filtered:
        if production_only:
            raise TemplateSelectionError(
                f"No production template matched selector for scene_type={scene_type} with probe={probe}"
            )
        # Development fallback to highest-priority candidate for this scene type.
        return candidates[0]

    filtered.sort(key=lambda s: s.priority)
    return filtered[0]
