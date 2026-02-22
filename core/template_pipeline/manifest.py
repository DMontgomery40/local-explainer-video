"""Template manifest loading and candidate resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scene_schemas import normalize_scene_type

TEMPLATE_ORIGINS = {"qwen_curated", "scaffold_only"}
TEMPLATE_OPERATIONAL_STATUSES = {"production", "dev_only"}


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    scene_types: tuple[str, ...]
    template_path: Path
    anchors_path: Path
    priority: int
    selector: dict[str, Any]
    selector_metadata: dict[str, Any]
    tags: tuple[str, ...]
    origin: str
    operational_status: str
    production_ready: bool
    curation_status: str
    approved_by: str | None
    approved_at: str | None


@dataclass(frozen=True)
class TemplateManifest:
    manifest_version: str
    templates: tuple[TemplateSpec, ...]
    path: Path


class TemplateManifestError(ValueError):
    """Raised on malformed template manifest files."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    return _repo_root() / "templates" / "manifest.json"


def load_manifest(path: Path | None = None) -> TemplateManifest:
    manifest_path = path or default_manifest_path()
    if not manifest_path.exists():
        raise TemplateManifestError(f"Template manifest not found: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TemplateManifestError(f"Failed parsing template manifest: {manifest_path}") from exc

    manifest_version = str(payload.get("manifest_version") or "").strip()
    if not manifest_version:
        raise TemplateManifestError("manifest_version is required")

    raw_templates = payload.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise TemplateManifestError("manifest templates must be a non-empty array")

    repo_root = _repo_root()
    specs: list[TemplateSpec] = []
    ids_seen: set[str] = set()
    for idx, raw in enumerate(raw_templates):
        if not isinstance(raw, dict):
            raise TemplateManifestError(f"templates[{idx}] must be an object")

        template_id = str(raw.get("template_id") or "").strip()
        if not template_id:
            raise TemplateManifestError(f"templates[{idx}] missing template_id")
        if template_id in ids_seen:
            raise TemplateManifestError(f"Duplicate template_id: {template_id}")
        ids_seen.add(template_id)

        raw_scene_types = raw.get("scene_types")
        if isinstance(raw_scene_types, str):
            raw_scene_types = [raw_scene_types]
        if not isinstance(raw_scene_types, list) or not raw_scene_types:
            raise TemplateManifestError(f"{template_id}: scene_types must be non-empty list")
        scene_types: list[str] = []
        for st in raw_scene_types:
            normalized = normalize_scene_type(str(st))
            if not normalized:
                raise TemplateManifestError(f"{template_id}: invalid scene_type value: {st!r}")
            scene_types.append(normalized)

        template_rel = str(raw.get("template_path") or "").strip()
        anchors_rel = str(raw.get("anchors_path") or "").strip()
        if not template_rel or not anchors_rel:
            raise TemplateManifestError(f"{template_id}: template_path and anchors_path are required")

        template_path = (repo_root / template_rel).resolve()
        anchors_path = (repo_root / anchors_rel).resolve()
        if not template_path.exists():
            raise TemplateManifestError(f"{template_id}: missing template image: {template_path}")
        if not anchors_path.exists():
            raise TemplateManifestError(f"{template_id}: missing anchors file: {anchors_path}")

        selector = raw.get("selector")
        if selector is None:
            selector = {}
        if not isinstance(selector, dict):
            raise TemplateManifestError(f"{template_id}: selector must be an object")
        selector_metadata = raw.get("selector_metadata")
        if selector_metadata is None:
            selector_metadata = {}
        if not isinstance(selector_metadata, dict):
            raise TemplateManifestError(f"{template_id}: selector_metadata must be an object")

        tags_raw = raw.get("tags") or []
        if not isinstance(tags_raw, list):
            raise TemplateManifestError(f"{template_id}: tags must be a list")
        tags = tuple(str(t).strip() for t in tags_raw if str(t).strip())

        origin = str(raw.get("origin") or "").strip()
        if origin not in TEMPLATE_ORIGINS:
            raise TemplateManifestError(
                f"{template_id}: origin must be one of {sorted(TEMPLATE_ORIGINS)}"
            )
        operational_status = str(raw.get("operational_status") or "").strip()
        if operational_status not in TEMPLATE_OPERATIONAL_STATUSES:
            raise TemplateManifestError(
                f"{template_id}: operational_status must be one of {sorted(TEMPLATE_OPERATIONAL_STATUSES)}"
            )
        if "production_ready" not in raw:
            raise TemplateManifestError(f"{template_id}: production_ready is required")
        production_ready = bool(raw.get("production_ready"))
        if production_ready and operational_status != "production":
            raise TemplateManifestError(
                f"{template_id}: production_ready=true requires operational_status=production"
            )
        curation_status = str(raw.get("curation_status") or "").strip() or "pending"
        approved_by = str(raw.get("approved_by") or "").strip() or None
        approved_at = str(raw.get("approved_at") or "").strip() or None

        priority = int(raw.get("priority", 100))
        specs.append(
            TemplateSpec(
                template_id=template_id,
                scene_types=tuple(scene_types),
                template_path=template_path,
                anchors_path=anchors_path,
                priority=priority,
                selector=selector,
                selector_metadata=selector_metadata,
                tags=tags,
                origin=origin,
                operational_status=operational_status,
                production_ready=production_ready,
                curation_status=curation_status,
                approved_by=approved_by,
                approved_at=approved_at,
            )
        )

    specs.sort(key=lambda s: s.priority)
    return TemplateManifest(
        manifest_version=manifest_version,
        templates=tuple(specs),
        path=manifest_path.resolve(),
    )


def selector_probe(scene: dict[str, Any]) -> dict[str, Any]:
    """Build a small probe object used for selector condition checks."""
    structured_data = scene.get("structured_data")
    if not isinstance(structured_data, dict):
        structured_data = {}

    bars = structured_data.get("bars")
    edges = structured_data.get("edges")
    items = structured_data.get("items")
    rows = structured_data.get("rows")
    sessions = structured_data.get("sessions")
    rings = structured_data.get("rings")
    traces = structured_data.get("traces")
    points = structured_data.get("points")

    probe = {
        "scene_type": normalize_scene_type(scene.get("scene_type")),
        "bar_count": len(bars) if isinstance(bars, list) else None,
        "edge_count": len(edges) if isinstance(edges, list) else None,
        "item_count": len(items) if isinstance(items, list) else None,
        "row_count": len(rows) if isinstance(rows, list) else None,
        "session_count": len(sessions) if isinstance(sessions, list) else None,
        "ring_count": len(rings) if isinstance(rings, list) else None,
        "trace_count": len(traces) if isinstance(traces, list) else None,
        "point_count": len(points) if isinstance(points, list) else None,
        "trend": str(structured_data.get("trend") or "").strip().lower() or None,
    }
    return probe
