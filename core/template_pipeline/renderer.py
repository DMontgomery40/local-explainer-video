"""Template renderer orchestration for deterministic scene images."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .compositor import CompositorError, compose_scene
from .manifest import TemplateManifest, load_manifest
from .scene_schemas import (
    SceneValidationError,
    normalize_scene_type,
    validate_scene,
)
from .selector import TemplateSelectionError, select_template


@dataclass(frozen=True)
class RenderDecision:
    template_id: str
    template_path: Path
    anchors_path: Path
    backend: str


class TemplateRenderError(RuntimeError):
    """Raised when deterministic template rendering fails."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
        scene_id: int | None = None,
        scene_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.scene_id = scene_id
        self.scene_type = scene_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "message": str(self),
        }


def use_template_pipeline() -> bool:
    raw = str(os.getenv("USE_TEMPLATE_PIPELINE", "true")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def allow_qwen_fallback() -> bool:
    raw = str(os.getenv("ALLOW_TEMPLATE_FALLBACK_TO_QWEN", "false")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def template_pipeline_mode() -> str:
    raw = str(os.getenv("TEMPLATE_PIPELINE_MODE", "development")).strip().lower()
    if raw in {"prod", "production"}:
        return "production"
    return "development"


def production_template_mode() -> bool:
    return template_pipeline_mode() == "production"


def compositor_backend() -> str:
    return str(os.getenv("TEMPLATE_COMPOSITOR_BACKEND", "pillow")).strip().lower() or "pillow"


def _scene_output_path(scene: dict[str, Any], project_dir: Path) -> Path:
    scene_id = int(scene.get("id", 0))
    return project_dir / "images" / f"scene_{scene_id:03d}.png"


def _safe_scene_id(scene: dict[str, Any]) -> int | None:
    try:
        return int(scene.get("id", 0))
    except Exception:
        return None


def build_render_decision(
    scene: dict[str, Any],
    *,
    manifest: TemplateManifest | None = None,
    backend: str | None = None,
    production_only: bool | None = None,
) -> RenderDecision:
    loaded_manifest = manifest or load_manifest()
    validated_scene = validate_scene(scene)
    spec = select_template(
        validated_scene,
        loaded_manifest,
        production_only=production_template_mode() if production_only is None else bool(production_only),
    )
    return RenderDecision(
        template_id=spec.template_id,
        template_path=spec.template_path,
        anchors_path=spec.anchors_path,
        backend=(backend or compositor_backend()),
    )


def classify_render_exception(exc: Exception) -> str:
    if isinstance(exc, SceneValidationError):
        return "validation"
    if isinstance(exc, TemplateSelectionError):
        return "selection"
    if isinstance(exc, CompositorError):
        return "compositor"
    if isinstance(exc, ValueError):
        return "value_error"
    return "unknown"


def _append_render_audit(project_dir: Path, payload: dict[str, Any]) -> None:
    audit_dir = project_dir / "artifacts"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "template_render_audit.jsonl"
    event = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        **payload,
    }
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def render_scene_to_image(
    scene: dict[str, Any],
    *,
    project_dir: Path,
    manifest: TemplateManifest | None = None,
    backend: str | None = None,
    output_path: Path | None = None,
    production_only: bool | None = None,
) -> Path:
    try:
        validated_scene = validate_scene(scene)
        decision = build_render_decision(
            validated_scene,
            manifest=manifest,
            backend=backend,
            production_only=production_only,
        )
        out = output_path or _scene_output_path(scene, project_dir)
        rendered = compose_scene(
            template_path=decision.template_path,
            anchors_path=decision.anchors_path,
            scene=validated_scene,
            output_path=out,
            backend=decision.backend,
        )
        scene["scene_type"] = validated_scene["scene_type"]
        scene["structured_data"] = validated_scene["structured_data"]
        scene["template_id"] = decision.template_id
        scene["render_backend"] = decision.backend
        scene["render_mode"] = "template_deterministic"
        scene["image_path"] = str(rendered)
        _append_render_audit(
            project_dir=project_dir,
            payload={
                "event": "render_success",
                "scene_id": int(scene.get("id", 0)),
                "scene_type": validated_scene.get("scene_type"),
                "template_id": decision.template_id,
                "render_backend": decision.backend,
                "pipeline_mode": template_pipeline_mode(),
            },
        )
        return rendered
    except (SceneValidationError, TemplateSelectionError, CompositorError, ValueError) as exc:
        category = classify_render_exception(exc)
        scene_id = _safe_scene_id(scene)
        scene_type = normalize_scene_type(scene.get("scene_type"))
        _append_render_audit(
            project_dir=project_dir,
            payload={
                "event": "render_failure",
                "scene_id": scene_id,
                "scene_type": scene_type,
                "pipeline_mode": template_pipeline_mode(),
                "category": category,
                "error": str(exc),
            },
        )
        scene["render_error"] = {
            "category": category,
            "message": str(exc),
        }
        raise TemplateRenderError(
            str(exc),
            category=category,
            scene_id=scene_id,
            scene_type=scene_type,
        ) from exc


def scene_eligible_for_template(scene: dict[str, Any]) -> bool:
    scene_type = normalize_scene_type(scene.get("scene_type"))
    if not scene_type:
        return False
    structured_data = scene.get("structured_data")
    return isinstance(structured_data, dict)
