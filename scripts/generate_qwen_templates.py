#!/usr/bin/env python3
"""Generate curated text-free Qwen template backgrounds for phase-2 workflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.image_gen import DEFAULT_IMAGE_GEN_MODEL, generate_image


DEFAULT_MANIFEST_PATH = REPO_ROOT / "templates" / "manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "template_generation"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    templates = payload.get("templates")
    if not isinstance(templates, list):
        raise ValueError("manifest templates must be a list")
    return payload


def _default_prompt_for_template(template_id: str, scene_types: list[str]) -> str:
    scene_label = ", ".join(scene_types) if scene_types else "clinical data visualization"
    return (
        f"Text-free cinematic medical background for template {template_id}. "
        f"Archetype focus: {scene_label}. "
        "Premium qEEG visual style, atmospheric depth, clean composition, no labels, "
        "no letters, no numbers, no words, no watermarks, no logos, no UI chrome."
    )


def _target_templates(
    manifest_templates: list[dict[str, Any]],
    *,
    template_ids: set[str],
    all_archetypes: bool,
    include_dev_only: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in manifest_templates:
        if not isinstance(item, dict):
            continue
        template_id = str(item.get("template_id") or "").strip()
        if not template_id:
            continue
        if template_ids and template_id not in template_ids:
            continue
        if not template_ids:
            if all_archetypes and template_id == "generic_data_panel_v1":
                continue
            if not all_archetypes:
                if template_id == "generic_data_panel_v1":
                    continue
                if str(item.get("origin") or "") != "scaffold_only":
                    continue
        if (
            not include_dev_only
            and str(item.get("operational_status") or "") == "dev_only"
            and str(item.get("origin") or "") != "scaffold_only"
        ):
            continue
        out.append(item)
    out.sort(key=lambda x: int(x.get("priority", 1000)))
    return out


def run_generation(
    *,
    repo_root: Path,
    manifest_path: Path,
    template_ids: set[str],
    all_archetypes: bool,
    include_dev_only: bool,
    live: bool,
    model: str,
    output_root: Path,
    update_manifest: bool,
) -> tuple[int, dict[str, Any]]:
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    manifest = _load_manifest(manifest_path)
    templates = manifest.get("templates") or []
    if not isinstance(templates, list):
        raise ValueError("manifest templates must be a list")

    targets = _target_templates(
        templates,
        template_ids=template_ids,
        all_archetypes=all_archetypes,
        include_dev_only=include_dev_only,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_dir = output_root / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures = 0
    generated = 0

    for template in targets:
        template_id = str(template.get("template_id") or "").strip()
        scene_types = template.get("scene_types") if isinstance(template.get("scene_types"), list) else []
        scene_types = [str(s) for s in scene_types]
        rel_path = str(template.get("template_path") or "").strip()
        if not rel_path:
            rows.append(
                {
                    "template_id": template_id,
                    "status": "failed",
                    "error": "template_path missing",
                }
            )
            failures += 1
            continue

        out_path = (repo_root / rel_path).resolve()
        prompt = str(template.get("generation_prompt") or "").strip()
        if not prompt:
            prompt = _default_prompt_for_template(template_id, scene_types)

        row: dict[str, Any] = {
            "template_id": template_id,
            "scene_types": scene_types,
            "output_path": str(out_path),
            "model": model,
            "live": bool(live),
            "prompt": prompt,
            "timestamp_utc": _utc_now(),
        }

        if not live:
            row["status"] = "dry_run"
            rows.append(row)
            continue

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            generated_path = generate_image(
                prompt,
                output_path=out_path,
                model=model,
                apply_style=True,
                use_eeg_10_20_guide=False,
            )
            row["status"] = "generated"
            row["generated_path"] = str(generated_path)
            generated += 1
            if update_manifest:
                template["origin"] = "qwen_curated"
                template["production_ready"] = False
                template["curation_status"] = "generated_pending_approval"
                template["approved_by"] = None
                template["approved_at"] = None
        except Exception as exc:
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            failures += 1
        rows.append(row)

    if live and update_manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "created_utc": _utc_now(),
        "manifest_path": str(manifest_path.resolve()),
        "env_loaded": str(env_path.resolve()) if env_path.exists() else "",
        "replicate_token_present": bool(os.getenv("REPLICATE_API_TOKEN")),
        "live": bool(live),
        "model": model,
        "target_count": len(targets),
        "generated_count": generated,
        "failed_count": failures,
        "artifact_dir": str(artifact_dir.resolve()),
        "rows": rows,
    }
    report_path = artifact_dir / "generation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return (1 if failures else 0, report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text-free Qwen template assets.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--template-id", action="append", default=[])
    parser.add_argument("--all-archetypes", action="store_true")
    parser.add_argument("--include-dev-only", action="store_true")
    parser.add_argument("--live", action="store_true", help="Call provider and write PNG outputs")
    parser.add_argument("--model", default=DEFAULT_IMAGE_GEN_MODEL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-manifest-update", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    code, report = run_generation(
        repo_root=REPO_ROOT,
        manifest_path=Path(args.manifest),
        template_ids={str(t).strip() for t in args.template_id if str(t).strip()},
        all_archetypes=bool(args.all_archetypes),
        include_dev_only=bool(args.include_dev_only),
        live=bool(args.live),
        model=str(args.model),
        output_root=Path(args.output_root),
        update_manifest=not bool(args.no_manifest_update),
    )
    print(json.dumps({
        "status": "ok" if code == 0 else "failed",
        "target_count": report.get("target_count"),
        "generated_count": report.get("generated_count"),
        "failed_count": report.get("failed_count"),
        "artifact_dir": report.get("artifact_dir"),
    }, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
