#!/usr/bin/env python3
"""Build or refresh anchor JSON files after template selection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
DEFAULT_MANIFEST_PATH = REPO_ROOT / "templates" / "manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "template_anchors"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be JSON object")
    if not isinstance(payload.get("templates"), list):
        raise ValueError("manifest templates must be list")
    return payload


def _select_templates(
    templates: list[dict[str, Any]],
    *,
    template_ids: set[str],
    include_all: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in templates:
        if not isinstance(item, dict):
            continue
        template_id = str(item.get("template_id") or "").strip()
        if not template_id:
            continue
        if template_ids and template_id not in template_ids:
            continue
        if not template_ids and not include_all:
            if str(item.get("origin") or "") != "qwen_curated":
                continue
        out.append(item)
    out.sort(key=lambda x: int(x.get("priority", 1000)))
    return out


def _prototype_for_template(template: dict[str, Any]) -> str:
    scene_types = template.get("scene_types") if isinstance(template.get("scene_types"), list) else []
    selector = template.get("selector") if isinstance(template.get("selector"), dict) else {}
    primary = str(scene_types[0]) if scene_types else ""

    if primary == "bar_volume_chart":
        if int(selector.get("bar_count_eq", 3) or 3) == 2:
            return "bar_volume_chart_2panel_v1.json"
        return "bar_volume_chart_3panel_v1.json"
    if primary == "split_opposing_trends":
        return "split_opposing_trends_v1.json"
    if primary == "roadmap_agenda":
        return "roadmap_agenda_v1.json"
    if primary == "waveform_voltage_panel":
        return "waveform_voltage_panel_v1.json"
    if primary == "radial_kpi_ring":
        return "radial_kpi_ring_v1.json"
    if primary.startswith("atmospheric_"):
        return "title_card_v1.json"
    return "generic_data_panel_v1.json"


def run_anchor_build(
    *,
    repo_root: Path,
    manifest_path: Path,
    output_root: Path,
    template_ids: set[str],
    include_all: bool,
    force: bool,
    dry_run: bool,
) -> tuple[int, dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    templates = manifest.get("templates") or []
    if not isinstance(templates, list):
        raise ValueError("manifest templates must be list")

    targets = _select_templates(templates, template_ids=template_ids, include_all=include_all)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_dir = output_root / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures = 0
    created = 0

    for template in targets:
        template_id = str(template.get("template_id") or "").strip()
        rel_anchor = str(template.get("anchors_path") or "").strip()
        if not rel_anchor:
            rows.append({"template_id": template_id, "status": "failed", "error": "anchors_path missing"})
            failures += 1
            continue

        anchor_path = (repo_root / rel_anchor).resolve()
        row: dict[str, Any] = {
            "template_id": template_id,
            "anchors_path": str(anchor_path),
            "timestamp_utc": _utc_now(),
            "dry_run": bool(dry_run),
        }

        if anchor_path.exists() and not force:
            row["status"] = "existing"
            rows.append(row)
            continue

        prototype_name = _prototype_for_template(template)
        prototype_path = (repo_root / "templates" / "anchors" / prototype_name).resolve()
        row["prototype"] = str(prototype_path)
        if not prototype_path.exists():
            row["status"] = "failed"
            row["error"] = f"prototype anchor missing: {prototype_path}"
            rows.append(row)
            failures += 1
            continue

        payload = json.loads(prototype_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            row["status"] = "failed"
            row["error"] = "prototype payload not object"
            rows.append(row)
            failures += 1
            continue

        payload["template_id"] = template_id

        if dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue

        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        row["status"] = "created"
        rows.append(row)
        created += 1

    report = {
        "created_utc": _utc_now(),
        "manifest_path": str(manifest_path.resolve()),
        "target_count": len(targets),
        "created_count": created,
        "failed_count": failures,
        "artifact_dir": str(artifact_dir.resolve()),
        "rows": rows,
    }
    report_path = artifact_dir / "anchor_build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return (1 if failures else 0, report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build anchor JSON files from archetype prototypes.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--template-id", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="Include all templates instead of qwen_curated-only")
    parser.add_argument("--force", action="store_true", help="Overwrite existing anchors")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    code, report = run_anchor_build(
        repo_root=REPO_ROOT,
        manifest_path=Path(args.manifest),
        output_root=Path(args.output_root),
        template_ids={str(t).strip() for t in args.template_id if str(t).strip()},
        include_all=bool(args.all),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({
        "status": "ok" if code == 0 else "failed",
        "target_count": report.get("target_count"),
        "created_count": report.get("created_count"),
        "failed_count": report.get("failed_count"),
        "artifact_dir": report.get("artifact_dir"),
    }, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
