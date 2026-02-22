#!/usr/bin/env python3
"""Build phase-2 template approval packet and write pending approval artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
DEFAULT_MANIFEST_PATH = REPO_ROOT / "templates" / "manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "template_approval"
DEFAULT_APPROVAL_PATH = REPO_ROOT / ".codex" / "template_approval" / "approval.json"


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from core.template_pipeline.scene_schemas import DATA_SCENE_TYPES


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


def _template_rows(repo_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest.get("templates") or []:
        if not isinstance(item, dict):
            continue
        template_id = str(item.get("template_id") or "").strip()
        if not template_id:
            continue
        rel_bg = str(item.get("template_path") or "").strip()
        rel_anchor = str(item.get("anchors_path") or "").strip()
        bg_path = (repo_root / rel_bg).resolve() if rel_bg else None
        anchor_path = (repo_root / rel_anchor).resolve() if rel_anchor else None
        scene_types = item.get("scene_types") if isinstance(item.get("scene_types"), list) else []
        rows.append(
            {
                "template_id": template_id,
                "scene_types": [str(s) for s in scene_types],
                "template_path": str(bg_path) if bg_path else "",
                "anchors_path": str(anchor_path) if anchor_path else "",
                "template_exists": bool(bg_path and bg_path.exists()),
                "anchors_exists": bool(anchor_path and anchor_path.exists()),
                "origin": str(item.get("origin") or ""),
                "operational_status": str(item.get("operational_status") or ""),
                "production_ready": bool(item.get("production_ready", False)),
                "curation_status": str(item.get("curation_status") or ""),
                "priority": int(item.get("priority", 1000)),
            }
        )
    rows.sort(key=lambda r: (r["priority"], r["template_id"]))
    return rows


def _manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    origin_counts = Counter(str(r.get("origin") or "unknown") for r in rows)
    status_counts = Counter(str(r.get("operational_status") or "unknown") for r in rows)
    curation_counts = Counter(str(r.get("curation_status") or "unknown") for r in rows)

    return {
        "total_templates": len(rows),
        "origin_counts": dict(origin_counts),
        "operational_status_counts": dict(status_counts),
        "curation_status_counts": dict(curation_counts),
        "production_ready_count": sum(1 for r in rows if bool(r.get("production_ready"))),
        "qwen_curated_count": sum(1 for r in rows if str(r.get("origin")) == "qwen_curated"),
        "missing_backgrounds": [r["template_id"] for r in rows if not bool(r.get("template_exists"))],
        "missing_anchors": [r["template_id"] for r in rows if not bool(r.get("anchors_exists"))],
    }


def _coverage_gaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_scene: dict[str, Any] = {}
    missing_registered: list[str] = []
    missing_prod_ready: list[str] = []

    for scene_type in DATA_SCENE_TYPES:
        registered = [
            r["template_id"]
            for r in rows
            if scene_type in (r.get("scene_types") or [])
        ]
        production_ready = [
            r["template_id"]
            for r in rows
            if scene_type in (r.get("scene_types") or [])
            and str(r.get("operational_status")) == "production"
            and bool(r.get("production_ready"))
        ]
        curated = [
            r["template_id"]
            for r in rows
            if scene_type in (r.get("scene_types") or []) and str(r.get("origin")) == "qwen_curated"
        ]
        if not registered:
            missing_registered.append(scene_type)
        if not production_ready:
            missing_prod_ready.append(scene_type)

        per_scene[scene_type] = {
            "registered_templates": registered,
            "production_ready_templates": production_ready,
            "qwen_curated_templates": curated,
        }

    return {
        "missing_registered_scene_types": sorted(missing_registered),
        "missing_production_ready_scene_types": sorted(missing_prod_ready),
        "per_scene_type": per_scene,
    }


def _write_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> Path:
    cards = [r for r in rows if bool(r.get("template_exists")) and str(r.get("template_path"))]
    if not cards:
        img = Image.new("RGB", (1248, 720), (18, 24, 34))
        draw = ImageDraw.Draw(img)
        draw.text((40, 40), "No template backgrounds found", fill=(230, 236, 245))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="PNG")
        return out_path

    thumb_w, thumb_h = 416, 232
    cols = 4
    rows_count = (len(cards) + cols - 1) // cols
    cell_h = thumb_h + 66
    sheet = Image.new("RGB", (cols * thumb_w, rows_count * cell_h), (12, 18, 26))
    draw = ImageDraw.Draw(sheet)

    for idx, card in enumerate(cards):
        col = idx % cols
        row = idx // cols
        x = col * thumb_w
        y = row * cell_h
        path = Path(str(card["template_path"]))
        tile = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + cell_h), fill=(8, 14, 21))
        label = f"{card['template_id']} | {card['origin']}"
        draw.text((x + 8, y + thumb_h + 12), label, fill=(220, 232, 244))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")
    return out_path


def build_approval_packet(
    *,
    repo_root: Path,
    manifest_path: Path,
    output_root: Path,
    approval_path: Path,
    packet_name: str | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    rows = _template_rows(repo_root, manifest)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    packet_dir = output_root / (packet_name or stamp)
    packet_dir.mkdir(parents=True, exist_ok=True)

    (packet_dir / "manifest_snapshot.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = _manifest_summary(rows)
    coverage = _coverage_gaps(rows)

    (packet_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (packet_dir / "coverage_gaps.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (packet_dir / "template_inventory.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    contact_sheet = _write_contact_sheet(rows, packet_dir / "template_contact_sheet.png")

    approval_payload = {
        "approved": False,
        "requested_utc": _utc_now(),
        "packet_dir": str(packet_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
        "required_action": "Review packet and set approved=true with approved_by before resuming phase >2.",
    }
    (packet_dir / "approval_request.json").write_text(
        json.dumps(approval_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    approval_path.parent.mkdir(parents=True, exist_ok=True)
    approval_path.write_text(json.dumps(approval_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "created_utc": _utc_now(),
        "packet_dir": str(packet_dir.resolve()),
        "manifest_summary": summary,
        "coverage_gaps": coverage,
        "approval_path": str(approval_path.resolve()),
    }
    (packet_dir / "packet_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build template approval packet and pending approval artifact.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--approval-path", default=str(DEFAULT_APPROVAL_PATH))
    parser.add_argument("--packet-name", default="")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = build_approval_packet(
        repo_root=REPO_ROOT,
        manifest_path=Path(args.manifest),
        output_root=Path(args.output_root),
        approval_path=Path(args.approval_path),
        packet_name=str(args.packet_name or "").strip() or None,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "packet_dir": result["packet_dir"],
                "approval_path": result["approval_path"],
                "missing_production_ready_scene_types": len(
                    result["coverage_gaps"].get("missing_production_ready_scene_types", [])
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
