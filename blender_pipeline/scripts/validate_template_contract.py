#!/usr/bin/env python3
"""Validate required qEEG brain base-model objects/materials inside a .blend template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import bpy


REQUIRED_COLLECTIONS = (
    "COL_WORLD",
    "COL_ELECTRODES",
    "COL_ELECTRODE_ANCHORS",
    "COL_LINES",
    "COL_TEXT",
    "COL_GUIDES",
)
REQUIRED_OBJECTS = (
    "CAM_MAIN",
    "CAM_TARGET",
    "MESH_BRAIN",
    "MESH_HEAD_SHELL",
    "VOL_ATMOSPHERE",
    "LIGHT_KEY",
    "LIGHT_FILL",
    "LIGHT_RIM",
    "LIGHT_OVERHEAD",
    "TXT_TITLE",
    "TXT_SUBTITLE",
    "TXT_FOOTER",
)
REQUIRED_MATERIALS = (
    "MAT_BRAIN_BASE",
    "MAT_HEAD_SHELL",
    "MAT_FOG_VOLUME",
    "MAT_TEXT_BASE",
    "MAT_ELECTRODE_BASE",
    "MAT_LABEL_BASE",
    "MAT_LINE_BASE",
)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Validate qEEG template contract")
    parser.add_argument("--template", required=True, help="Path to .blend template")
    parser.add_argument("--report", default="", help="Optional JSON report output path")
    return parser.parse_args(argv)


def _mat_has_emission_channels(mat: bpy.types.Material) -> bool:
    if not mat.use_nodes or mat.node_tree is None:
        return False
    for node in mat.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        em_color = node.inputs.get("Emission Color") is not None
        em_strength = node.inputs.get("Emission Strength") is not None
        if em_color and em_strength:
            return True
    return False


def _validate() -> dict[str, object]:
    missing_collections = [name for name in REQUIRED_COLLECTIONS if bpy.data.collections.get(name) is None]
    missing_objects = [name for name in REQUIRED_OBJECTS if bpy.data.objects.get(name) is None]
    missing_materials = [name for name in REQUIRED_MATERIALS if bpy.data.materials.get(name) is None]

    bad_material_channels: list[str] = []
    for mat_name in REQUIRED_MATERIALS:
        if mat_name == "MAT_FOG_VOLUME":
            continue
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            continue
        if not _mat_has_emission_channels(mat):
            bad_material_channels.append(mat_name)

    anchors = [obj.name for obj in bpy.data.objects if obj.name.startswith("ANCH_E_")]
    electrodes = [obj.name for obj in bpy.data.objects if obj.name.startswith("E_")]
    labels = [obj.name for obj in bpy.data.objects if obj.name.startswith("LBL_")]

    errors: list[str] = []
    if missing_collections:
        errors.append(f"Missing collections: {', '.join(missing_collections)}")
    if missing_objects:
        errors.append(f"Missing objects: {', '.join(missing_objects)}")
    if missing_materials:
        errors.append(f"Missing materials: {', '.join(missing_materials)}")
    if bad_material_channels:
        errors.append(f"Materials missing emission channels: {', '.join(bad_material_channels)}")
    if len(anchors) < 19:
        errors.append(f"Expected >=19 anchors, found {len(anchors)}")
    if len(electrodes) < 19:
        errors.append(f"Expected >=19 electrode objects, found {len(electrodes)}")
    if len(labels) < 19:
        errors.append(f"Expected >=19 electrode labels, found {len(labels)}")

    scene = bpy.context.scene
    template_version = scene.get("qeeg_template_version")
    if not isinstance(template_version, str) or not template_version.strip():
        errors.append("Missing scene['qeeg_template_version']")

    return {
        "ok": len(errors) == 0,
        "template_version": template_version,
        "counts": {
            "anchors": len(anchors),
            "electrodes": len(electrodes),
            "labels": len(labels),
        },
        "missing": {
            "collections": missing_collections,
            "objects": missing_objects,
            "materials": missing_materials,
            "material_channels": bad_material_channels,
        },
        "errors": errors,
    }


def main() -> int:
    args = _parse_args()
    template_path = Path(args.template).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if str(args.report).strip() else None

    if not template_path.exists():
        raise RuntimeError(f"Template .blend not found: {template_path}")

    bpy.ops.wm.open_mainfile(filepath=str(template_path))
    report = _validate()

    print(json.dumps(report, indent=2))

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[validate_template_contract] report={report_path}")

    return 0 if bool(report.get("ok")) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[validate_template_contract] ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
