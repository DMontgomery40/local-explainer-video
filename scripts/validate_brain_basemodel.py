#!/usr/bin/env python3
"""Render deterministic validation fixtures for the qEEG brain base model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_blender_bin() -> str:
    env = str(os.getenv("BLENDER_BIN") or "").strip()
    if env:
        return env
    hardcoded = "/Volumes/Blender/Blender.app/Contents/MacOS/Blender"
    if Path(hardcoded).exists():
        return hardcoded
    found = shutil.which("blender")
    return str(found or "")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate qEEG brain base model")
    parser.add_argument("--blender-bin", default=_default_blender_bin())
    parser.add_argument("--template", default=str(_repo_root() / "blender_pipeline" / "assets" / "qeeg_template.blend"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--width", type=int, default=1664)
    parser.add_argument("--height", type=int, default=928)
    parser.add_argument(
        "--output-dir",
        default=str(_repo_root() / "blender_pipeline" / "_brain_basemodel_validation"),
    )
    parser.add_argument("--skip-rebuild-template", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode for debugging")
    return parser.parse_args()


def _run(cmd: list[str], *, label: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return
    details = [
        f"{label} failed (exit {proc.returncode})",
        " ".join(cmd),
    ]
    if proc.stdout:
        details.extend(["--- stdout ---", proc.stdout[-5000:]])
    if proc.stderr:
        details.extend(["--- stderr ---", proc.stderr[-5000:]])
    raise RuntimeError("\n".join(details))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _render_contact_sheet(
    *,
    images: list[Path],
    titles: list[str],
    output_path: Path,
    columns: int = 3,
) -> None:
    if not images:
        raise RuntimeError("No images provided for contact sheet")
    loaded: list[Image.Image] = []
    for path in images:
        if not path.exists():
            raise RuntimeError(f"Missing image for contact sheet: {path}")
        loaded.append(Image.open(path).convert("RGB"))

    w = loaded[0].width
    h = loaded[0].height
    label_h = 54
    cols = max(1, columns)
    rows = (len(loaded) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + label_h)), (10, 20, 34))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, img in enumerate(loaded):
        r = idx // cols
        c = idx % cols
        x = c * w
        y = r * (h + label_h)
        sheet.paste(img, (x, y))
        title = titles[idx] if idx < len(titles) else img.filename
        draw.rectangle((x, y + h, x + w, y + h + label_h), fill=(12, 26, 46))
        draw.text((x + 12, y + h + 18), title, fill=(232, 239, 252), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _build_fixture_specs() -> list[dict[str, Any]]:
    return [
        {
            "contract_version": "brain_basemodel_v1",
            "scene_id": 101,
            "title": "Fixture 1 - Left Dominant Alpha Power",
            "subtitle": "Deterministic electrode value mapping",
            "footer": "Validation fixture: clinical_glow",
            "extract": {"session_index": 1, "band": "alpha", "metric": "power"},
            "style": {
                "lighting_preset": "clinical_glow",
                "camera_preset": "three_quarter_left",
                "palette": "teal-amber",
            },
            "electrode_values": {
                "F3": 2.2,
                "C3": 2.8,
                "P3": 2.5,
                "F4": -1.4,
                "C4": -1.7,
                "P4": -1.2,
                "Cz": 0.7,
                "Pz": 0.3,
                "Fz": 0.5,
            },
            "coherence_edges": [
                {"a": "F3", "b": "C3", "value": 0.42},
                {"a": "C3", "b": "P3", "value": 0.38},
                {"a": "F4", "b": "C4", "value": 0.18},
            ],
            "value_map": {"type": "zscore", "clip": 2.5},
            "coherence_map": {"type": "magnitude", "min": 0.0, "max": 1.0},
            "animation": {"enabled": False},
        },
        {
            "contract_version": "brain_basemodel_v1",
            "scene_id": 102,
            "title": "Fixture 2 - Central Coherence Surge",
            "subtitle": "Central-parietal network strengthening",
            "footer": "Validation fixture: clinical_glow",
            "extract": {"session_index": 2, "band": "beta", "metric": "coherence"},
            "style": {
                "lighting_preset": "clinical_glow",
                "camera_preset": "frontal",
                "palette": "teal-amber",
            },
            "electrode_values": {
                "C3": 1.8,
                "Cz": 2.2,
                "C4": 1.9,
                "P3": 1.5,
                "Pz": 1.6,
                "P4": 1.7,
                "F3": 0.8,
                "F4": 0.7,
                "O1": -0.3,
                "O2": -0.4,
            },
            "coherence_edges": [
                {"a": "C3", "b": "Cz", "value": 0.86},
                {"a": "Cz", "b": "C4", "value": 0.89},
                {"a": "C3", "b": "P3", "value": 0.78},
                {"a": "P3", "b": "Pz", "value": 0.84},
                {"a": "Pz", "b": "P4", "value": 0.82},
                {"a": "C4", "b": "P4", "value": 0.80},
            ],
            "value_map": {"type": "zscore", "clip": 2.5},
            "coherence_map": {"type": "magnitude", "min": 0.0, "max": 1.0},
            "animation": {"enabled": False},
        },
        {
            "contract_version": "brain_basemodel_v1",
            "scene_id": 103,
            "title": "Fixture 3 - Posterior Reorganization",
            "subtitle": "Posterior coherence with reduced frontal load",
            "footer": "Validation fixture: clinical_glow",
            "extract": {"session_index": 3, "band": "theta", "metric": "coherence"},
            "style": {
                "lighting_preset": "clinical_glow",
                "camera_preset": "three_quarter_right",
                "palette": "teal-amber",
            },
            "electrode_values": {
                "O1": 2.4,
                "P3": 1.8,
                "Pz": 1.6,
                "P4": 1.9,
                "O2": 2.2,
                "F3": -1.1,
                "Fz": -0.8,
                "F4": -1.2,
                "Cz": 0.4,
            },
            "coherence_edges": [
                {"a": "O1", "b": "P3", "value": 0.86},
                {"a": "P3", "b": "Pz", "value": 0.82},
                {"a": "Pz", "b": "P4", "value": 0.84},
                {"a": "P4", "b": "O2", "value": 0.87},
                {"a": "P3", "b": "P4", "value": 0.78},
                {"a": "F3", "b": "F4", "value": 0.24},
            ],
            "value_map": {"type": "zscore", "clip": 2.5},
            "coherence_map": {"type": "magnitude", "min": 0.0, "max": 1.0},
            "animation": {"enabled": False},
        },
    ]


def _build_style_preview_spec(base: dict[str, Any], preset: str) -> dict[str, Any]:
    spec = json.loads(json.dumps(base))
    spec["scene_id"] = int(spec.get("scene_id", 900)) + {"clinical_glow": 0, "calm_precision": 1, "focus_contrast": 2}[preset]
    spec["title"] = f"Style Preview - {preset}"
    spec["style"]["lighting_preset"] = preset
    spec["style"]["camera_preset"] = "three_quarter_left"
    return spec


def main() -> int:
    args = _parse_args()
    blender_bin = str(args.blender_bin).strip()
    if not blender_bin:
        raise RuntimeError("Blender binary not found. Set --blender-bin or BLENDER_BIN.")
    use_gpu = not bool(args.cpu)

    repo = _repo_root()
    template_path = Path(args.template).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    fixtures_dir = out_dir / "fixtures"
    renders_dir = out_dir / "renders"
    checkpoints_dir = out_dir / "checkpoints"
    reports_dir = out_dir / "reports"
    comps_dir = out_dir / "comps"
    for p in (fixtures_dir, renders_dir, checkpoints_dir, reports_dir, comps_dir):
        p.mkdir(parents=True, exist_ok=True)

    build_script = repo / "blender_pipeline" / "scripts" / "build_template.py"
    render_script = repo / "blender_pipeline" / "scripts" / "render_batch.py"
    validate_script = repo / "blender_pipeline" / "scripts" / "validate_template_contract.py"
    montage_path = repo / "blender_pipeline" / "assets" / "montage" / "standard_1020.json"
    font_path = repo / "blender_pipeline" / "assets" / "fonts" / "NotoSans-Regular.ttf"

    if not args.skip_rebuild_template:
        _run(
            [
                blender_bin,
                "-b",
                "--factory-startup",
                "--python",
                str(build_script),
                "--",
                "--output",
                str(template_path),
                "--montage",
                str(montage_path),
                "--font",
                str(font_path),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--samples",
                str(args.samples),
                *(["--gpu"] if use_gpu else []),
            ],
            label="build-template",
        )

    contract_report = reports_dir / "template_contract_report.json"
    _run(
        [
            blender_bin,
            "-b",
            "--factory-startup",
            "--python",
            str(validate_script),
            "--",
            "--template",
            str(template_path),
            "--report",
            str(contract_report),
        ],
        label="validate-template-contract",
    )

    fixture_specs = _build_fixture_specs()
    fixture_jobs: list[dict[str, str]] = []
    fixture_render_paths: list[Path] = []
    fixture_titles: list[str] = []

    for idx, spec in enumerate(fixture_specs, start=1):
        spec_path = fixtures_dir / f"fixture_{idx:02d}.json"
        out_path = renders_dir / f"fixture_{idx:02d}.png"
        _write_json(spec_path, spec)
        fixture_jobs.append({"scene_id": spec["scene_id"], "spec": str(spec_path), "out": str(out_path), "video_out": ""})
        fixture_render_paths.append(out_path)
        fixture_titles.append(str(spec.get("title") or spec_path.stem))

    fixture_batch = fixtures_dir / "fixture_batch_jobs.json"
    _write_json(
        fixture_batch,
        {
            "template": str(template_path),
            "jobs": fixture_jobs,
            "render_settings": {
                "samples": int(args.samples),
                "width": int(args.width),
                "height": int(args.height),
            },
        },
    )

    _run(
        [
            blender_bin,
            "-b",
            "--factory-startup",
            "--python",
            str(render_script),
            "--",
            "--template",
            str(template_path),
            "--batch",
            str(fixture_batch),
            "--samples",
            str(args.samples),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--checkpoint-dir",
            str(checkpoints_dir),
            *(["--gpu"] if use_gpu else []),
        ],
        label="render-fixtures",
    )

    for p in fixture_render_paths:
        if not p.exists():
            raise RuntimeError(f"Fixture render missing: {p}")

    _render_contact_sheet(
        images=fixture_render_paths,
        titles=fixture_titles,
        output_path=comps_dir / "fixtures_contact_sheet.png",
        columns=3,
    )

    # Style-only previews to prove presets are pure visual transforms.
    style_jobs: list[dict[str, str]] = []
    style_paths: list[Path] = []
    style_titles: list[str] = []
    base_style_spec = fixture_specs[0]
    for preset in ("clinical_glow", "calm_precision", "focus_contrast"):
        spec = _build_style_preview_spec(base_style_spec, preset)
        spec_path = fixtures_dir / f"style_preview_{preset}.json"
        out_path = renders_dir / f"style_preview_{preset}.png"
        _write_json(spec_path, spec)
        style_jobs.append({"scene_id": spec["scene_id"], "spec": str(spec_path), "out": str(out_path), "video_out": ""})
        style_paths.append(out_path)
        style_titles.append(preset)

    style_batch = fixtures_dir / "style_preview_batch_jobs.json"
    _write_json(
        style_batch,
        {
            "template": str(template_path),
            "jobs": style_jobs,
            "render_settings": {
                "samples": int(args.samples),
                "width": int(args.width),
                "height": int(args.height),
            },
        },
    )

    _run(
        [
            blender_bin,
            "-b",
            "--factory-startup",
            "--python",
            str(render_script),
            "--",
            "--template",
            str(template_path),
            "--batch",
            str(style_batch),
            "--samples",
            str(args.samples),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            *(["--gpu"] if use_gpu else []),
        ],
        label="render-style-previews",
    )

    _render_contact_sheet(
        images=style_paths,
        titles=style_titles,
        output_path=comps_dir / "style_presets_contact_sheet.png",
        columns=3,
    )

    missing_checkpoints = [
        p
        for p in [
            checkpoints_dir / "scene_101_01_style_text.png",
            checkpoints_dir / "scene_101_02_data_bound.png",
            checkpoints_dir / "scene_102_01_style_text.png",
            checkpoints_dir / "scene_102_02_data_bound.png",
            checkpoints_dir / "scene_103_01_style_text.png",
            checkpoints_dir / "scene_103_02_data_bound.png",
        ]
        if not p.exists()
    ]

    summary = {
        "ok": len(missing_checkpoints) == 0,
        "template": str(template_path),
        "fixture_specs": [
            {
                "name": f"fixture_{idx:02d}",
                "scene_id": spec.get("scene_id"),
                "title": spec.get("title"),
                "spec_path": str(fixtures_dir / f"fixture_{idx:02d}.json"),
                "render_path": str(renders_dir / f"fixture_{idx:02d}.png"),
            }
            for idx, spec in enumerate(fixture_specs, start=1)
        ],
        "style_previews": [
            {
                "preset": preset,
                "spec_path": str(fixtures_dir / f"style_preview_{preset}.json"),
                "render_path": str(renders_dir / f"style_preview_{preset}.png"),
            }
            for preset in ("clinical_glow", "calm_precision", "focus_contrast")
        ],
        "artifacts": {
            "contract_report": str(contract_report),
            "fixture_batch": str(fixture_batch),
            "style_batch": str(style_batch),
            "fixtures_contact_sheet": str(comps_dir / "fixtures_contact_sheet.png"),
            "style_presets_contact_sheet": str(comps_dir / "style_presets_contact_sheet.png"),
            "checkpoints_dir": str(checkpoints_dir),
        },
        "render_mode": "gpu" if use_gpu else "cpu",
        "missing_checkpoints": [str(p) for p in missing_checkpoints],
    }

    summary_path = reports_dir / "validation_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
