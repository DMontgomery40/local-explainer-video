#!/usr/bin/env python3
"""Phase-1 POC render for BAR archetype using deterministic templates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.template_pipeline.renderer import render_scene_to_image


def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32)
    return float(np.abs(arr_a - arr_b).mean())


def _scene_payload() -> dict:
    return {
        "id": 4,
        "uid": "pocscene",
        "title": "Signal Strength: Nearly Doubled",
        "narration": (
            "The brain's detection signal nearly doubled across sessions. "
            "This deterministic slide should match the cinematic BAR style while keeping text exact."
        ),
        "scene_type": "bar_volume_chart",
        "structured_data": {
            "metric": "P300 Signal Voltage",
            "unit": "µV",
            "bars": [
                {"label": "Session 1", "value": 13.1, "unit": "µV"},
                {"label": "Session 2", "value": 22.9, "unit": "µV"},
                {"label": "Session 3", "value": 24.0, "unit": "µV"},
            ],
            "target_band": {"min": 6, "max": 14, "unit": "µV"},
            "trend": "ascending",
        },
        "visual_prompt": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render deterministic BAR POC and compare outputs.")
    parser.add_argument(
        "--project",
        default="09-05-1954-0",
        help="Project ID under ./projects (default: 09-05-1954-0).",
    )
    parser.add_argument(
        "--reference-image",
        default="",
        help="Optional path to reference image for comparison (defaults to projects/<id>/images/scene_004.png).",
    )
    args = parser.parse_args()

    repo = REPO_ROOT
    project_dir = repo / "projects" / str(args.project)
    original = (
        Path(args.reference_image).expanduser().resolve()
        if str(args.reference_image).strip()
        else project_dir / "images" / "scene_004.png"
    )
    if not original.exists():
        raise SystemExit(f"Missing source image: {original}")

    out_dir = project_dir / "poc_template_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = _scene_payload()
    pillow_out = out_dir / "scene_004_template_pillow.png"
    render_scene_to_image(scene.copy(), project_dir=project_dir, backend="pillow", output_path=pillow_out)

    cairo_out = out_dir / "scene_004_template_cairo.png"
    cairo_error = None
    try:
        render_scene_to_image(scene.copy(), project_dir=project_dir, backend="cairo", output_path=cairo_out)
    except Exception as e:  # pragma: no cover - optional backend
        cairo_error = str(e)
        try:
            cairo_out.unlink(missing_ok=True)
        except Exception:
            pass

    orig_img = Image.open(original).convert("RGB")
    pillow_img = Image.open(pillow_out).convert("RGB")
    mad_pillow = _mean_abs_diff(orig_img, pillow_img)

    if cairo_out.exists():
        cairo_img = Image.open(cairo_out).convert("RGB")
        mad_cairo = _mean_abs_diff(orig_img, cairo_img)
    else:
        cairo_img = Image.new("RGB", orig_img.size, (20, 25, 35))
        draw = ImageDraw.Draw(cairo_img)
        draw.text((120, 420), f"Cairo unavailable\n{cairo_error or ''}", fill=(220, 235, 245))
        mad_cairo = None

    strip = Image.new("RGB", (orig_img.width * 3, orig_img.height), (0, 0, 0))
    strip.paste(orig_img, (0, 0))
    strip.paste(pillow_img, (orig_img.width, 0))
    strip.paste(cairo_img, (orig_img.width * 2, 0))
    draw = ImageDraw.Draw(strip)
    draw.rectangle((0, 0, strip.width, 74), fill=(0, 0, 0, 180))
    draw.text((30, 24), "Original", fill=(255, 255, 255))
    draw.text((orig_img.width + 30, 24), f"Pillow MAD={mad_pillow:.1f}", fill=(255, 255, 255))
    draw.text(
        (orig_img.width * 2 + 30, 24),
        f"Cairo MAD={mad_cairo:.1f}" if mad_cairo is not None else "Cairo unavailable",
        fill=(255, 255, 255),
    )
    strip_path = out_dir / "scene_004_comparison_strip.png"
    strip.save(strip_path, format="PNG")

    report = {
        "original": str(original),
        "pillow_output": str(pillow_out),
        "cairo_output": str(cairo_out) if cairo_out.exists() else None,
        "comparison_strip": str(strip_path),
        "mean_abs_diff_pillow": mad_pillow,
        "mean_abs_diff_cairo": mad_cairo,
        "cairo_error": cairo_error,
        "scene_payload": scene,
    }
    report_path = out_dir / "poc_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {strip_path}")


if __name__ == "__main__":
    main()
