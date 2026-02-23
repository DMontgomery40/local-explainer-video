#!/usr/bin/env python3
"""Batch render qEEG scenes from a template .blend and per-scene specs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import bpy
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render qEEG Blender batch jobs")
    parser.add_argument("--template", required=True, help="Path to template .blend")
    parser.add_argument("--batch", required=True, help="Batch JSON path with jobs [{spec,out}]")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--width", type=int, default=1664)
    parser.add_argument("--height", type=int, default=928)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args(argv)


def _configure_render(scene: bpy.types.Scene, *, width: int, height: int, samples: int, gpu: bool) -> None:
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.samples = int(samples)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.device = "GPU" if gpu else "CPU"
    if gpu:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dtype in ("METAL", "CUDA", "OPTIX", "HIP", "ONEAPI"):
                try:
                    prefs.compute_device_type = dtype
                    break
                except Exception:
                    continue
            for device in getattr(prefs, "devices", []):
                try:
                    device.use = True
                except Exception:
                    continue
        except Exception:
            scene.cycles.device = "CPU"


def _configure_still_output(scene: bpy.types.Scene, out_path: Path) -> None:
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(out_path)


def _ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, float(t)))
    return tuple((1.0 - t) * aa + t * bb for aa, bb in zip(a, b))


def _diverging_color(
    value: float,
    *,
    center: float,
    scale: float,
    clip: float = 2.5,
) -> tuple[tuple[float, float, float], float]:
    if scale <= 1e-9:
        scale = 1.0
    clip = max(0.25, float(clip))
    z = (float(value) - center) / scale
    z = max(-clip, min(clip, z))
    t = (z + clip) / (2.0 * clip)
    blue = (0.18, 0.44, 0.95)
    white = (0.94, 0.96, 0.99)
    red = (0.93, 0.26, 0.20)
    color = _lerp(blue, white, t * 2.0) if t <= 0.5 else _lerp(white, red, (t - 0.5) * 2.0)
    emission = 0.35 + (abs(z) / clip) * 1.9
    return color, emission


def _coherence_color(value: float, *, minimum: float, maximum: float) -> tuple[tuple[float, float, float], float, float]:
    if maximum <= minimum:
        maximum = minimum + 1.0
    n = (float(value) - minimum) / (maximum - minimum)
    n = max(0.0, min(1.0, n))
    low = (0.16, 0.65, 0.98)
    high = (1.0, 0.42, 0.14)
    color = _lerp(low, high, n)
    emission = 0.6 + 2.0 * n
    width = 0.008 + 0.025 * n
    return color, emission, width


def _new_emission_material(name: str, *, color: tuple[float, float, float], emission: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
    bsdf.inputs["Emission Strength"].default_value = float(emission)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _electrode_lookup() -> dict[str, bpy.types.Object]:
    col = bpy.data.collections.get("COL_ELECTRODES")
    if col is None:
        return {}
    out: dict[str, bpy.types.Object] = {}
    for obj in col.objects:
        if not obj.name.startswith("E_"):
            continue
        label = obj.name[2:]
        out[label] = obj
        out[label.upper()] = obj
    return out


def _normalize_label(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace(" ", "")
    alias = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
    }
    up = text.upper()
    if up in alias:
        return alias[up]
    if len(up) >= 2 and up[0].isalpha():
        return up[0] + up[1:].lower() if up.endswith("Z") else up[0] + up[1:]
    return text


def _apply_text(spec: dict[str, Any]) -> None:
    for key, obj_name in (("title", "TXT_TITLE"), ("subtitle", "TXT_SUBTITLE"), ("footer", "TXT_FOOTER")):
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != "FONT":
            continue
        obj.data.body = str(spec.get(key) or "")


def _apply_electrodes(spec: dict[str, Any]) -> None:
    raw_values = spec.get("electrode_values")
    if not isinstance(raw_values, dict):
        raw_values = {}
    values: dict[str, float] = {}
    for key, value in raw_values.items():
        label = _normalize_label(key)
        if not label:
            continue
        try:
            values[label] = float(value)
        except Exception:
            continue

    samples = list(values.values())
    mean = statistics.fmean(samples) if samples else 0.0
    std = statistics.pstdev(samples) if len(samples) > 1 else 1.0
    value_map = spec.get("value_map")
    value_map_cfg = value_map if isinstance(value_map, dict) else {}
    map_type = str(value_map_cfg.get("type") or "zscore").strip().lower()
    clip = float(value_map_cfg.get("clip", 2.5))

    if map_type == "minmax":
        if samples:
            min_v = float(value_map_cfg.get("min", min(samples)))
            max_v = float(value_map_cfg.get("max", max(samples)))
        else:
            min_v, max_v = 0.0, 1.0
        mean = (min_v + max_v) * 0.5
        std = max(1e-6, (max_v - min_v) * 0.5)
    elif map_type == "fixed":
        mean = float(value_map_cfg.get("center", 0.0))
        std = max(1e-6, float(value_map_cfg.get("scale", 1.0)))

    lookup = _electrode_lookup()
    neutral_color = (0.80, 0.84, 0.90)
    for label_key, obj in lookup.items():
        # Skip the uppercase aliases in lookup to avoid duplicate material assignment.
        if obj.name != f"E_{label_key}":
            continue
        label = obj.name[2:]
        value = values.get(label)
        if value is None:
            color = neutral_color
            emission = 0.2
        else:
            color, emission = _diverging_color(value, center=mean, scale=std, clip=clip)
        mat = _new_emission_material(
            f"MAT_JOB_E_{obj.name}",
            color=color,
            emission=emission,
        )
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def _apply_coherence_lines(spec: dict[str, Any]) -> None:
    col = _ensure_collection("COL_LINES")
    edges = spec.get("coherence_edges")
    if not isinstance(edges, list):
        edges = []

    lookup = _electrode_lookup()
    values: list[float] = []
    parsed_edges: list[tuple[str, str, float]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        a = _normalize_label(edge.get("a"))
        b = _normalize_label(edge.get("b"))
        if not a or not b or a == b:
            continue
        try:
            value = float(edge.get("value"))
        except Exception:
            continue
        if a not in lookup or b not in lookup:
            continue
        parsed_edges.append((a, b, value))
        values.append(value)

    if not parsed_edges:
        return

    coherence_map = spec.get("coherence_map")
    coherence_map_cfg = coherence_map if isinstance(coherence_map, dict) else {}
    map_type = str(coherence_map_cfg.get("type") or "magnitude").strip().lower()
    vmin = float(coherence_map_cfg.get("min", min(values)))
    vmax = float(coherence_map_cfg.get("max", max(values)))
    mean = statistics.fmean(values) if values else 0.0
    std = statistics.pstdev(values) if len(values) > 1 else 1.0
    clip = float(coherence_map_cfg.get("clip", 2.5))

    for idx, (a, b, value) in enumerate(parsed_edges):
        start_obj = lookup[a]
        end_obj = lookup[b]
        start = start_obj.matrix_world.translation.copy()
        end = end_obj.matrix_world.translation.copy()
        midpoint = (start + end) * 0.5
        midpoint.z += 0.35 + 0.12 * min(2.5, (start - end).length)

        if map_type == "zscore":
            color, emission = _diverging_color(value, center=mean, scale=std, clip=clip)
            width = 0.01 + 0.02 * min(1.0, abs((value - mean) / max(1e-6, std)) / max(0.25, clip))
        else:
            color, emission, width = _coherence_color(value, minimum=vmin, maximum=vmax)
        curve_data = bpy.data.curves.new(name=f"CURVE_JOB_{idx:04d}", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.resolution_u = 18
        curve_data.bevel_depth = width
        curve_data.bevel_resolution = 5
        spline = curve_data.splines.new(type="POLY")
        spline.points.add(2)
        spline.points[0].co = (start.x, start.y, start.z, 1.0)
        spline.points[1].co = (midpoint.x, midpoint.y, midpoint.z, 1.0)
        spline.points[2].co = (end.x, end.y, end.z, 1.0)

        obj = bpy.data.objects.new(f"LINE_JOB_{idx:04d}", curve_data)
        col.objects.link(obj)
        mat = _new_emission_material(f"MAT_JOB_LINE_{idx:04d}", color=color, emission=emission)
        obj.data.materials.append(mat)


def _cleanup_after_job() -> None:
    lines = bpy.data.collections.get("COL_LINES")
    if lines is not None:
        for obj in list(lines.objects):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and getattr(data, "users", 0) == 0:
                if isinstance(data, bpy.types.Curve):
                    bpy.data.curves.remove(data)
    for mat in list(bpy.data.materials):
        if mat.name.startswith("MAT_JOB_") and mat.users == 0:
            bpy.data.materials.remove(mat)


def _set_linear_interpolation(obj: bpy.types.ID) -> None:
    anim = getattr(obj, "animation_data", None)
    action = getattr(anim, "action", None) if anim else None
    if action is None:
        return
    fcurves = getattr(action, "fcurves", None)
    if not fcurves:
        # Blender 5+ animation data model may not expose fcurves directly.
        # Deterministic keyframe values still apply; interpolation fallback is acceptable.
        return
    for fcurve in fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = "LINEAR"


def _material_emission_socket(mat: bpy.types.Material):
    if not mat.use_nodes:
        return None
    nt = mat.node_tree
    if nt is None:
        return None
    for node in nt.nodes:
        if node.type == "BSDF_PRINCIPLED":
            sock = node.inputs.get("Emission Strength")
            if sock is not None:
                return sock
    return None


def _apply_animation(scene: bpy.types.Scene, spec: dict[str, Any]) -> tuple[bool, int]:
    anim_cfg = spec.get("animation")
    if not isinstance(anim_cfg, dict) or not bool(anim_cfg.get("enabled", False)):
        return False, int(scene.render.fps or 24)

    duration = max(1.0, float(anim_cfg.get("duration_sec", 5.0)))
    fps = max(1, int(float(anim_cfg.get("fps", 24))))
    frames = max(2, int(round(duration * fps)))
    orbit_deg = float(anim_cfg.get("camera_orbit_deg", 14.0))
    pulse_hz = max(0.0, float(anim_cfg.get("pulse_hz", 0.45)))
    pulse_depth = max(0.0, float(anim_cfg.get("pulse_depth", 0.28)))

    scene.frame_start = 1
    scene.frame_end = frames
    scene.frame_set(1)
    scene.render.fps = fps

    cam = bpy.data.objects.get("CAM_MAIN")
    target = bpy.data.objects.get("CAM_TARGET")
    if cam is not None and target is not None:
        rel = cam.location.copy() - target.location.copy()
        orbit_rad = math.radians(orbit_deg)

        def _rot_z(theta: float) -> Vector:
            c = math.cos(theta)
            s = math.sin(theta)
            x = rel.x * c - rel.y * s
            y = rel.x * s + rel.y * c
            return Vector((x, y, rel.z)) + target.location.copy()

        cam.location = _rot_z(-orbit_rad * 0.5)
        cam.keyframe_insert(data_path="location", frame=1)
        cam.location = _rot_z(orbit_rad * 0.5)
        cam.keyframe_insert(data_path="location", frame=frames)
        _set_linear_interpolation(cam)

    materials: list[bpy.types.Material] = []
    for col_name in ("COL_ELECTRODES", "COL_LINES"):
        col = bpy.data.collections.get(col_name)
        if col is None:
            continue
        for obj in col.objects:
            mats = getattr(obj.data, "materials", None)
            if mats is None:
                continue
            for mat in mats:
                if mat is None:
                    continue
                if not mat.name.startswith("MAT_JOB_"):
                    continue
                if mat not in materials:
                    materials.append(mat)

    key_steps = max(8, min(28, int(duration * pulse_hz * 16) if pulse_hz > 0 else 8))
    for mat in materials:
        socket = _material_emission_socket(mat)
        if socket is None:
            continue
        base = float(socket.default_value)
        for i in range(key_steps + 1):
            frame = 1 + int(round((frames - 1) * (i / max(1, key_steps))))
            t = (frame - 1) / float(fps)
            pulse = 1.0 + pulse_depth * math.sin(2.0 * math.pi * pulse_hz * t)
            socket.default_value = base * pulse
            socket.keyframe_insert(data_path="default_value", frame=frame)
        _set_linear_interpolation(mat)

    scene.frame_set(1)
    return True, fps


def _render_job(
    scene: bpy.types.Scene,
    *,
    spec_path: Path,
    out_path: Path,
    video_out_path: Path | None,
) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise RuntimeError(f"Invalid scene spec JSON (expected object): {spec_path}")
    _apply_text(spec)
    _apply_electrodes(spec)
    _apply_coherence_lines(spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_still_output(scene, out_path)
    bpy.context.view_layer.update()
    scene.frame_set(1)
    bpy.ops.render.render(write_still=True)

    has_animation, fps = _apply_animation(scene, spec)
    if has_animation and video_out_path is not None:
        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="blender_frames_") as tmp_dir_raw:
            frames_dir = Path(tmp_dir_raw)
            scene.render.image_settings.file_format = "PNG"
            scene.render.filepath = str(frames_dir / "frame_")
            bpy.ops.render.render(animation=True)
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(int(max(1, fps))),
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not video_out_path.exists():
                details = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    f"Failed encoding animation video with ffmpeg: {video_out_path}\n"
                    + (details[-1000:] if details else "No ffmpeg diagnostics available.")
                )

    _cleanup_after_job()


def main() -> int:
    args = _parse_args()
    template_path = Path(args.template).expanduser().resolve()
    batch_path = Path(args.batch).expanduser().resolve()
    if not template_path.exists():
        raise RuntimeError(f"Template .blend not found: {template_path}")
    if not batch_path.exists():
        raise RuntimeError(f"Batch job JSON not found: {batch_path}")

    batch_payload = json.loads(batch_path.read_text(encoding="utf-8"))
    jobs = batch_payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"Batch JSON has no jobs: {batch_path}")

    for idx, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise RuntimeError(f"Invalid job entry at index {idx - 1}: expected object")
        spec_path = Path(str(job.get("spec") or "")).expanduser().resolve()
        out_path = Path(str(job.get("out") or "")).expanduser().resolve()
        video_out_raw = str(job.get("video_out") or "").strip()
        video_out_path = Path(video_out_raw).expanduser().resolve() if video_out_raw else None
        if not spec_path.exists():
            raise RuntimeError(f"Spec file does not exist: {spec_path}")
        bpy.ops.wm.open_mainfile(filepath=str(template_path))
        scene = bpy.context.scene
        _configure_render(scene, width=args.width, height=args.height, samples=args.samples, gpu=bool(args.gpu))
        _ensure_collection("COL_LINES")
        print(
            f"[render_batch] Rendering {idx}/{len(jobs)} spec={spec_path} out={out_path}"
            + (f" video={video_out_path}" if video_out_path else "")
        )
        _render_job(
            scene,
            spec_path=spec_path,
            out_path=out_path,
            video_out_path=video_out_path,
        )
        if not out_path.exists():
            raise RuntimeError(f"Render finished but output missing: {out_path}")
        if video_out_path is not None and not video_out_path.exists():
            raise RuntimeError(f"Animation render finished but output missing: {video_out_path}")

    print(f"[render_batch] Completed {len(jobs)} job(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[render_batch] ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
