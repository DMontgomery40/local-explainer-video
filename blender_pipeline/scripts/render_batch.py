#!/usr/bin/env python3
"""Batch render deterministic qEEG brain scenes from a template .blend and scene specs."""

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
from typing import Any, Mapping

import bpy
from mathutils import Vector


COL_WORLD = "COL_WORLD"
COL_ELECTRODES = "COL_ELECTRODES"
COL_ELECTRODE_ANCHORS = "COL_ELECTRODE_ANCHORS"
COL_LINES = "COL_LINES"
COL_TEXT = "COL_TEXT"
COL_GUIDES = "COL_GUIDES"

OBJ_CAM = "CAM_MAIN"
OBJ_CAM_TARGET = "CAM_TARGET"
OBJ_BRAIN = "MESH_BRAIN"
OBJ_HEAD = "MESH_HEAD_SHELL"
OBJ_FOG = "VOL_ATMOSPHERE"

MAT_BRAIN = "MAT_BRAIN_BASE"
MAT_HEAD = "MAT_HEAD_SHELL"
MAT_FOG = "MAT_FOG_VOLUME"
MAT_TEXT = "MAT_TEXT_BASE"
MAT_ELECTRODE = "MAT_ELECTRODE_BASE"
MAT_LABEL = "MAT_LABEL_BASE"
MAT_LINE = "MAT_LINE_BASE"

REQUIRED_COLLECTIONS = (
    COL_WORLD,
    COL_ELECTRODES,
    COL_ELECTRODE_ANCHORS,
    COL_LINES,
    COL_TEXT,
    COL_GUIDES,
)
REQUIRED_OBJECTS = (
    OBJ_CAM,
    OBJ_CAM_TARGET,
    OBJ_BRAIN,
    OBJ_HEAD,
    OBJ_FOG,
    "LIGHT_KEY",
    "LIGHT_FILL",
    "LIGHT_RIM",
    "LIGHT_OVERHEAD",
    "TXT_TITLE",
    "TXT_SUBTITLE",
    "TXT_FOOTER",
)
REQUIRED_MATERIALS = (
    MAT_BRAIN,
    MAT_HEAD,
    MAT_FOG,
    MAT_TEXT,
    MAT_ELECTRODE,
    MAT_LABEL,
    MAT_LINE,
)

CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "three_quarter_left": {"location": (-5.4, -9.8, 3.8), "lens": 38.0},
    "three_quarter_right": {"location": (5.4, -9.8, 3.8), "lens": 38.0},
    "frontal": {"location": (0.0, -12.0, 1.7), "lens": 34.0},
    "top_center": {"location": (0.0, -8.8, 5.8), "lens": 34.0},
}

LIGHTING_PRESETS: dict[str, dict[str, Any]] = {
    "clinical_glow": {
        "world_color": (0.006, 0.012, 0.028, 1.0),
        "world_strength": 1.02,
        "fog_color": (0.20, 0.40, 0.72, 1.0),
        "fog_density": 0.022,
        "key": {"energy": 980.0, "color": (0.76, 0.86, 1.0)},
        "fill": {"energy": 430.0, "color": (0.42, 0.54, 0.92)},
        "rim": {"energy": 640.0, "color": (0.98, 0.55, 0.28)},
        "overhead": {"energy": 280.0, "color": (0.60, 0.74, 1.0)},
        "brain": {
            "base": (0.12, 0.26, 0.52, 1.0),
            "emission": (0.16, 0.40, 0.80, 1.0),
            "emission_strength": 0.20,
            "roughness": 0.30,
        },
        "head": {
            "base": (0.05, 0.10, 0.18, 1.0),
            "emission": (0.07, 0.20, 0.34, 1.0),
            "emission_strength": 0.16,
            "roughness": 0.26,
        },
    },
    "calm_precision": {
        "world_color": (0.008, 0.016, 0.034, 1.0),
        "world_strength": 0.88,
        "fog_color": (0.16, 0.28, 0.56, 1.0),
        "fog_density": 0.014,
        "key": {"energy": 820.0, "color": (0.70, 0.83, 1.0)},
        "fill": {"energy": 380.0, "color": (0.50, 0.66, 0.95)},
        "rim": {"energy": 500.0, "color": (0.76, 0.86, 1.0)},
        "overhead": {"energy": 220.0, "color": (0.56, 0.72, 1.0)},
        "brain": {
            "base": (0.11, 0.24, 0.48, 1.0),
            "emission": (0.13, 0.36, 0.76, 1.0),
            "emission_strength": 0.16,
            "roughness": 0.34,
        },
        "head": {
            "base": (0.06, 0.11, 0.20, 1.0),
            "emission": (0.07, 0.18, 0.33, 1.0),
            "emission_strength": 0.12,
            "roughness": 0.30,
        },
    },
    "focus_contrast": {
        "world_color": (0.004, 0.010, 0.022, 1.0),
        "world_strength": 1.10,
        "fog_color": (0.15, 0.27, 0.50, 1.0),
        "fog_density": 0.010,
        "key": {"energy": 1140.0, "color": (0.82, 0.92, 1.0)},
        "fill": {"energy": 300.0, "color": (0.34, 0.44, 0.82)},
        "rim": {"energy": 720.0, "color": (1.0, 0.50, 0.22)},
        "overhead": {"energy": 320.0, "color": (0.74, 0.86, 1.0)},
        "brain": {
            "base": (0.10, 0.21, 0.42, 1.0),
            "emission": (0.12, 0.30, 0.64, 1.0),
            "emission_strength": 0.12,
            "roughness": 0.28,
        },
        "head": {
            "base": (0.05, 0.09, 0.17, 1.0),
            "emission": (0.07, 0.16, 0.29, 1.0),
            "emission_strength": 0.10,
            "roughness": 0.24,
        },
    },
}

PALETTES: dict[str, dict[str, tuple[float, float, float]]] = {
    "teal-amber": {
        "low": (0.16, 0.68, 0.95),
        "mid": (0.94, 0.96, 0.99),
        "high": (1.0, 0.56, 0.22),
    },
    "ice-white": {
        "low": (0.32, 0.76, 1.0),
        "mid": (0.98, 0.99, 1.0),
        "high": (0.62, 0.90, 1.0),
    },
    "cyan-orange": {
        "low": (0.16, 0.84, 1.0),
        "mid": (0.96, 0.97, 0.99),
        "high": (1.0, 0.48, 0.18),
    },
}


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
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Optional directory for checkpoint renders after major update phases",
    )
    return parser.parse_args(argv)


def _configure_render(scene: bpy.types.Scene, *, width: int, height: int, samples: int, gpu: bool) -> None:
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.render.use_persistent_data = True
    except Exception:
        pass
    scene.cycles.samples = int(samples)
    scene.cycles.use_adaptive_sampling = True
    _set_numeric_if_present(scene.cycles, "adaptive_threshold", 0.01)
    _set_numeric_if_present(scene.cycles, "max_bounces", 6)
    _set_numeric_if_present(scene.cycles, "diffuse_bounces", 4)
    _set_numeric_if_present(scene.cycles, "glossy_bounces", 4)
    _set_numeric_if_present(scene.cycles, "transmission_bounces", 6)
    _set_numeric_if_present(scene.cycles, "transparent_max_bounces", 6)
    _set_numeric_if_present(scene.cycles, "volume_bounces", 0)
    _set_bool_if_present(scene.cycles, "caustics_reflective", False)
    _set_bool_if_present(scene.cycles, "caustics_refractive", False)
    _set_bool_if_present(scene.cycles, "use_light_tree", True)
    scene.cycles.device = "GPU" if gpu else "CPU"
    _set_bool_if_present(scene.cycles, "use_denoising", True)
    _set_bool_if_present(scene.cycles, "use_preview_denoising", True)
    _set_preferred_enum(scene.cycles, "denoiser", ("OPENIMAGEDENOISE", "OPTIX", "NLM"))
    _set_preferred_enum(scene.cycles, "preview_denoiser", ("OPENIMAGEDENOISE", "OPTIX", "NLM"))
    _set_preferred_enum(scene.cycles, "denoising_quality", ("BALANCED", "HIGH", "FAST"))
    _set_preferred_enum(scene.cycles, "preview_denoising_quality", ("BALANCED", "HIGH", "FAST"))
    _set_preferred_enum(scene.cycles, "denoising_prefilter", ("FAST", "ACCURATE", "NONE"))
    _set_preferred_enum(scene.cycles, "preview_denoising_prefilter", ("FAST", "ACCURATE", "NONE"))
    _set_preferred_enum(scene.cycles, "denoising_input_passes", ("RGB_ALBEDO", "RGB_ALBEDO_NORMAL", "RGB"))
    _set_preferred_enum(scene.cycles, "preview_denoising_input_passes", ("RGB_ALBEDO", "RGB_ALBEDO_NORMAL", "RGB"))
    _set_bool_if_present(scene.cycles, "denoising_use_gpu", bool(gpu))
    _set_bool_if_present(scene.cycles, "preview_denoising_use_gpu", bool(gpu))
    _set_bool_if_present(scene.cycles, "use_gpu_denoising", bool(gpu))
    if gpu:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            try:
                prefs.get_devices()
            except Exception:
                pass
            _set_preferred_enum(prefs, "compute_device_type", ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI", "NONE"))
            _set_preferred_enum(prefs, "metalrt", ("ON", "AUTO", "OFF"))
            _set_preferred_enum(prefs, "kernel_optimization_level", ("FULL", "INTERSECT", "OFF"))
            for device in getattr(prefs, "devices", []):
                try:
                    dev_type = str(getattr(device, "type", "")).upper()
                    device.use = dev_type != "CPU"
                except Exception:
                    continue
            if not any(bool(getattr(device, "use", False)) for device in getattr(prefs, "devices", [])):
                for device in getattr(prefs, "devices", []):
                    try:
                        device.use = True
                    except Exception:
                        continue
        except Exception:
            scene.cycles.device = "CPU"


def _enum_identifiers(obj: bpy.types.ID, prop_name: str) -> set[str]:
    try:
        prop = obj.bl_rna.properties.get(prop_name)
        if prop is None:
            return set()
        return {str(item.identifier) for item in prop.enum_items}
    except Exception:
        return set()


def _set_bool_if_present(obj: bpy.types.ID, prop_name: str, value: bool) -> None:
    try:
        if hasattr(obj, prop_name):
            setattr(obj, prop_name, bool(value))
    except Exception:
        pass


def _set_numeric_if_present(obj: bpy.types.ID, prop_name: str, value: float) -> None:
    try:
        if hasattr(obj, prop_name):
            setattr(obj, prop_name, value)
    except Exception:
        pass


def _set_preferred_enum(obj: bpy.types.ID, prop_name: str, preferred: tuple[str, ...]) -> None:
    options = _enum_identifiers(obj, prop_name)
    for candidate in preferred:
        if options and candidate not in options:
            continue
        try:
            setattr(obj, prop_name, candidate)
            return
        except Exception:
            continue


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


def _palette(name: str) -> dict[str, tuple[float, float, float]]:
    key = str(name or "teal-amber").strip().lower()
    return PALETTES.get(key, PALETTES["teal-amber"])


def _diverging_color(
    value: float,
    *,
    center: float,
    scale: float,
    clip: float = 2.5,
    palette_name: str = "teal-amber",
) -> tuple[tuple[float, float, float], float]:
    if scale <= 1e-9:
        scale = 1.0
    clip = max(0.25, float(clip))
    z = (float(value) - center) / scale
    z = max(-clip, min(clip, z))
    t = (z + clip) / (2.0 * clip)

    pal = _palette(palette_name)
    low = pal["low"]
    mid = pal["mid"]
    high = pal["high"]
    color = _lerp(low, mid, t * 2.0) if t <= 0.5 else _lerp(mid, high, (t - 0.5) * 2.0)
    emission = 0.45 + (abs(z) / clip) * 2.1
    return color, emission


def _coherence_color(
    value: float,
    *,
    minimum: float,
    maximum: float,
    palette_name: str = "teal-amber",
) -> tuple[tuple[float, float, float], float, float]:
    if maximum <= minimum:
        maximum = minimum + 1.0
    n = (float(value) - minimum) / (maximum - minimum)
    n = max(0.0, min(1.0, n))
    pal = _palette(palette_name)
    low = pal["low"]
    high = pal["high"]
    color = _lerp(low, high, n)
    emission = 0.75 + 2.2 * n
    width = 0.004 + 0.010 * n
    return color, emission, width


def _set_principled_value(node: bpy.types.Node, socket_name: str, value: object) -> None:
    sock = node.inputs.get(socket_name)
    if sock is not None:
        sock.default_value = value


def _set_material_principled(
    mat: bpy.types.Material,
    *,
    base_color: tuple[float, float, float, float],
    roughness: float,
    emission_color: tuple[float, float, float, float],
    emission_strength: float,
    alpha: float | None = None,
) -> None:
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    principled: bpy.types.Node | None = None
    output: bpy.types.Node | None = None
    for node in nt.nodes:
        if node.type == "BSDF_PRINCIPLED":
            principled = node
        elif node.type == "OUTPUT_MATERIAL":
            output = node
    if principled is None:
        principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    if output is None:
        output = nt.nodes.new("ShaderNodeOutputMaterial")
    if not any(link.from_node == principled and link.to_node == output for link in nt.links):
        nt.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    _set_principled_value(principled, "Base Color", base_color)
    _set_principled_value(principled, "Roughness", float(roughness))
    _set_principled_value(principled, "Emission Color", emission_color)
    _set_principled_value(principled, "Emission Strength", float(emission_strength))
    if alpha is not None:
        _set_principled_value(principled, "Alpha", float(alpha))
        mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"


def _upsert_emission_material(
    name: str,
    *,
    color: tuple[float, float, float],
    emission: float,
    roughness: float = 0.28,
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    _set_material_principled(
        mat,
        base_color=(*color, 1.0),
        roughness=float(roughness),
        emission_color=(*color, 1.0),
        emission_strength=float(emission),
    )
    return mat


def _electrode_lookup() -> dict[str, bpy.types.Object]:
    col = bpy.data.collections.get(COL_ELECTRODES)
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


def _label_lookup() -> dict[str, bpy.types.Object]:
    col = bpy.data.collections.get(COL_TEXT)
    if col is None:
        return {}
    out: dict[str, bpy.types.Object] = {}
    for obj in col.objects:
        if not obj.name.startswith("LBL_"):
            continue
        label = obj.name[4:]
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


def _style_cfg(spec: dict[str, Any]) -> dict[str, str]:
    raw = spec.get("style")
    style = raw if isinstance(raw, Mapping) else {}
    lighting = str(style.get("lighting_preset") or "clinical_glow").strip().lower()
    camera = str(style.get("camera_preset") or "three_quarter_left").strip().lower()
    palette = str(style.get("palette") or "teal-amber").strip().lower()
    if lighting not in LIGHTING_PRESETS:
        lighting = "clinical_glow"
    if camera not in CAMERA_PRESETS:
        camera = "three_quarter_left"
    if palette not in PALETTES:
        palette = "teal-amber"
    return {"lighting_preset": lighting, "camera_preset": camera, "palette": palette}


def _apply_style(scene: bpy.types.Scene, spec: dict[str, Any]) -> dict[str, str]:
    style = _style_cfg(spec)
    lighting = LIGHTING_PRESETS[style["lighting_preset"]]
    camera_preset = CAMERA_PRESETS[style["camera_preset"]]

    cam = bpy.data.objects.get(OBJ_CAM)
    if cam is not None and cam.type == "CAMERA":
        cam.location = Vector(camera_preset["location"])
        cam.data.lens = float(camera_preset["lens"])

    light_map = {
        "LIGHT_KEY": lighting["key"],
        "LIGHT_FILL": lighting["fill"],
        "LIGHT_RIM": lighting["rim"],
        "LIGHT_OVERHEAD": lighting["overhead"],
    }
    for name, cfg in light_map.items():
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "LIGHT":
            continue
        obj.data.energy = float(cfg["energy"])
        obj.data.color = tuple(cfg["color"])

    world = scene.world
    if world is not None and world.use_nodes and world.node_tree is not None:
        bg = world.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs["Color"].default_value = lighting["world_color"]
            bg.inputs["Strength"].default_value = float(lighting["world_strength"])

    fog = bpy.data.objects.get(OBJ_FOG)
    if fog is not None and fog.data is not None and fog.data.materials:
        mat = fog.data.materials[0]
        if mat and mat.use_nodes and mat.node_tree is not None:
            for node in mat.node_tree.nodes:
                if node.type == "VOLUME_SCATTER":
                    node.inputs["Color"].default_value = lighting["fog_color"]
                    node.inputs["Density"].default_value = float(lighting["fog_density"])

    mat_brain = bpy.data.materials.get(MAT_BRAIN)
    if mat_brain is not None:
        _set_material_principled(
            mat_brain,
            base_color=lighting["brain"]["base"],
            roughness=float(lighting["brain"]["roughness"]),
            emission_color=lighting["brain"]["emission"],
            emission_strength=float(lighting["brain"]["emission_strength"]),
        )

    mat_head = bpy.data.materials.get(MAT_HEAD)
    if mat_head is not None:
        _set_material_principled(
            mat_head,
            base_color=lighting["head"]["base"],
            roughness=float(lighting["head"]["roughness"]),
            emission_color=lighting["head"]["emission"],
            emission_strength=float(lighting["head"]["emission_strength"]),
            alpha=0.08,
        )

    return style


def _apply_text(spec: dict[str, Any]) -> None:
    for key, obj_name in (("title", "TXT_TITLE"), ("subtitle", "TXT_SUBTITLE"), ("footer", "TXT_FOOTER")):
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != "FONT":
            continue
        obj.data.body = str(spec.get(key) or "")


def _apply_electrodes(spec: dict[str, Any], *, palette_name: str) -> None:
    raw_values = spec.get("electrode_values")
    if not isinstance(raw_values, Mapping):
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
    value_map_cfg = value_map if isinstance(value_map, Mapping) else {}
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
    labels = _label_lookup()
    neutral_color = (0.82, 0.86, 0.92)
    for label_key, obj in lookup.items():
        if obj.name != f"E_{label_key}":
            continue
        label = obj.name[2:]
        value = values.get(label)
        if value is None:
            color = neutral_color
            emission = 0.08
            label_emission = 0.35
            label_color = (0.94, 0.96, 1.0)
            scale = 0.35
            show_label = False
        else:
            color, emission = _diverging_color(
                value,
                center=mean,
                scale=std,
                clip=clip,
                palette_name=palette_name,
            )
            label_emission = max(0.95, emission * 0.62)
            label_color = color
            if std > 1e-6:
                z_abs = abs((value - mean) / std)
            else:
                z_abs = abs(value)
            scale = 1.0 + min(0.55, z_abs * 0.18)
            show_label = True

        mat = _upsert_emission_material(
            f"MAT_JOB_E_{label}",
            color=color,
            emission=emission,
            roughness=0.18,
        )
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.scale = (scale, scale, scale)

        label_obj = labels.get(label)
        if label_obj is not None and label_obj.type == "FONT":
            label_obj.data.body = label
            label_obj.hide_render = not show_label
            label_obj.hide_viewport = not show_label
            label_mat = _upsert_emission_material(
                f"MAT_JOB_LBL_{label}",
                color=label_color,
                emission=label_emission,
                roughness=0.12,
            )
            label_obj.data.materials.clear()
            label_obj.data.materials.append(label_mat)
            label_obj.data.size = 0.185 if show_label else 0.110


def _coherence_edge_name(a: str, b: str) -> str:
    return f"LINE_{a}_{b}".replace("-", "_")


def _delete_object_and_data(obj: bpy.types.Object) -> None:
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and getattr(data, "users", 0) == 0:
        if isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)


def _clear_existing_job_lines(col: bpy.types.Collection) -> None:
    for obj in list(col.objects):
        if obj.name == "LINE_TEMPLATE":
            continue
        if not obj.name.startswith("LINE_"):
            continue
        _delete_object_and_data(obj)


def _upsert_curve_line(
    *,
    col: bpy.types.Collection,
    name: str,
    start: Vector,
    mid: Vector,
    end: Vector,
    width: float,
) -> bpy.types.Object:
    existing = bpy.data.objects.get(name)
    if existing is not None and isinstance(existing.data, bpy.types.Curve):
        obj = existing
        curve_data = obj.data
    else:
        curve_data = bpy.data.curves.new(name=f"CURVE_{name}", type="CURVE")
        curve_data.dimensions = "3D"
        obj = bpy.data.objects.new(name, curve_data)
        col.objects.link(obj)

    curve_data.resolution_u = 18
    curve_data.bevel_depth = float(width)
    curve_data.bevel_resolution = 5
    while curve_data.splines:
        curve_data.splines.remove(curve_data.splines[0])
    spline = curve_data.splines.new(type="POLY")
    spline.points.add(2)
    spline.points[0].co = (start.x, start.y, start.z, 1.0)
    spline.points[1].co = (mid.x, mid.y, mid.z, 1.0)
    spline.points[2].co = (end.x, end.y, end.z, 1.0)
    return obj


def _apply_coherence_lines(spec: dict[str, Any], *, palette_name: str) -> None:
    col = _ensure_collection(COL_LINES)
    _clear_existing_job_lines(col)

    edges = spec.get("coherence_edges")
    if not isinstance(edges, list):
        edges = []

    lookup = _electrode_lookup()
    values: list[float] = []
    parsed_edges: list[tuple[str, str, float]] = []
    for edge in edges:
        if not isinstance(edge, Mapping):
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
    coherence_map_cfg = coherence_map if isinstance(coherence_map, Mapping) else {}
    map_type = str(coherence_map_cfg.get("type") or "magnitude").strip().lower()
    vmin = float(coherence_map_cfg.get("min", min(values)))
    vmax = float(coherence_map_cfg.get("max", max(values)))
    mean = statistics.fmean(values) if values else 0.0
    std = statistics.pstdev(values) if len(values) > 1 else 1.0
    clip = float(coherence_map_cfg.get("clip", 2.5))

    for a, b, value in sorted(parsed_edges, key=lambda t: (t[0], t[1])):
        start = lookup[a].matrix_world.translation.copy()
        end = lookup[b].matrix_world.translation.copy()
        midpoint = (start + end) * 0.5
        dist = (start - end).length
        midpoint.z += 0.44 + 0.14 * min(3.0, dist)

        if map_type == "zscore":
            color, emission = _diverging_color(
                value,
                center=mean,
                scale=std,
                clip=clip,
                palette_name=palette_name,
            )
            width = 0.004 + 0.008 * min(1.0, abs((value - mean) / max(1e-6, std)) / max(0.25, clip))
        else:
            color, emission, width = _coherence_color(
                value,
                minimum=vmin,
                maximum=vmax,
                palette_name=palette_name,
            )

        line_name = _coherence_edge_name(a, b)
        obj = _upsert_curve_line(
            col=col,
            name=line_name,
            start=start,
            mid=midpoint,
            end=end,
            width=width,
        )
        mat = _upsert_emission_material(
            f"MAT_JOB_LINE_{a}_{b}",
            color=color,
            emission=emission,
            roughness=0.16,
        )
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def _cleanup_after_job() -> None:
    lines = bpy.data.collections.get(COL_LINES)
    if lines is not None:
        for obj in list(lines.objects):
            if obj.name == "LINE_TEMPLATE":
                continue
            if not obj.name.startswith("LINE_"):
                continue
            _delete_object_and_data(obj)
    for mat in list(bpy.data.materials):
        if not mat.name.startswith("MAT_JOB_"):
            continue
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def _set_linear_interpolation(obj: bpy.types.ID) -> None:
    anim = getattr(obj, "animation_data", None)
    action = getattr(anim, "action", None) if anim else None
    if action is None:
        return
    fcurves = getattr(action, "fcurves", None)
    if not fcurves:
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
    if not isinstance(anim_cfg, Mapping) or not bool(anim_cfg.get("enabled", False)):
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

    cam = bpy.data.objects.get(OBJ_CAM)
    target = bpy.data.objects.get(OBJ_CAM_TARGET)
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
    for col_name in (COL_ELECTRODES, COL_LINES, COL_TEXT):
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


def _scene_id(spec: Mapping[str, Any], out_path: Path) -> int:
    raw = spec.get("scene_id")
    try:
        return int(raw)
    except Exception:
        pass
    stem = out_path.stem
    if stem.startswith("scene_"):
        try:
            return int(stem.split("_")[-1])
        except Exception:
            pass
    return 0


def _render_checkpoint(scene: bpy.types.Scene, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_still_output(scene, out_path)
    bpy.context.view_layer.update()
    scene.frame_set(1)
    bpy.ops.render.render(write_still=True)


def _material_has_principled_emission(mat: bpy.types.Material) -> bool:
    if not mat.use_nodes or mat.node_tree is None:
        return False
    for node in mat.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        has_em_color = node.inputs.get("Emission Color") is not None
        has_em_strength = node.inputs.get("Emission Strength") is not None
        if has_em_color and has_em_strength:
            return True
    return False


def _assert_template_contract(scene: bpy.types.Scene) -> None:
    for col_name in REQUIRED_COLLECTIONS:
        if bpy.data.collections.get(col_name) is None:
            raise RuntimeError(f"Template contract failure: missing collection '{col_name}'")

    for obj_name in REQUIRED_OBJECTS:
        if bpy.data.objects.get(obj_name) is None:
            raise RuntimeError(f"Template contract failure: missing object '{obj_name}'")

    anchors = [obj for obj in bpy.data.objects if obj.name.startswith("ANCH_E_")]
    electrodes = [obj for obj in bpy.data.objects if obj.name.startswith("E_")]
    labels = [obj for obj in bpy.data.objects if obj.name.startswith("LBL_")]
    if len(anchors) < 19:
        raise RuntimeError(
            f"Template contract failure: expected >=19 electrode anchors, found {len(anchors)}"
        )
    if len(electrodes) < 19:
        raise RuntimeError(
            f"Template contract failure: expected >=19 electrode objects, found {len(electrodes)}"
        )
    if len(labels) < 19:
        raise RuntimeError(
            f"Template contract failure: expected >=19 electrode labels, found {len(labels)}"
        )

    for mat_name in REQUIRED_MATERIALS:
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            raise RuntimeError(f"Template contract failure: missing material '{mat_name}'")
        if mat_name == MAT_FOG:
            continue
        if not _material_has_principled_emission(mat):
            raise RuntimeError(
                f"Template contract failure: material '{mat_name}' missing principled emission channels"
            )

    template_version = scene.get("qeeg_template_version")
    if not isinstance(template_version, str) or not template_version.strip():
        raise RuntimeError("Template contract failure: missing scene['qeeg_template_version']")


def _render_job(
    scene: bpy.types.Scene,
    *,
    spec_path: Path,
    out_path: Path,
    video_out_path: Path | None,
    checkpoint_dir: Path | None,
) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise RuntimeError(f"Invalid scene spec JSON (expected object): {spec_path}")

    _assert_template_contract(scene)
    style = _apply_style(scene, spec)
    _apply_text(spec)

    sid = _scene_id(spec, out_path)
    if checkpoint_dir is not None:
        _render_checkpoint(scene, checkpoint_dir / f"scene_{sid:03d}_01_style_text.png")

    _apply_electrodes(spec, palette_name=style["palette"])
    _apply_coherence_lines(spec, palette_name=style["palette"])
    if checkpoint_dir is not None:
        _render_checkpoint(scene, checkpoint_dir / f"scene_{sid:03d}_02_data_bound.png")

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
                    + (details[-1200:] if details else "No ffmpeg diagnostics available.")
                )

    _cleanup_after_job()


def main() -> int:
    args = _parse_args()
    template_path = Path(args.template).expanduser().resolve()
    batch_path = Path(args.batch).expanduser().resolve()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if str(args.checkpoint_dir).strip()
        else None
    )

    if not template_path.exists():
        raise RuntimeError(f"Template .blend not found: {template_path}")
    if not batch_path.exists():
        raise RuntimeError(f"Batch job JSON not found: {batch_path}")

    batch_payload = json.loads(batch_path.read_text(encoding="utf-8"))
    jobs = batch_payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"Batch JSON has no jobs: {batch_path}")

    for idx, job in enumerate(jobs, start=1):
        if not isinstance(job, Mapping):
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
        _ensure_collection(COL_LINES)

        print(
            f"[render_batch] Rendering {idx}/{len(jobs)} spec={spec_path} out={out_path}"
            + (f" video={video_out_path}" if video_out_path else "")
            + (f" checkpoints={checkpoint_dir}" if checkpoint_dir is not None else "")
        )
        _render_job(
            scene,
            spec_path=spec_path,
            out_path=out_path,
            video_out_path=video_out_path,
            checkpoint_dir=checkpoint_dir,
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
