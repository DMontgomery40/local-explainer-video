#!/usr/bin/env python3
"""Programmatically build the deterministic qEEG Blender template."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Build qEEG template .blend")
    parser.add_argument("--output", required=True, help="Output .blend path")
    parser.add_argument("--montage", required=True, help="Montage JSON path")
    parser.add_argument("--font", required=True, help="Bundled font file path")
    parser.add_argument("--width", type=int, default=1664)
    parser.add_argument("--height", type=int, default=928)
    parser.add_argument("--samples", type=int, default=64)
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
    scene.cycles.max_bounces = 6
    scene.cycles.transparent_max_bounces = 6
    scene.cycles.device = "GPU" if gpu else "CPU"
    scene.frame_start = 1
    scene.frame_end = 1

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
            # Determinism beats optimization. If GPU setup fails, continue on CPU.
            scene.cycles.device = "CPU"


def _new_material(name: str, base_color: tuple[float, float, float, float], *, emission: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Specular IOR Level"].default_value = 0.4
    if emission > 0.0:
        bsdf.inputs["Emission Color"].default_value = base_color
        bsdf.inputs["Emission Strength"].default_value = emission
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_fog_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumeScatter")
    vol.inputs["Color"].default_value = (0.25, 0.45, 0.80, 1.0)
    vol.inputs["Density"].default_value = 0.03
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    return mat


def _clear_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "qEEG_Template"
    return scene


def _collection(name: str, *, parent: bpy.types.Collection) -> bpy.types.Collection:
    col = bpy.data.collections.new(name)
    parent.children.link(col)
    return col


def _link_to_collection(obj: bpy.types.Object, col: bpy.types.Collection, *, unlink_from_scene_root: bool = True) -> None:
    if obj.name not in col.objects:
        col.objects.link(obj)
    if unlink_from_scene_root:
        root = bpy.context.scene.collection
        if obj.name in root.objects:
            root.objects.unlink(obj)


def _setup_environment(scene: bpy.types.Scene) -> None:
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    assert nt is not None
    bg = nt.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.008, 0.014, 0.028, 1.0)
        bg.inputs["Strength"].default_value = 0.9


def _create_head_brain_and_fog(*, world_col: bpy.types.Collection) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=2.20, segments=96, ring_count=72, location=(0.0, 0.0, 0.0))
    head = bpy.context.active_object
    assert head is not None
    head.name = "MESH_HEAD"
    head.scale = (1.02, 1.12, 1.0)
    head_mat = _new_material("MAT_HEAD", (0.09, 0.14, 0.23, 1.0), emission=0.05)
    head.data.materials.append(head_mat)
    _link_to_collection(head, world_col)

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1.62, location=(0.0, 0.0, 0.08))
    brain = bpy.context.active_object
    assert brain is not None
    brain.name = "MESH_BRAIN"
    tex = bpy.data.textures.new("TEX_BRAIN", type="CLOUDS")
    tex.noise_scale = 0.38
    disp = brain.modifiers.new("DISP_BRAIN", type="DISPLACE")
    disp.texture = tex
    disp.strength = 0.08
    brain_mat = _new_material("MAT_BRAIN", (0.14, 0.28, 0.56, 1.0), emission=0.12)
    brain.data.materials.append(brain_mat)
    _link_to_collection(brain, world_col)

    bpy.ops.mesh.primitive_cube_add(size=11.0, location=(0.0, 0.0, 0.0))
    fog = bpy.context.active_object
    assert fog is not None
    fog.name = "VOL_FOG"
    fog_mat = _make_fog_material("MAT_FOG")
    fog.data.materials.append(fog_mat)
    _link_to_collection(fog, world_col)


def _create_lighting_and_camera(scene: bpy.types.Scene, *, world_col: bpy.types.Collection) -> None:
    bpy.ops.object.light_add(type="AREA", location=(3.8, -3.8, 4.0))
    key = bpy.context.active_object
    assert key is not None
    key.name = "LIGHT_KEY"
    key.data.energy = 900
    key.data.color = (0.78, 0.86, 1.0)
    key.scale = (1.9, 1.9, 1.9)
    _link_to_collection(key, world_col)

    bpy.ops.object.light_add(type="AREA", location=(-4.3, -2.3, 2.2))
    fill = bpy.context.active_object
    assert fill is not None
    fill.name = "LIGHT_FILL"
    fill.data.energy = 420
    fill.data.color = (0.38, 0.50, 0.95)
    fill.scale = (2.6, 2.6, 2.6)
    _link_to_collection(fill, world_col)

    bpy.ops.object.light_add(type="POINT", location=(0.0, 4.4, 3.5))
    rim = bpy.context.active_object
    assert rim is not None
    rim.name = "LIGHT_RIM"
    rim.data.energy = 620
    rim.data.color = (0.96, 0.45, 0.24)
    _link_to_collection(rim, world_col)

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.4))
    target = bpy.context.active_object
    assert target is not None
    target.name = "CAM_TARGET"
    _link_to_collection(target, world_col)

    bpy.ops.object.camera_add(location=(0.0, -7.4, 1.3))
    cam = bpy.context.active_object
    assert cam is not None
    cam.name = "CAM_MAIN"
    cam.data.lens = 50
    cam.data.clip_start = 0.1
    cam.data.clip_end = 100.0
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    _link_to_collection(cam, world_col)
    scene.camera = cam


def _create_electrodes(*, electrode_col: bpy.types.Collection, montage_path: Path) -> None:
    payload = json.loads(montage_path.read_text(encoding="utf-8"))
    channels = payload.get("channels")
    if not isinstance(channels, list) or not channels:
        raise RuntimeError(f"Invalid montage JSON (missing channels): {montage_path}")

    mat = _new_material("MAT_ELECTRODE_BASE", (0.92, 0.96, 1.0, 1.0), emission=0.35)
    radius = 2.28
    for item in channels:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        vec = Vector((
            float(item.get("x", 0.0)),
            float(item.get("y", 0.0)),
            float(item.get("z", 1.0)),
        ))
        if vec.length == 0:
            vec = Vector((0.0, 0.0, 1.0))
        pos = vec.normalized() * radius
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, segments=24, ring_count=16, location=tuple(pos))
        obj = bpy.context.active_object
        assert obj is not None
        obj.name = f"E_{label}"
        obj.data.materials.append(mat.copy())
        _link_to_collection(obj, electrode_col)


def _create_text_objects(*, text_col: bpy.types.Collection, font_path: Path) -> None:
    if not font_path.exists():
        raise RuntimeError(f"Bundled font path does not exist: {font_path}")
    font = bpy.data.fonts.load(str(font_path))

    text_mat = _new_material("MAT_TEXT", (0.94, 0.96, 1.0, 1.0), emission=0.8)
    slots = [
        ("TXT_TITLE", (0.0, -1.95, 2.72), 0.34, "qEEG Overview"),
        ("TXT_SUBTITLE", (0.0, -1.95, 2.26), 0.20, "Band activity and coherence"),
        ("TXT_FOOTER", (0.0, -1.95, -2.62), 0.16, "Deterministic clinical-data render"),
    ]
    for name, loc, size, body in slots:
        bpy.ops.object.text_add(location=loc, rotation=(math.radians(90), 0.0, 0.0))
        obj = bpy.context.active_object
        assert obj is not None
        obj.name = name
        obj.data.body = body
        obj.data.align_x = "CENTER"
        obj.data.align_y = "CENTER"
        obj.data.size = size
        obj.data.extrude = 0.01
        obj.data.font = font
        obj.data.materials.append(text_mat.copy())
        _link_to_collection(obj, text_col)


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    montage_path = Path(args.montage).expanduser().resolve()
    font_path = Path(args.font).expanduser().resolve()

    if not montage_path.exists():
        raise RuntimeError(f"Montage JSON missing: {montage_path}")
    if not font_path.exists():
        raise RuntimeError(f"Bundled font missing: {font_path}")

    scene = _clear_scene()
    _configure_render(scene, width=args.width, height=args.height, samples=args.samples, gpu=bool(args.gpu))
    _setup_environment(scene)

    world_col = _collection("COL_WORLD", parent=scene.collection)
    electrode_col = _collection("COL_ELECTRODES", parent=scene.collection)
    _collection("COL_LINES", parent=scene.collection)
    text_col = _collection("COL_TEXT", parent=scene.collection)

    _create_head_brain_and_fog(world_col=world_col)
    _create_lighting_and_camera(scene, world_col=world_col)
    _create_electrodes(electrode_col=electrode_col, montage_path=montage_path)
    _create_text_objects(text_col=text_col, font_path=font_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), compress=False)
    print(f"[build_template] Saved: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[build_template] ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
