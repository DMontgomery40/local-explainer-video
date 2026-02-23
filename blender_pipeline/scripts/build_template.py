#!/usr/bin/env python3
"""Build the canonical deterministic qEEG brain base model template (.blend)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Vector


TEMPLATE_VERSION = "brain_basemodel_v2"

# MRI-derived brain mesh assets (brainder.org, CC-BY-SA 3.0)
BRAIN_MESH_DIR = Path(__file__).resolve().parent.parent / "assets" / "brain_mesh"
BRAIN_MESH_SCALE = 0.0223  # MNI mm -> Blender units (~45mm per BU)

COL_WORLD = "COL_WORLD"
COL_ELECTRODES = "COL_ELECTRODES"
COL_ELECTRODE_ANCHORS = "COL_ELECTRODE_ANCHORS"
COL_LINES = "COL_LINES"
COL_TEXT = "COL_TEXT"
COL_GUIDES = "COL_GUIDES"

OBJ_BRAIN = "MESH_BRAIN"
OBJ_HEAD = "MESH_HEAD_SHELL"
OBJ_FOG = "VOL_ATMOSPHERE"
OBJ_CAM = "CAM_MAIN"
OBJ_CAM_TARGET = "CAM_TARGET"
OBJ_SAFE_TITLE = "SAFE_TITLE"
OBJ_SAFE_SUBTITLE = "SAFE_SUBTITLE"
OBJ_SAFE_FOOTER = "SAFE_FOOTER"

MAT_BRAIN = "MAT_BRAIN_BASE"
MAT_HEAD = "MAT_HEAD_SHELL"
MAT_FOG = "MAT_FOG_VOLUME"
MAT_TEXT = "MAT_TEXT_BASE"
MAT_ELECTRODE = "MAT_ELECTRODE_BASE"
MAT_LABEL = "MAT_LABEL_BASE"
MAT_LINE = "MAT_LINE_BASE"


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
    parser.add_argument("--mesh-dir", default=str(BRAIN_MESH_DIR), help="Path to brain mesh OBJ directory")
    return parser.parse_args(argv)


def _clear_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "qEEG_Template"
    return scene


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
    scene.cycles.max_bounces = 6
    _set_numeric_if_present(scene.cycles, "diffuse_bounces", 4)
    _set_numeric_if_present(scene.cycles, "glossy_bounces", 4)
    _set_numeric_if_present(scene.cycles, "transmission_bounces", 6)
    scene.cycles.transparent_max_bounces = 6
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
    scene.frame_start = 1
    scene.frame_end = 1

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


def _set_principled_value(node: bpy.types.Node, socket_name: str, value: object) -> None:
    sock = node.inputs.get(socket_name)
    if sock is not None:
        sock.default_value = value


def _new_principled_material(
    name: str,
    *,
    base_color: tuple[float, float, float, float],
    roughness: float,
    emission_color: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
    alpha: float = 1.0,
    transmission: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_fake_user = True
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_principled_value(bsdf, "Base Color", base_color)
    _set_principled_value(bsdf, "Roughness", float(roughness))
    _set_principled_value(bsdf, "Alpha", float(alpha))
    _set_principled_value(bsdf, "Transmission Weight", float(transmission))
    _set_principled_value(bsdf, "Transmission", float(transmission))
    _set_principled_value(bsdf, "Specular IOR Level", 0.35)
    _set_principled_value(bsdf, "Emission Color", emission_color or base_color)
    _set_principled_value(bsdf, "Emission Strength", float(emission_strength))
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    return mat


def _new_fog_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_fake_user = True
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumeScatter")
    vol.inputs["Color"].default_value = (0.20, 0.40, 0.72, 1.0)
    vol.inputs["Density"].default_value = 0.022
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    return mat


def _setup_environment(scene: bpy.types.Scene) -> None:
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    assert nt is not None
    bg = nt.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.006, 0.012, 0.028, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def _shade_smooth(obj: bpy.types.Object) -> None:
    if obj.type != "MESH":
        return
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
    finally:
        obj.select_set(False)



def _new_brain_material(
    name: str,
    *,
    base_color: tuple[float, float, float, float],
    roughness: float,
    emission_color: tuple[float, float, float, float],
    emission_strength: float,
) -> bpy.types.Material:
    """Create brain material with SSS and procedural bump for organic quality."""
    mat = bpy.data.materials.new(name=name)
    mat.use_fake_user = True
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    for node in list(nt.nodes):
        nt.nodes.remove(node)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (500, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)

    _set_principled_value(bsdf, "Base Color", base_color)
    _set_principled_value(bsdf, "Roughness", float(roughness))
    _set_principled_value(bsdf, "Emission Color", emission_color)
    _set_principled_value(bsdf, "Emission Strength", float(emission_strength))
    _set_principled_value(bsdf, "Specular IOR Level", 0.35)

    # Subsurface scattering for organic translucency (try both API naming eras)
    _set_principled_value(bsdf, "Subsurface Weight", 0.10)
    _set_principled_value(bsdf, "Subsurface", 0.10)
    _set_principled_value(bsdf, "Subsurface Radius", (0.4, 0.15, 0.10))
    _set_principled_value(bsdf, "Subsurface Scale", 0.08)

    # Procedural bump for fine cortical micro-texture (shader-level detail)
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-700, -300)

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-400, -300)
    noise.inputs["Scale"].default_value = 28.0
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.65

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-100, -300)
    bump.inputs["Strength"].default_value = 0.05
    sock_dist = bump.inputs.get("Distance")
    if sock_dist is not None:
        sock_dist.default_value = 0.008

    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat


def _recenter_mesh_to_origin(obj: bpy.types.Object) -> None:
    """Shift all vertices so the bounding-box center sits at the object origin."""
    mesh = obj.data
    bb = [Vector(v) for v in obj.bound_box]
    center = sum(bb, Vector()) / len(bb)
    for vert in mesh.vertices:
        vert.co -= center
    mesh.update()


def _import_brain_mesh(
    mesh_dir: Path,
    *,
    name: str = OBJ_BRAIN,
    scale: float = BRAIN_MESH_SCALE,
    location: tuple[float, float, float] = (0.0, 0.0, 0.08),
    decimate_ratio: float = 0.35,
    smooth_iterations: int = 8,
    smooth_factor: float = 0.5,
) -> bpy.types.Object:
    """Import brainder.org pial OBJ hemispheres into a single brain mesh.

    Pipeline: import -> join -> reconsistent normals -> recenter -> scale
    -> decimate -> smooth -> subdivision surface -> shade smooth.
    """
    lh_path = mesh_dir / "lh.pial.obj"
    rh_path = mesh_dir / "rh.pial.obj"
    if not lh_path.exists() or not rh_path.exists():
        raise RuntimeError(
            f"Brain mesh OBJ files not found in {mesh_dir}. "
            f"Expected lh.pial.obj and rh.pial.obj"
        )

    imported_objects: list[bpy.types.Object] = []
    for obj_path in (lh_path, rh_path):
        before = set(bpy.data.objects)
        # Try Blender 4.x+ OBJ importer first, fall back to legacy
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_path))
        except (AttributeError, RuntimeError):
            bpy.ops.import_scene.obj(filepath=str(obj_path))
        after = set(bpy.data.objects)
        new_objs = after - before
        imported_objects.extend(o for o in new_objs if o.type == "MESH")

    if not imported_objects:
        raise RuntimeError("OBJ import produced no mesh objects")

    # Select all imported objects and join into one
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_objects[0]
    if len(imported_objects) > 1:
        bpy.ops.object.join()

    brain = bpy.context.active_object
    assert brain is not None
    brain.name = name

    # Reconsistent normals (edit mode)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Recenter to origin
    _recenter_mesh_to_origin(brain)

    # Apply uniform scale from MNI mm to Blender units
    brain.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(scale=True)

    # Set final position
    brain.location = location

    # Decimate to reduce poly count (~580K -> ~200K faces)
    dec = brain.modifiers.new("DECIMATE_BRAIN", type="DECIMATE")
    dec.ratio = decimate_ratio

    # Smooth for medical-illustration polish
    sm = brain.modifiers.new("SMOOTH_BRAIN", type="SMOOTH")
    sm.iterations = smooth_iterations
    sm.factor = smooth_factor

    # Subdivision surface (viewport=0 for performance, render=1 for quality)
    ss = brain.modifiers.new("SUBSURF_BRAIN", type="SUBSURF")
    ss.levels = 0
    ss.render_levels = 1

    _shade_smooth(brain)
    return brain


def _brain_surface_point(
    brain_obj: bpy.types.Object,
    direction: Vector,
    *,
    ray_origin_distance: float = 10.0,
    fallback_radius: float = 2.0,
) -> tuple[Vector, Vector]:
    """Find where a ray from outside hits the brain surface.

    Returns (world_point, world_normal). Uses ray_cast with closest_point_on_mesh
    fallback, and finally a sphere fallback if both miss.
    """
    d = direction.normalized()
    origin = d * ray_origin_distance
    ray_dir = -d

    # Use evaluated mesh (with modifiers applied)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    brain_eval = brain_obj.evaluated_get(depsgraph)

    # Transform ray into object-local space
    mat_inv = brain_obj.matrix_world.inverted()
    local_origin = mat_inv @ origin
    local_dir = (mat_inv @ (origin + ray_dir) - local_origin).normalized()

    hit, loc, normal, _ = brain_eval.ray_cast(local_origin, local_dir)
    if hit:
        world_point = brain_obj.matrix_world @ loc
        world_normal = (brain_obj.matrix_world.to_3x3() @ normal).normalized()
        return world_point, world_normal

    # Fallback: closest point on mesh
    found, loc, normal, _ = brain_eval.closest_point_on_mesh(local_origin)
    if found:
        world_point = brain_obj.matrix_world @ loc
        world_normal = (brain_obj.matrix_world.to_3x3() @ normal).normalized()
        return world_point, world_normal

    # Final fallback: sphere at fallback_radius
    point = d * fallback_radius + brain_obj.location
    return point, d


def _create_brain_head_and_fog(*, mesh_dir: Path, world_col: bpy.types.Collection) -> bpy.types.Object:
    """Create brain from MRI mesh, transparent head shell, and fog volume.

    Returns the brain object (needed for electrode surface raycasting).
    """
    # Head shell — barely visible silhouette, provides atmospheric framing
    bpy.ops.mesh.primitive_uv_sphere_add(radius=2.20, segments=128, ring_count=96, location=(0.0, 0.0, 0.0))
    head = bpy.context.active_object
    assert head is not None
    head.name = OBJ_HEAD
    head.scale = (1.02, 1.10, 0.98)
    _shade_smooth(head)
    mat_head = _new_principled_material(
        MAT_HEAD,
        base_color=(0.05, 0.10, 0.18, 1.0),
        roughness=0.14,
        emission_color=(0.07, 0.20, 0.34, 1.0),
        emission_strength=0.06,
        alpha=0.08,
        transmission=0.22,
    )
    head.data.materials.append(mat_head)
    _link_to_collection(head, world_col)

    # Brain — real MRI pial surface (replaces UV sphere + displacement)
    brain = _import_brain_mesh(mesh_dir)
    mat_brain = _new_brain_material(
        MAT_BRAIN,
        base_color=(0.12, 0.26, 0.52, 1.0),
        roughness=0.24,
        emission_color=(0.16, 0.40, 0.80, 1.0),
        emission_strength=0.12,
    )
    brain.data.materials.append(mat_brain)
    _link_to_collection(brain, world_col)

    # Atmospheric fog volume
    bpy.ops.mesh.primitive_cube_add(size=11.5, location=(0.0, 0.0, 0.0))
    fog = bpy.context.active_object
    assert fog is not None
    fog.name = OBJ_FOG
    fog_mat = _new_fog_material(MAT_FOG)
    fog.data.materials.append(fog_mat)
    _link_to_collection(fog, world_col)

    return brain


def _create_lighting_and_camera(scene: bpy.types.Scene, *, world_col: bpy.types.Collection) -> None:
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.24))
    target = bpy.context.active_object
    assert target is not None
    target.name = OBJ_CAM_TARGET
    _link_to_collection(target, world_col)

    bpy.ops.object.camera_add(location=(-5.4, -9.6, 4.2))
    cam = bpy.context.active_object
    assert cam is not None
    cam.name = OBJ_CAM
    cam.data.lens = 42
    cam.data.clip_start = 0.1
    cam.data.clip_end = 100.0
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    _link_to_collection(cam, world_col)
    scene.camera = cam

    bpy.ops.object.light_add(type="AREA", location=(4.0, -4.4, 4.6))
    key = bpy.context.active_object
    assert key is not None
    key.name = "LIGHT_KEY"
    key.data.energy = 980
    key.data.color = (0.76, 0.86, 1.0)
    key.scale = (2.2, 2.2, 2.2)
    _link_to_collection(key, world_col)

    bpy.ops.object.light_add(type="AREA", location=(-4.8, -2.1, 2.4))
    fill = bpy.context.active_object
    assert fill is not None
    fill.name = "LIGHT_FILL"
    fill.data.energy = 430
    fill.data.color = (0.42, 0.54, 0.92)
    fill.scale = (2.8, 2.8, 2.8)
    _link_to_collection(fill, world_col)

    bpy.ops.object.light_add(type="POINT", location=(0.0, 4.5, 3.8))
    rim = bpy.context.active_object
    assert rim is not None
    rim.name = "LIGHT_RIM"
    rim.data.energy = 640
    rim.data.color = (0.98, 0.55, 0.28)
    _link_to_collection(rim, world_col)

    bpy.ops.object.light_add(type="SPOT", location=(0.0, -5.6, 5.8))
    overhead = bpy.context.active_object
    assert overhead is not None
    overhead.name = "LIGHT_OVERHEAD"
    overhead.data.energy = 280
    overhead.data.color = (0.60, 0.74, 1.0)
    overhead.data.spot_size = math.radians(70.0)
    overhead.data.spot_blend = 0.52
    _link_to_collection(overhead, world_col)


def _create_safe_regions(*, guides_col: bpy.types.Collection) -> None:
    slots = [
        (OBJ_SAFE_TITLE, (0.0, -2.75, 2.95)),
        (OBJ_SAFE_SUBTITLE, (0.0, -2.75, 2.42)),
        (OBJ_SAFE_FOOTER, (0.0, -2.75, -2.70)),
    ]
    for name, loc in slots:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=loc)
        obj = bpy.context.active_object
        assert obj is not None
        obj.name = name
        obj.empty_display_size = 0.12
        _link_to_collection(obj, guides_col)


def _create_text_objects(*, text_col: bpy.types.Collection, font_path: Path) -> None:
    if not font_path.exists():
        raise RuntimeError(f"Bundled font path does not exist: {font_path}")
    font = bpy.data.fonts.load(str(font_path))

    text_mat = _new_principled_material(
        MAT_TEXT,
        base_color=(0.92, 0.96, 1.0, 1.0),
        roughness=0.15,
        emission_color=(0.92, 0.96, 1.0, 1.0),
        emission_strength=0.92,
    )
    slots = [
        ("TXT_TITLE", OBJ_SAFE_TITLE, 0.32, "qEEG Clinical Overview"),
        ("TXT_SUBTITLE", OBJ_SAFE_SUBTITLE, 0.20, "Band activity and coherence"),
        ("TXT_FOOTER", OBJ_SAFE_FOOTER, 0.15, "Deterministic patient-data render"),
    ]
    for name, anchor_name, size, body in slots:
        anchor = bpy.data.objects.get(anchor_name)
        if anchor is None:
            raise RuntimeError(f"Missing safe-region anchor: {anchor_name}")
        bpy.ops.object.text_add(location=tuple(anchor.location), rotation=(0.0, 0.0, 0.0))
        obj = bpy.context.active_object
        assert obj is not None
        obj.name = name
        obj.data.body = body
        obj.data.align_x = "CENTER"
        obj.data.align_y = "CENTER"
        obj.data.size = size
        obj.data.extrude = 0.010
        obj.data.font = font
        obj.data.materials.append(text_mat.copy())
        cam = bpy.data.objects.get(OBJ_CAM)
        if cam is not None:
            copy_rot = obj.constraints.new("COPY_ROTATION")
            copy_rot.target = cam
        _link_to_collection(obj, text_col)


def _load_montage_channels(montage_path: Path) -> list[dict[str, object]]:
    payload = json.loads(montage_path.read_text(encoding="utf-8"))
    channels = payload.get("channels")
    if not isinstance(channels, list) or not channels:
        raise RuntimeError(f"Invalid montage JSON (missing channels): {montage_path}")

    out: list[dict[str, object]] = []
    for item in channels:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        out.append(
            {
                "label": label,
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "z": float(item.get("z", 1.0)),
            }
        )
    if not out:
        raise RuntimeError(f"Montage JSON had no valid channel entries: {montage_path}")
    return out


def _create_electrodes_and_anchors(
    *,
    brain_obj: bpy.types.Object,
    anchor_col: bpy.types.Collection,
    electrode_col: bpy.types.Collection,
    text_col: bpy.types.Collection,
    font_path: Path,
    montage_path: Path,
) -> list[str]:
    channels = _load_montage_channels(montage_path)
    font = bpy.data.fonts.load(str(font_path))

    electrode_mat = _new_principled_material(
        MAT_ELECTRODE,
        base_color=(0.90, 0.95, 1.0, 1.0),
        roughness=0.18,
        emission_color=(0.56, 0.92, 1.0, 1.0),
        emission_strength=0.70,
    )
    label_mat = _new_principled_material(
        MAT_LABEL,
        base_color=(0.98, 0.99, 1.0, 1.0),
        roughness=0.09,
        emission_color=(0.90, 0.97, 1.0, 1.0),
        emission_strength=0.80,
    )

    cam = bpy.data.objects.get(OBJ_CAM)
    labels: list[str] = []
    for item in channels:
        label = str(item["label"])
        vec = Vector((float(item["x"]), float(item["y"]), float(item["z"])))
        if vec.length == 0:
            vec = Vector((0.0, 0.0, 1.0))

        # Raycast to brain surface for anatomically-correct placement
        surface_point, surface_normal = _brain_surface_point(brain_obj, vec)
        pos = surface_point + surface_normal * 0.06
        label_pos = surface_point + surface_normal * 0.22

        bpy.ops.object.empty_add(type="PLAIN_AXES", location=tuple(pos))
        anch = bpy.context.active_object
        assert anch is not None
        anch.name = f"ANCH_E_{label}"
        anch.empty_display_size = 0.055
        _link_to_collection(anch, anchor_col)

        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.075, segments=24, ring_count=16, location=tuple(pos))
        elec = bpy.context.active_object
        assert elec is not None
        elec.name = f"E_{label}"
        elec.data.materials.append(electrode_mat.copy())
        elec.parent = anch
        _shade_smooth(elec)
        _link_to_collection(elec, electrode_col)

        bpy.ops.object.text_add(location=tuple(label_pos), rotation=(0.0, 0.0, 0.0))
        txt = bpy.context.active_object
        assert txt is not None
        txt.name = f"LBL_{label}"
        txt.data.body = label
        txt.data.align_x = "CENTER"
        txt.data.align_y = "CENTER"
        txt.data.size = 0.135
        txt.data.extrude = 0.004
        txt.data.font = font
        txt.data.materials.append(label_mat.copy())
        if cam is not None:
            copy_rot = txt.constraints.new("COPY_ROTATION")
            copy_rot.target = cam
        _link_to_collection(txt, text_col)
        labels.append(label)

    return labels


def _create_line_system(*, line_col: bpy.types.Collection) -> None:
    curve = bpy.data.curves.new("CURVE_LINE_TEMPLATE", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 16
    curve.bevel_depth = 0.012
    curve.bevel_resolution = 4
    spline = curve.splines.new(type="POLY")
    spline.points.add(2)
    spline.points[0].co = (-0.6, 0.0, 1.6, 1.0)
    spline.points[1].co = (0.0, 0.0, 1.9, 1.0)
    spline.points[2].co = (0.6, 0.0, 1.6, 1.0)

    obj = bpy.data.objects.new("LINE_TEMPLATE", curve)
    line_col.objects.link(obj)
    mat = _new_principled_material(
        MAT_LINE,
        base_color=(0.36, 0.82, 1.0, 1.0),
        roughness=0.16,
        emission_color=(0.38, 0.90, 1.0, 1.0),
        emission_strength=1.10,
    )
    obj.data.materials.append(mat)
    obj.hide_viewport = True
    obj.hide_render = True


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    montage_path = Path(args.montage).expanduser().resolve()
    font_path = Path(args.font).expanduser().resolve()
    mesh_dir = Path(args.mesh_dir).expanduser().resolve()

    if not montage_path.exists():
        raise RuntimeError(f"Montage JSON missing: {montage_path}")
    if not font_path.exists():
        raise RuntimeError(f"Bundled font missing: {font_path}")
    if not (mesh_dir / "lh.pial.obj").exists() or not (mesh_dir / "rh.pial.obj").exists():
        raise RuntimeError(f"Brain mesh files missing in: {mesh_dir}")

    scene = _clear_scene()
    _configure_render(scene, width=args.width, height=args.height, samples=args.samples, gpu=bool(args.gpu))
    _setup_environment(scene)

    world_col = _collection(COL_WORLD, parent=scene.collection)
    electrode_col = _collection(COL_ELECTRODES, parent=scene.collection)
    anchor_col = _collection(COL_ELECTRODE_ANCHORS, parent=scene.collection)
    line_col = _collection(COL_LINES, parent=scene.collection)
    text_col = _collection(COL_TEXT, parent=scene.collection)
    guides_col = _collection(COL_GUIDES, parent=scene.collection)

    brain_obj = _create_brain_head_and_fog(mesh_dir=mesh_dir, world_col=world_col)
    _create_lighting_and_camera(scene, world_col=world_col)
    _create_safe_regions(guides_col=guides_col)
    _create_text_objects(text_col=text_col, font_path=font_path)
    channel_labels = _create_electrodes_and_anchors(
        brain_obj=brain_obj,
        anchor_col=anchor_col,
        electrode_col=electrode_col,
        text_col=text_col,
        font_path=font_path,
        montage_path=montage_path,
    )
    _create_line_system(line_col=line_col)

    scene["qeeg_template_version"] = TEMPLATE_VERSION
    scene["qeeg_channel_labels"] = channel_labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), compress=False)
    print(f"[build_template] Saved: {output_path}")
    print(f"[build_template] Version: {TEMPLATE_VERSION}")
    print(f"[build_template] Electrodes: {len(channel_labels)}")
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
