"""Deterministic Blender backend for qEEG clinical scenes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from core.image_gen import TARGET_HEIGHT, TARGET_WIDTH
from core.qeeg_extract import extract_qeeg_visual_data, normalize_electrode_label


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pipeline_dir() -> Path:
    return _repo_root() / "blender_pipeline"


def _build_script_path() -> Path:
    return _pipeline_dir() / "scripts" / "build_template.py"


def _render_script_path() -> Path:
    return _pipeline_dir() / "scripts" / "render_batch.py"


def _default_template_path() -> Path:
    return _pipeline_dir() / "assets" / "qeeg_template.blend"


def _default_font_path() -> Path:
    return _pipeline_dir() / "assets" / "fonts" / "NotoSans-Regular.ttf"


def _default_montage_path() -> Path:
    return _pipeline_dir() / "assets" / "montage" / "standard_1020.json"


@dataclass(frozen=True)
class BlenderRenderConfig:
    blender_bin: str | None
    template_path: Path
    samples: int = 64
    gpu: bool = False
    cache: bool = True

    @classmethod
    def from_env(cls) -> "BlenderRenderConfig":
        env_bin = str(os.getenv("BLENDER_BIN") or "").strip()
        found = env_bin or shutil.which("blender") or None
        samples = int(os.getenv("BLENDER_SAMPLES", "64"))
        gpu = str(os.getenv("BLENDER_GPU", "")).strip().lower() in {"1", "true", "yes", "on"}
        cache = str(os.getenv("BLENDER_CACHE", "1")).strip().lower() not in {"0", "false", "no", "off"}
        return cls(
            blender_bin=found,
            template_path=_default_template_path(),
            samples=max(1, samples),
            gpu=gpu,
            cache=cache,
        )


def ensure_template_exists(config: BlenderRenderConfig | None = None) -> Path:
    cfg = config or BlenderRenderConfig.from_env()
    template_path = Path(cfg.template_path)
    if template_path.exists():
        return template_path

    blender_bin = _require_blender_bin(cfg)
    build_script = _build_script_path()
    montage_path = _default_montage_path()
    font_path = _default_font_path()

    if not build_script.exists():
        raise FileNotFoundError(f"Missing Blender template builder script: {build_script}")
    if not montage_path.exists():
        raise FileNotFoundError(f"Missing montage coordinates JSON: {montage_path}")
    if not font_path.exists():
        raise FileNotFoundError(f"Missing bundled font file: {font_path}")

    template_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        str(TARGET_WIDTH),
        "--height",
        str(TARGET_HEIGHT),
        "--samples",
        str(int(cfg.samples)),
    ]
    if cfg.gpu:
        cmd.append("--gpu")
    _run_blender_command(cmd, label="build-template")

    if not template_path.exists():
        raise RuntimeError(f"Blender template build completed but output is missing: {template_path}")
    return template_path


def render_blender_scene(
    scene: Mapping[str, Any],
    project_dir: Path,
    *,
    data_pack: Mapping[str, Any] | dict[str, Any] | None = None,
    config: BlenderRenderConfig | None = None,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> Path:
    rendered = render_blender_scenes_batch(
        scenes=[scene],
        project_dir=project_dir,
        data_pack=data_pack,
        config=config,
        force=force,
        log=log,
    )
    scene_id = int(scene.get("id", 0))
    path = rendered.get(scene_id)
    if not path:
        raise RuntimeError(f"Blender renderer did not return an output for scene {scene_id}")
    maybe_video = blender_scene_video_path(Path(project_dir), scene_id)
    if isinstance(scene, dict):
        if maybe_video.exists():
            scene["video_path"] = str(maybe_video)
        elif "video_path" in scene:
            scene.pop("video_path", None)
    return path


def render_blender_scenes_batch(
    scenes: Sequence[Mapping[str, Any]],
    project_dir: Path,
    *,
    data_pack: Mapping[str, Any] | dict[str, Any] | None = None,
    config: BlenderRenderConfig | None = None,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[int, Path]:
    if not scenes:
        return {}

    cfg = config or BlenderRenderConfig.from_env()
    blender_bin = _require_blender_bin(cfg)
    template_path = ensure_template_exists(cfg)

    project_dir = Path(project_dir)
    images_dir = project_dir / "images"
    videos_dir = project_dir / "videos"
    blender_dir = project_dir / "blender"
    specs_dir = blender_dir / "specs"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)

    logger = log or (lambda _: None)

    template_signature = _template_signature(template_path)
    settings_payload = {
        "samples": int(cfg.samples),
        "gpu": bool(cfg.gpu),
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
    }

    cache_path = blender_dir / "cache_index.json"
    cache_index = _load_cache_index(cache_path)
    cache_entries = cache_index.setdefault("entries", {})

    outputs: dict[int, Path] = {}
    jobs: list[dict[str, Any]] = []
    rendered_cache_updates: dict[str, dict[str, Any]] = {}

    for idx, scene in enumerate(scenes):
        scene_id = int(scene.get("id", idx))
        output_path = images_dir / f"scene_{scene_id:03d}.png"
        video_out: Path | None = blender_scene_video_path(project_dir, scene_id)
        spec = _build_scene_spec(scene=scene, data_pack=data_pack)
        spec_path = specs_dir / f"scene_{scene_id:03d}.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        animation_cfg = spec.get("animation")
        animation_enabled = bool(isinstance(animation_cfg, Mapping) and animation_cfg.get("enabled"))

        spec_canonical_json = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        cache_key = _spec_cache_key(spec_json=spec_canonical_json, template_signature=template_signature, settings=settings_payload)
        cache_entry = cache_entries.get(str(scene_id), {})
        expected_outputs = [output_path]
        if animation_enabled:
            expected_outputs.append(video_out)

        if (
            bool(cfg.cache)
            and not force
            and isinstance(cache_entry, Mapping)
            and cache_entry.get("cache_key") == cache_key
            and all(Path(str(p)).exists() for p in expected_outputs)
        ):
            cached_path = Path(str(cache_entry.get("output_path") or output_path))
            outputs[scene_id] = cached_path
            logger(f"Blender cache hit: scene {scene_id} -> {cached_path}")
            continue

        jobs.append(
            {
                "scene_id": scene_id,
                "spec": str(spec_path),
                "out": str(output_path),
                "video_out": str(video_out) if animation_enabled else "",
            }
        )
        rendered_cache_updates[str(scene_id)] = {
            "cache_key": cache_key,
            "output_path": str(output_path),
            "video_out": str(video_out) if animation_enabled else "",
            "spec_path": str(spec_path),
            "rendered_at_utc": _utc_now_iso(),
        }
        outputs[scene_id] = output_path

    if jobs:
        batch_path = blender_dir / "batch_jobs.json"
        batch_payload = {
            "template": str(template_path),
            "jobs": jobs,
            "render_settings": settings_payload,
        }
        batch_path.write_text(json.dumps(batch_payload, indent=2, sort_keys=True), encoding="utf-8")

        cmd = [
            blender_bin,
            "-b",
            "--factory-startup",
            "--python",
            str(_render_script_path()),
            "--",
            "--template",
            str(template_path),
            "--batch",
            str(batch_path),
            "--samples",
            str(int(cfg.samples)),
            "--width",
            str(TARGET_WIDTH),
            "--height",
            str(TARGET_HEIGHT),
        ]
        if cfg.gpu:
            cmd.append("--gpu")
        _run_blender_command(cmd, label=f"render-batch ({len(jobs)} scene(s))")

        for job in jobs:
            out = Path(str(job["out"]))
            if not out.exists():
                raise RuntimeError(f"Blender batch reported success but output is missing: {out}")
            video_out = str(job.get("video_out") or "").strip()
            if video_out:
                video_path = Path(video_out)
                if not video_path.exists():
                    raise RuntimeError(f"Blender batch reported success but animation output is missing: {video_path}")

    if bool(cfg.cache):
        cache_index["template_signature"] = template_signature
        cache_index["render_settings"] = settings_payload
        cache_entries.update(rendered_cache_updates)
        cache_path.write_text(json.dumps(cache_index, indent=2, sort_keys=True), encoding="utf-8")

    return outputs


def _build_scene_spec(
    *,
    scene: Mapping[str, Any],
    data_pack: Mapping[str, Any] | dict[str, Any] | None,
) -> dict[str, Any]:
    scene_id = int(scene.get("id", 0))
    blender_cfg = scene.get("blender")
    blender_cfg_map = blender_cfg if isinstance(blender_cfg, Mapping) else {}

    title = _string_or_empty(blender_cfg_map.get("title") or scene.get("title"))
    subtitle = _string_or_empty(blender_cfg_map.get("subtitle") or scene.get("subtitle"))
    footer = _string_or_empty(blender_cfg_map.get("footer") or scene.get("footer"))

    scene_electrode = _coerce_manual_electrode_values(
        blender_cfg_map.get("electrode_values") or scene.get("electrode_values")
    )
    scene_edges = _coerce_manual_edges(
        blender_cfg_map.get("coherence_edges") or scene.get("coherence_edges")
    )

    extraction_cfg = blender_cfg_map.get("extract")
    electrode_values: dict[str, float] = dict(scene_electrode)
    coherence_edges: list[dict[str, float | str]] = list(scene_edges)

    if isinstance(data_pack, Mapping):
        auto_electrode, auto_edges = extract_qeeg_visual_data(data_pack, scene=scene, config=extraction_cfg)
        if not electrode_values:
            electrode_values = auto_electrode
        if not coherence_edges:
            coherence_edges = auto_edges

    value_map = blender_cfg_map.get("value_map")
    coherence_map = blender_cfg_map.get("coherence_map")
    animation_map = blender_cfg_map.get("animation") or scene.get("animation")
    if not isinstance(value_map, Mapping):
        value_map = {"type": "zscore", "clip": 2.5}
    if not isinstance(coherence_map, Mapping):
        coherence_map = {"type": "magnitude", "min": 0.0, "max": 1.0}
    animation_cfg = _coerce_animation_cfg(animation_map)

    return {
        "scene_id": scene_id,
        "title": title,
        "subtitle": subtitle,
        "footer": footer,
        "electrode_values": {k: electrode_values[k] for k in sorted(electrode_values)},
        "coherence_edges": _sorted_edges(coherence_edges),
        "value_map": dict(value_map),
        "coherence_map": dict(coherence_map),
        "animation": animation_cfg,
    }


def _coerce_manual_electrode_values(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        label = normalize_electrode_label(key)
        if not label:
            continue
        try:
            out[label] = float(value)
        except Exception:
            continue
    return out


def _coerce_manual_edges(raw: Any) -> list[dict[str, float | str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, float | str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        a = normalize_electrode_label(item.get("a"))
        b = normalize_electrode_label(item.get("b"))
        if not a or not b or a == b:
            continue
        try:
            value = float(item.get("value"))
        except Exception:
            continue
        a_sorted, b_sorted = sorted((a, b))
        out.append({"a": a_sorted, "b": b_sorted, "value": value})
    return _sorted_edges(out)


def _sorted_edges(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, float | str]]:
    out: dict[tuple[str, str], float] = {}
    for edge in edges:
        a = normalize_electrode_label(edge.get("a"))
        b = normalize_electrode_label(edge.get("b"))
        if not a or not b or a == b:
            continue
        try:
            value = float(edge.get("value"))
        except Exception:
            continue
        k = tuple(sorted((a, b)))
        current = out.get(k)
        if current is None or abs(value) > abs(current):
            out[k] = value
    return [{"a": a, "b": b, "value": out[(a, b)]} for a, b in sorted(out)]


def _spec_cache_key(*, spec_json: str, template_signature: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    payload = {
        "spec": json.loads(spec_json),
        "template": dict(template_signature),
        "settings": dict(settings),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _template_signature(template_path: Path) -> dict[str, Any]:
    stat = template_path.stat()
    template_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
    return {
        "path": str(template_path),
        "mtime_ns": int(stat.st_mtime_ns),
        "size_bytes": int(stat.st_size),
        "sha256": template_hash,
    }


def _load_cache_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "entries": {}}
    if not isinstance(raw, Mapping):
        return {"version": 1, "entries": {}}
    entries = raw.get("entries")
    if not isinstance(entries, Mapping):
        raw = dict(raw)
        raw["entries"] = {}
    return dict(raw)


def _run_blender_command(cmd: list[str], *, label: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return
    details = [
        f"Blender command failed during {label} (exit {proc.returncode}).",
        "Command:",
        " ".join(cmd),
    ]
    if proc.stdout:
        details.extend(["--- stdout ---", proc.stdout[-4000:]])
    if proc.stderr:
        details.extend(["--- stderr ---", proc.stderr[-4000:]])
    raise RuntimeError("\n".join(details))


def _require_blender_bin(config: BlenderRenderConfig) -> str:
    if config.blender_bin:
        return str(config.blender_bin)
    raise RuntimeError(
        "Blender binary not found. Set BLENDER_BIN or install blender in PATH."
    )


def _string_or_empty(raw: Any) -> str:
    return str(raw).strip() if isinstance(raw, str) else ""


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def blender_scene_video_path(project_dir: Path, scene_id: int) -> Path:
    return Path(project_dir) / "videos" / f"scene_{int(scene_id):03d}.mp4"


def _coerce_animation_cfg(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"enabled": False}
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    duration = _float_or(raw.get("duration_sec"), 5.0, minimum=1.0)
    fps = int(_float_or(raw.get("fps"), 24.0, minimum=1.0))
    camera_orbit_deg = _float_or(raw.get("camera_orbit_deg"), 14.0)
    pulse_hz = _float_or(raw.get("pulse_hz"), 0.45, minimum=0.0)
    pulse_depth = _float_or(raw.get("pulse_depth"), 0.28, minimum=0.0)
    return {
        "enabled": True,
        "duration_sec": duration,
        "fps": fps,
        "camera_orbit_deg": camera_orbit_deg,
        "pulse_hz": pulse_hz,
        "pulse_depth": pulse_depth,
    }


def _float_or(raw: Any, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    if minimum is not None and value < minimum:
        return float(minimum)
    return value
