#!/usr/bin/env python3
"""Batch-refresh qEEG patient video stills with native Codex gpt-image-2 generation.

This script keeps the only agentic step narrowly scoped to what actually needs it:
native image generation. Everything else around it is deterministic and local:

- discover latest patient projects across local-explainer-video and Cathode
- filter to latest portal patients whose plans still carry usable visual prompts
- back up current still PNGs
- invoke `codex exec` per patient/project to regenerate prompt-bearing scene images
- re-render the MP4 in the source repo
- mirror local-explainer projects into Cathode when needed
- publish the final MP4 into qEEG-analysis/data/portal_patients/<PATIENT_ID>/
- optionally sync that patient to Thrylen immediately

Example:
    python3.10 scripts/qeeg_gpt_image2_batch.py --dry-run
    python3.10 scripts/qeeg_gpt_image2_batch.py --patients 10-31-2008-0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_DIR = Path.home()
LOCAL_EXPLAINER_ROOT = REPO_ROOT
CATHODE_ROOT = (REPO_ROOT / "../cathode").resolve()
QEEG_ANALYSIS_ROOT = (REPO_ROOT / "../qEEG-analysis").resolve()
PORTAL_PATIENTS_DIR = QEEG_ANALYSIS_ROOT / "data" / "portal_patients"
CODEX_GENERATED_IMAGES_ROOT = HOME_DIR / ".codex" / "generated_images"
PATIENT_ID_RE = re.compile(r"^\d{2}-\d{2}-\d{4}-\d+$")
VIDEO_VERSION_SUFFIX_RE = re.compile(r"([_ ]v\d+(?:\.\d+)?)$", re.IGNORECASE)
TARGET_WIDTH = 1664
TARGET_HEIGHT = 928
TARGET_ASPECT_RATIO = "16:9"
DEFAULT_THRYLEN_SYNC_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class ProjectCandidate:
    patient_id: str
    repo_name: str
    repo_root: str
    project_name: str
    project_dir: str
    plan_path: str
    video_path: str | None
    video_exists: bool
    recency_source: str
    recency_ts: float
    recency_iso: str | None
    scene_count: int
    prompt_scene_count: int
    promptless_image_scene_count: int
    prompt_scene_ids: list[int]
    promptless_image_scene_ids: list[int]
    portal_latest_video: str | None = None
    portal_latest_video_mtime: float | None = None
    portal_latest_video_iso: str | None = None
    selection_reason: str = ""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_from_timestamp(ts: float) -> str | None:
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_patient_id(value: str) -> bool:
    return bool(PATIENT_ID_RE.fullmatch((value or "").strip()))


def infer_patient_id(project_name: str) -> str | None:
    if not project_name:
        return None
    base = project_name.split("__", 1)[0]
    return base if is_patient_id(base) else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.partial")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _int_env(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _parse_iso_datetime(raw: Any) -> float:
    value = str(raw or "").strip()
    if not value:
        return 0.0
    candidate = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).timestamp()
    except ValueError:
        return 0.0


def _scene_is_image_like(scene: dict[str, Any]) -> bool:
    scene_type = str(scene.get("scene_type") or "").strip().lower()
    if not scene_type:
        return True
    return scene_type == "image"


def _count_prompt_scenes(scenes: Iterable[dict[str, Any]]) -> tuple[list[int], list[int]]:
    prompt_scene_ids: list[int] = []
    promptless_image_scene_ids: list[int] = []
    for index, scene in enumerate(scenes):
        scene_id = int(scene.get("id", index) or index)
        visual_prompt = str(scene.get("visual_prompt") or "").strip()
        if visual_prompt:
            prompt_scene_ids.append(scene_id)
            continue
        if _scene_is_image_like(scene):
            promptless_image_scene_ids.append(scene_id)
    return prompt_scene_ids, promptless_image_scene_ids


def _normalize_video_stem(name: str) -> str:
    stem = Path(name).stem
    stem = VIDEO_VERSION_SUFFIX_RE.sub("", stem).strip()
    return stem


def _resolve_project_video_path(plan: dict[str, Any], project_dir: Path) -> Path | None:
    meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
    raw_path = str(meta.get("video_path") or "").strip()
    if not raw_path:
        for candidate_name in (f"{project_dir.name}.mp4", "final_video.mp4"):
            candidate = project_dir / candidate_name
            if candidate.exists():
                return candidate.resolve()
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (project_dir / path).resolve()
    return path.resolve()


def _candidate_recency(plan: dict[str, Any], project_dir: Path, video_path: Path | None) -> tuple[str, float, str | None]:
    if video_path and video_path.exists():
        ts = video_path.stat().st_mtime
        return "video_mtime", ts, _iso_from_timestamp(ts)

    meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
    for field in ("rendered_utc", "created_utc"):
        ts = _parse_iso_datetime(meta.get(field))
        if ts > 0:
            return field, ts, _iso_from_timestamp(ts)

    ts = project_dir.stat().st_mtime
    return "project_mtime", ts, _iso_from_timestamp(ts)


def build_candidate(repo_name: str, repo_root: Path, project_dir: Path) -> ProjectCandidate | None:
    patient_id = infer_patient_id(project_dir.name)
    if not patient_id:
        return None

    plan_path = project_dir / "plan.json"
    if not plan_path.exists():
        return None

    plan = load_json(plan_path)
    scenes = plan.get("scenes")
    if not isinstance(scenes, list):
        scenes = []

    prompt_scene_ids, promptless_image_scene_ids = _count_prompt_scenes(
        scene for scene in scenes if isinstance(scene, dict)
    )
    video_path = _resolve_project_video_path(plan, project_dir)
    recency_source, recency_ts, recency_iso = _candidate_recency(plan, project_dir, video_path)

    return ProjectCandidate(
        patient_id=patient_id,
        repo_name=repo_name,
        repo_root=str(repo_root),
        project_name=project_dir.name,
        project_dir=str(project_dir.resolve()),
        plan_path=str(plan_path.resolve()),
        video_path=str(video_path) if video_path else None,
        video_exists=bool(video_path and video_path.exists()),
        recency_source=recency_source,
        recency_ts=recency_ts,
        recency_iso=recency_iso,
        scene_count=len([scene for scene in scenes if isinstance(scene, dict)]),
        prompt_scene_count=len(prompt_scene_ids),
        promptless_image_scene_count=len(promptless_image_scene_ids),
        prompt_scene_ids=prompt_scene_ids,
        promptless_image_scene_ids=promptless_image_scene_ids,
    )


def discover_repo_candidates(repo_name: str, repo_root: Path) -> dict[str, list[ProjectCandidate]]:
    projects_dir = repo_root / "projects"
    discovered: dict[str, list[ProjectCandidate]] = {}
    if not projects_dir.exists():
        return discovered

    for child in sorted(projects_dir.iterdir()):
        if not child.is_dir():
            continue
        candidate = build_candidate(repo_name, repo_root, child)
        if not candidate:
            continue
        discovered.setdefault(candidate.patient_id, []).append(candidate)
    return discovered


def choose_latest_candidate(candidates: Iterable[ProjectCandidate]) -> ProjectCandidate | None:
    ordered = sorted(
        candidates,
        key=lambda item: (item.recency_ts, item.video_exists, item.project_name),
        reverse=True,
    )
    return ordered[0] if ordered else None


def choose_portal_matched_candidate(
    patient_id: str,
    candidates: Iterable[ProjectCandidate],
    portal_latest_video: Path | None,
) -> ProjectCandidate | None:
    prompt_candidates = [candidate for candidate in candidates if candidate.prompt_scene_count > 0]
    if not prompt_candidates:
        return None

    if portal_latest_video is None:
        ordered = sorted(
            prompt_candidates,
            key=lambda item: (
                item.video_exists,
                item.recency_ts,
                item.repo_name == "local-explainer-video",
                item.project_name == patient_id,
                item.prompt_scene_count,
                item.project_name,
            ),
            reverse=True,
        )
        chosen = ordered[0]
        return replace(chosen, selection_reason="fallback_latest_source_no_portal_video")

    portal_name = portal_latest_video.name
    portal_norm = _normalize_video_stem(portal_name)
    portal_mtime = portal_latest_video.stat().st_mtime

    def score(item: ProjectCandidate) -> tuple[int, int, int, float, int, int, float, int, str]:
        candidate_name = Path(item.video_path).name if item.video_path else ""
        candidate_norm = _normalize_video_stem(candidate_name) if candidate_name else ""
        exact_name = int(candidate_name == portal_name and bool(candidate_name))
        normalized_name = int(candidate_norm == portal_norm and bool(candidate_norm))
        candidate_has_video = int(item.video_exists)
        gap = abs(item.recency_ts - portal_mtime) if item.recency_ts > 0 else float("inf")
        return (
            exact_name,
            normalized_name,
            candidate_has_video,
            -gap,
            int(item.project_name == patient_id),
            item.prompt_scene_count,
            item.recency_ts,
            int(item.repo_name == "local-explainer-video"),
            item.project_name,
        )

    chosen = sorted(prompt_candidates, key=score, reverse=True)[0]
    return replace(
        chosen,
        portal_latest_video=str(portal_latest_video.resolve()),
        portal_latest_video_mtime=portal_mtime,
        portal_latest_video_iso=_iso_from_timestamp(portal_mtime),
        selection_reason="matched_latest_portal_video",
    )


def discover_latest_prompt_projects(
    *,
    portal_patients_dir: Path = PORTAL_PATIENTS_DIR,
    local_explainer_root: Path = LOCAL_EXPLAINER_ROOT,
    cathode_root: Path = CATHODE_ROOT,
) -> list[ProjectCandidate]:
    portal_entries = sorted(
        entry
        for entry in portal_patients_dir.iterdir()
        if entry.is_dir() and is_patient_id(entry.name)
    )
    per_repo: dict[str, list[ProjectCandidate]] = {}
    for repo_candidates in (
        discover_repo_candidates("local-explainer-video", local_explainer_root),
        discover_repo_candidates("cathode", cathode_root),
    ):
        for patient_id, candidates in repo_candidates.items():
            per_repo.setdefault(patient_id, []).extend(candidates)

    selected: list[ProjectCandidate] = []
    for portal_entry in portal_entries:
        portal_videos = sorted(portal_entry.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
        latest = choose_portal_matched_candidate(
            portal_entry.name,
            per_repo.get(portal_entry.name, []),
            portal_videos[0] if portal_videos else None,
        )
        if latest:
            selected.append(latest)
    return sorted(selected, key=lambda item: item.patient_id)


def ensure_backup_dir(candidate: ProjectCandidate) -> Path:
    project_dir = Path(candidate.project_dir)
    backup_dir = project_dir / f"images_pre_gpt_image2_{utc_now().date().isoformat()}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    plan = load_json(Path(candidate.plan_path))
    for scene in plan.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        visual_prompt = str(scene.get("visual_prompt") or "").strip()
        image_path = Path(str(scene.get("image_path") or "")).expanduser()
        if not visual_prompt or not image_path.exists():
            continue
        dst = backup_dir / image_path.name
        if not dst.exists():
            shutil.copy2(image_path, dst)
    return backup_dir


def normalize_image_to_target(path: Path) -> tuple[int, int]:
    from PIL import Image, ImageFilter, ImageOps

    with Image.open(path) as img:
        source = img.convert("RGB")
        original_size = source.size

        if original_size == (TARGET_WIDTH, TARGET_HEIGHT):
            return original_size

        background = ImageOps.fit(
            source,
            (TARGET_WIDTH, TARGET_HEIGHT),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        background = background.filter(ImageFilter.GaussianBlur(radius=28))
        background = Image.blend(
            background,
            Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0)),
            0.32,
        )

        foreground = ImageOps.contain(
            source,
            (TARGET_WIDTH, TARGET_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )
        offset = (
            (TARGET_WIDTH - foreground.width) // 2,
            (TARGET_HEIGHT - foreground.height) // 2,
        )
        background.paste(foreground, offset)
        background.save(path, format="PNG", optimize=True)
        return original_size


def normalize_project_images(candidate: ProjectCandidate) -> list[dict[str, Any]]:
    plan = load_json(Path(candidate.plan_path))
    normalized: list[dict[str, Any]] = []
    for scene in plan.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        image_path_raw = str(scene.get("image_path") or "").strip()
        visual_prompt = str(scene.get("visual_prompt") or "").strip()
        if not image_path_raw or not visual_prompt:
            continue
        image_path = Path(image_path_raw).expanduser()
        if not image_path.exists():
            continue
        before = normalize_image_to_target(image_path)
        normalized.append(
            {
                "scene_id": int(scene.get("id", 0)),
                "path": str(image_path),
                "before": list(before),
                "after": [TARGET_WIDTH, TARGET_HEIGHT],
            }
        )
    return normalized


def build_codex_prompt(candidate: ProjectCandidate) -> str:
    project_dir = Path(candidate.project_dir)
    plan = load_json(Path(candidate.plan_path))
    prompt_scene_ids = []
    scene_jobs: list[dict[str, Any]] = []
    for scene in plan.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        visual_prompt = str(scene.get("visual_prompt") or "").strip()
        image_path = str(scene.get("image_path") or "").strip()
        if visual_prompt and image_path:
            scene_id = int(scene.get("id", 0))
            prompt_scene_ids.append(scene_id)
            scene_jobs.append(
                {
                    "scene_id": scene_id,
                    "image_path": image_path,
                    "visual_prompt": visual_prompt,
                }
            )
    scene_id_text = ", ".join(str(scene_id) for scene_id in prompt_scene_ids)
    scene_jobs_json = json.dumps(scene_jobs, indent=2, ensure_ascii=False)

    return (
        f"Work in {candidate.repo_root}.\n"
        "Use only Codex's built-in native image generation capability in this session.\n"
        "Do not use any skill script, CLI wrapper, MCP imagegen helper, Python image client, or API-key-based workflow.\n"
        "Do not inspect `~/.codex/skills`, `.env` files, config files, or search the filesystem for `OPENAI_API_KEY` or any other secret.\n"
        "If the built-in native image generation capability is unavailable, stop immediately and fail without further exploration.\n"
        "You may use shell commands only for reading the listed project files, copying the already-generated native image outputs into place, and verifying the resulting PNGs.\n\n"
        f"Read {candidate.plan_path} only if you need extra confirmation, but the exact job list is already provided below.\n\n"
        "Requirements:\n"
        "- Use the native image generation tool with gpt-image-2.\n"
        "- Use each scene's existing visual_prompt text as the core content prompt.\n"
        f"- Add only these fixed render constraints to every generation: landscape {TARGET_ASPECT_RATIO}, widescreen slide, target frame {TARGET_WIDTH}x{TARGET_HEIGHT}, no square composition, no portrait composition, keep all text and important graphics fully visible inside safe margins.\n"
        "- Do not otherwise rewrite, refine, or summarize the prompts.\n"
        "- Any quoted on-screen text or branded term in the prompt must be rendered exactly and case-sensitively. `LUMIT` must stay `LUMIT`, never `Lumen`.\n"
        "- Only operate on scenes that have both a non-empty visual_prompt and an existing image_path.\n"
        f"- For this project, the expected prompt-bearing scene ids are: {scene_id_text or '(none)'}.\n"
        "- Copy each generated PNG into the scene's existing target image_path.\n"
        "- Do not delete originals from ~/.codex/generated_images.\n"
        "- Do not modify plan.json, audio files, video files, or any other repo files.\n"
        f"- After finishing, verify the updated target PNGs exist and report each path plus file size and dimensions. The ideal final dimensions are {TARGET_WIDTH}x{TARGET_HEIGHT}.\n"
        f"- The project directory is {project_dir}.\n"
        "\nExact scene jobs:\n"
        f"{scene_jobs_json}\n"
    )


def run_codex_refresh(candidate: ProjectCandidate, *, run_dir: Path, model: str | None) -> tuple[int, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / f"{candidate.patient_id}.codex.jsonl"
    final_message_path = run_dir / f"{candidate.patient_id}.final.txt"

    cmd = [
        "codex",
        "exec",
        "--json",
        "--ignore-user-config",
        "-C",
        candidate.repo_root,
        "-s",
        "danger-full-access",
        "-c",
        'approval_policy="never"',
        "-o",
        str(final_message_path),
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append("-")

    prompt = build_codex_prompt(candidate)
    with jsonl_path.open("w", encoding="utf-8") as stdout_handle:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode, jsonl_path, final_message_path


def update_plan_metadata(candidate: ProjectCandidate) -> None:
    plan_path = Path(candidate.plan_path)
    plan = load_json(plan_path)
    meta = plan.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"Expected plan.meta to be an object in {plan_path}")
    meta["image_model"] = "gpt-image-2"
    meta["image_regen_note"] = f"Images regenerated via native Codex image tool on {utc_now().date().isoformat()}"
    meta["image_regenerated_utc"] = utc_now_iso()
    write_json(plan_path, plan)


def determine_output_filename(plan: dict[str, Any], project_dir: Path, patient_id: str) -> str:
    meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
    video_path = str(meta.get("video_path") or "").strip()
    if video_path:
        return Path(video_path).name
    for candidate in (f"{project_dir.name}.mp4", f"{patient_id}.mp4", "final_video.mp4"):
        if (project_dir / candidate).exists():
            return candidate
    return f"{patient_id}.mp4"


def rerender_local_explainer(candidate: ProjectCandidate) -> Path:
    plan = load_json(Path(candidate.plan_path))
    project_dir = Path(candidate.project_dir)
    output_filename = determine_output_filename(plan, project_dir, candidate.patient_id)
    fps = int(plan.get("meta", {}).get("fps") or 24) if isinstance(plan.get("meta"), dict) else 24

    cmd = [
        sys.executable,
        "-c",
        (
            "import json, pathlib; "
            "from core.video_assembly import assemble_video; "
            f"project_dir = pathlib.Path({project_dir.as_posix()!r}); "
            f"plan_path = pathlib.Path({Path(candidate.plan_path).as_posix()!r}); "
            "plan = json.loads(plan_path.read_text(encoding='utf-8')); "
            f"rendered = assemble_video([scene for scene in plan.get('scenes', []) if isinstance(scene, dict)], "
            f"project_dir, output_filename={output_filename!r}, fps={fps}); "
            "plan.setdefault('meta', {})['video_path'] = str(pathlib.Path(rendered).resolve()); "
            "plan.setdefault('meta', {})['rendered_utc'] = "
            f"{utc_now_iso()!r}; "
            "plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8'); "
            "print(pathlib.Path(rendered).resolve())"
        ),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Local-explainer render failed for {candidate.project_name}: {proc.stdout}\n{proc.stderr}")
    return Path(proc.stdout.strip().splitlines()[-1]).resolve()


def rerender_cathode(candidate: ProjectCandidate) -> Path:
    project_dir = Path(candidate.project_dir)
    plan = load_json(Path(candidate.plan_path))
    output_filename = determine_output_filename(plan, project_dir, candidate.patient_id)
    render_profile = plan.get("meta", {}).get("render_profile") if isinstance(plan.get("meta"), dict) else {}
    fps = 24
    if isinstance(render_profile, dict):
        try:
            fps = int(render_profile.get("fps") or 24)
        except (TypeError, ValueError):
            fps = 24

    cmd = [
        sys.executable,
        "-c",
        (
            "import json, pathlib; "
            "from core.video_assembly import assemble_video; "
            f"project_dir = pathlib.Path({project_dir.as_posix()!r}); "
            f"plan_path = pathlib.Path({Path(candidate.plan_path).as_posix()!r}); "
            "plan = json.loads(plan_path.read_text(encoding='utf-8')); "
            f"rendered = assemble_video(plan.get('scenes', []), project_dir, output_filename={output_filename!r}, fps={fps}, "
            "render_profile=(plan.get('meta', {}) or {}).get('render_profile')); "
            "plan.setdefault('meta', {})['video_path'] = str(pathlib.Path(rendered).resolve()); "
            "plan.setdefault('meta', {})['rendered_utc'] = "
            f"{utc_now_iso()!r}; "
            "plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8'); "
            "print(pathlib.Path(rendered).resolve())"
        ),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(CATHODE_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Cathode render failed for {candidate.project_name}: {proc.stdout}\n{proc.stderr}")
    rendered_path = Path(proc.stdout.strip().splitlines()[-1]).resolve()
    return rendered_path


def mirror_to_cathode(candidate: ProjectCandidate, video_path: Path) -> Path:
    if candidate.repo_name != "local-explainer-video":
        return Path(candidate.project_dir)

    cathode_dir = CATHODE_ROOT / "projects" / candidate.patient_id
    cathode_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("images", "audio"):
        (cathode_dir / subdir).mkdir(parents=True, exist_ok=True)

    plan = load_json(Path(candidate.plan_path))
    mirrored_plan = json.loads(json.dumps(plan))
    for scene in mirrored_plan.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        scene_id = int(scene.get("id", 0))
        src_img = Path(str(scene.get("image_path") or ""))
        if src_img.exists():
            dst_img = cathode_dir / "images" / f"scene_{scene_id:03d}.png"
            shutil.copy2(src_img, dst_img)
            scene["image_path"] = str(dst_img)
        src_audio = Path(str(scene.get("audio_path") or ""))
        if src_audio.exists():
            dst_audio = cathode_dir / "audio" / f"scene_{scene_id:03d}.wav"
            shutil.copy2(src_audio, dst_audio)
            scene["audio_path"] = str(dst_audio)
        scene.setdefault("scene_type", "image")
        scene.setdefault("video_path", None)
        scene.setdefault("preview_path", None)

    dst_video = cathode_dir / f"{candidate.patient_id}.mp4"
    shutil.copy2(video_path, dst_video)
    source_md = Path(candidate.project_dir) / "source_qeeg.md"
    if source_md.exists():
        shutil.copy2(source_md, cathode_dir / "source_qeeg.md")

    meta = mirrored_plan.setdefault("meta", {})
    meta["project_name"] = candidate.patient_id
    meta["created_by"] = "local-explainer-video"
    meta["source_pipeline"] = "legacy_qwen_no_remotion"
    meta["video_path"] = str(dst_video)
    meta["image_model"] = "gpt-image-2"
    meta["image_regen_note"] = f"Images regenerated via native Codex image tool on {utc_now().date().isoformat()}"
    write_json(cathode_dir / "plan.json", mirrored_plan)
    return cathode_dir


def publish_to_portal(patient_id: str, video_path: Path) -> Path:
    out_dir = PORTAL_PATIENTS_DIR / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{patient_id}.mp4"
    tmp = dest.with_name(f".{dest.name}.partial")
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    if dest.exists():
        dest.unlink()
    try:
        os.link(video_path, dest)
    except Exception:
        shutil.copy2(video_path, tmp)
        tmp.replace(dest)
    return dest


def sync_patient_to_thrylen(patient_id: str) -> bool:
    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [uv_bin, "run", "python", "-m", "backend.portal_sync", "--patient-label", patient_id]
    else:
        cmd = [sys.executable, "-m", "backend.portal_sync", "--patient-label", patient_id]
    timeout_seconds = _int_env("QEEG_THRYLEN_SYNC_TIMEOUT_SECONDS", DEFAULT_THRYLEN_SYNC_TIMEOUT_SECONDS)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(QEEG_ANALYSIS_ROOT),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-refresh qEEG prompt-bearing patient stills with native Codex gpt-image-2.")
    parser.add_argument("--dry-run", action="store_true", help="Discover candidates and print the batch plan without running Codex or rendering.")
    parser.add_argument("--patients", type=str, default="", help="Comma-separated patient ids to limit the run.")
    parser.add_argument("--max-patients", type=int, default=0, help="Optional cap after discovery/filtering.")
    parser.add_argument("--model", type=str, default="", help="Optional Codex exec model override.")
    parser.add_argument("--skip-thrylen-sync", action="store_true", help="Only update qEEG-analysis/data/portal_patients; do not push the patient into Thrylen immediately.")
    parser.add_argument("--run-label", type=str, default="", help="Optional label for the batch run directory.")
    return parser.parse_args()


def _selected_patients(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _patient_results_dir(label: str) -> Path:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()) if label.strip() else stamp
    return REPO_ROOT / "projects" / "queue_logs" / f"qeeg_gpt_image2_batch_{safe_label}_{stamp}"


def main() -> int:
    args = parse_args()
    selected_patients = _selected_patients(args.patients)
    run_dir = _patient_results_dir(args.run_label or ("dryrun" if args.dry_run else "run"))
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_latest_prompt_projects()
    if selected_patients:
        candidates = [candidate for candidate in candidates if candidate.patient_id in selected_patients]
    if args.max_patients > 0:
        candidates = candidates[: args.max_patients]

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "generated_at": utc_now_iso(),
        "dry_run": bool(args.dry_run),
        "model": args.model or None,
        "skip_thrylen_sync": bool(args.skip_thrylen_sync),
        "candidate_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    write_json(manifest_path, manifest)

    print(f"Discovered {len(candidates)} eligible patient project(s).")
    print(f"Manifest: {manifest_path}")

    if args.dry_run:
        for candidate in candidates:
            print(
                f"- {candidate.patient_id}: {candidate.repo_name}/{candidate.project_name} "
                f"(prompt_scenes={candidate.prompt_scene_count}, recency={candidate.recency_source}:{candidate.recency_iso})"
            )
        return 0

    results: list[dict[str, Any]] = []
    overall_ok = True
    for candidate in candidates:
        print("\n" + "=" * 80)
        print(f"{candidate.patient_id} -> {candidate.repo_name}/{candidate.project_name}")
        patient_result: dict[str, Any] = {
            "patient_id": candidate.patient_id,
            "repo_name": candidate.repo_name,
            "project_name": candidate.project_name,
            "project_dir": candidate.project_dir,
        }

        try:
            backup_dir = ensure_backup_dir(candidate)
            patient_result["backup_dir"] = str(backup_dir)
            print(f"Backup ready: {backup_dir}")

            returncode, jsonl_path, final_message_path = run_codex_refresh(candidate, run_dir=run_dir, model=args.model or None)
            patient_result["codex_log"] = str(jsonl_path)
            patient_result["codex_final_message"] = str(final_message_path)
            patient_result["codex_exit_code"] = returncode
            if returncode != 0:
                raise RuntimeError(f"codex exec failed for {candidate.patient_id}; inspect {jsonl_path}")

            normalized_images = normalize_project_images(candidate)
            patient_result["normalized_images"] = normalized_images
            update_plan_metadata(candidate)

            if candidate.repo_name == "local-explainer-video":
                rendered_path = rerender_local_explainer(candidate)
                mirrored_cathode_dir = mirror_to_cathode(candidate, rendered_path)
                patient_result["mirrored_cathode_dir"] = str(mirrored_cathode_dir)
            else:
                rendered_path = rerender_cathode(candidate)

            patient_result["rendered_video_path"] = str(rendered_path)
            portal_copy_path = publish_to_portal(candidate.patient_id, rendered_path)
            patient_result["portal_copy_path"] = str(portal_copy_path)

            if not args.skip_thrylen_sync:
                patient_result["thrylen_sync_ok"] = sync_patient_to_thrylen(candidate.patient_id)
            else:
                patient_result["thrylen_sync_ok"] = None

            patient_result["ok"] = True
            print(f"Rendered: {rendered_path}")
            print(f"Portal copy: {portal_copy_path}")
            if patient_result["thrylen_sync_ok"] is True:
                print("Thrylen sync: ok")
            elif patient_result["thrylen_sync_ok"] is False:
                print("Thrylen sync: failed")
        except Exception as exc:
            overall_ok = False
            patient_result["ok"] = False
            patient_result["error"] = str(exc)
            print(f"FAILED: {exc}")
        results.append(patient_result)
        write_json(run_dir / "results.json", {"generated_at": utc_now_iso(), "results": results})

    ok_count = sum(1 for result in results if result.get("ok"))
    fail_count = len(results) - ok_count
    print("\n" + "-" * 80)
    print(f"Done. ok={ok_count}, failed={fail_count}")
    print(f"Results: {run_dir / 'results.json'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
