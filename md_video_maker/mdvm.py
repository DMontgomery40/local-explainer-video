#!/opt/homebrew/bin/python3.10
"""Markdown-driven still-video maker.

This experiment keeps the application brain in a handful of markdown files and
uses a very small Python runtime only for exact execution:

- call Codex text planning against markdown rules + BRIEF.md
- call local-explainer-video gpt-image/TTS helpers
- assemble a slideshow MP4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
LOCAL_EXPLAINER_ROOT = THIS_DIR.parent
FACTORY_DOCS = [
    THIS_DIR / "VIDEO_BRAIN.md",
    THIS_DIR / "GOOD_EXAMPLES.md",
]

sys.path.insert(0, str(LOCAL_EXPLAINER_ROOT))
from core.generation_receipts import (AssetOperation, AssetFailures, ReceiptConflict, UnknownDispatch, atomic_bytes, atomic_json, digest_bytes, exclusive_lock)
from core.image_gen import generate_scene_image, codex_runtime_scope  # type: ignore  # noqa: E402
from core.video_assembly import assemble_video  # type: ignore  # noqa: E402
from core.voice_gen import generate_scene_audio  # type: ignore  # noqa: E402

from md_video_maker.mixed_video_assembly import assemble_mixed_video  # noqa: E402


def tool_binary(name: str) -> str:
    candidate = Path("/opt/homebrew/bin") / name
    if candidate.exists():
        return str(candidate)
    return name


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_plan_path(project_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = project_dir / candidate
    return candidate.resolve()


def plan_path_string(project_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_dir.resolve()))
    except ValueError:
        return str(resolved)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            tool_binary("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def render_profile(plan: dict[str, Any]) -> dict[str, Any]:
    meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
    for candidate in (meta.get("render_profile"), plan.get("render_profile"), meta.get("video_profile")):
        if isinstance(candidate, dict):
            return candidate
    return {}


def render_dimensions(plan: dict[str, Any]) -> tuple[int, int]:
    profile = render_profile(plan)
    width = _positive_int(profile.get("width"), 1664)
    height = _positive_int(profile.get("height"), 928)
    orientation = str(profile.get("orientation") or profile.get("aspect_ratio") or profile.get("aspect") or "").lower()
    if ("9:16" in orientation or "portrait" in orientation or "vertical" in orientation) and height <= width:
        return 1080, 1920
    if ("16:9" in orientation or "landscape" in orientation or "horizontal" in orientation) and width <= height:
        return 1664, 928
    return width, height


def apply_render_dimensions(scene: dict[str, Any], width: int, height: int) -> None:
    scene["render_width"] = width
    scene["render_height"] = height
    scene["render_orientation"] = "portrait" if height > width else "landscape"
    scene["render_aspect_ratio"] = "9:16" if height > width else "16:9"


def list_local_visual_assets(project_dir: Path) -> list[Path]:
    assets_dir = project_dir / "assets"
    if not assets_dir.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        path.resolve()
        for path in assets_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def build_planner_prompt(project_dir: Path, brief_text: str) -> str:
    docs_blob = "\n\n".join(
        f"===== {doc.name} =====\n{read_text(doc)}" for doc in FACTORY_DOCS
    )
    project_name = project_dir.name
    local_assets = list_local_visual_assets(project_dir)
    if local_assets:
        assets_blob = "\n".join(f"- {path}" for path in local_assets)
    else:
        assets_blob = "- none"
    return f"""You are the planning engine for a markdown-driven still-video factory.

Read the factory markdown documents below as the application brain. Follow them.

{docs_blob}

===== BRIEF.md =====
{brief_text}

===== AVAILABLE LOCAL VISUAL ASSETS =====
{assets_blob}

Now produce ONLY valid JSON. No markdown fences. No commentary.

JSON shape:
{{
  "meta": {{
    "title": "string",
    "project_name": "{project_name}",
    "summary": "1-3 sentence summary",
    "image_model": "gpt-image-2",
    "tts_provider": "elevenlabs",
    "voice": "Antoni",
    "elevenlabs_model_id": "eleven_multilingual_v2"
  }},
  "scenes": [
    {{
      "id": 0,
      "title": "string",
      "narration": "string",
      "visual_prompt": "string",
      "image_source_path": "/absolute/path/to/local/asset.png",
      "video_source_path": "/absolute/path/to/local/source.mov"
    }}
  ]
}}

Rules:
- follow the markdown docs exactly
- every scene must include `id`, `title`, and `narration`
- every scene must include at least one of `visual_prompt`, `image_source_path`, or `video_source_path`
- if you use `image_source_path`, use an exact path from the available local visual assets list
- keep narration natural and speakable
- treat the runtime target in the brief as real; for a five-minute explainer, landing below roughly 85 percent of target runtime is a failure
- when on-screen text matters, include exact quoted strings in the visual prompt
- explain the system honestly but do not undersell how much logic lives in markdown

Return JSON only.
"""


def build_brief_prompt(project_dir: Path, user_prompt_text: str) -> str:
    docs_blob = "\n\n".join(
        f"===== {doc.name} =====\n{read_text(doc)}" for doc in FACTORY_DOCS
    )
    project_name = project_dir.name
    return f"""You are the brief-writing stage for a markdown-driven still-video factory.

Read the markdown documents below as the system brain. Follow them.

{docs_blob}

===== USER_PROMPT.md =====
{user_prompt_text}

Write ONLY the contents of a strong `BRIEF.md`.

Rules:
- do not ask follow-up questions
- do not output commentary
- do not output markdown fences
- output a usable production brief
- preserve the spirit of the user's request
- make enough decisions that the planner can make a strong `plan.json`
- keep the strict still-image lane if the docs require it
- project name is `{project_name}`
"""


def run_codex_text(prompt: str, workdir: Path, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "planner.final.txt"
    jsonl_path = output_dir / "planner.jsonl"
    cmd = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
        # codex-cli's built-in default model (gpt-5.3-codex) is rejected on ChatGPT-account
        # auth as of 2026-07; pin an account-supported model explicitly since
        # --ignore-user-config bypasses the working pin in ~/.codex/config.toml.
        "-m",
        "gpt-5.5",
        "-s",
        "danger-full-access",
        "-c",
        'approval_policy="never"',
        "-o",
        str(final_path),
        "-",
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            cwd=str(workdir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"codex exec plan failed; inspect {jsonl_path}")
    return final_path.read_text(encoding="utf-8").strip()


def create_brief(project_dir: Path, *, force: bool = False) -> Path:
    prompt_path = project_dir / "USER_PROMPT.md"
    brief_path = project_dir / "BRIEF.md"
    log_dir = project_dir / "logs"
    if brief_path.exists() and not force:
        return brief_path
    if not prompt_path.exists():
        if brief_path.exists():
            return brief_path
        raise FileNotFoundError(f"Missing USER_PROMPT.md at {prompt_path}")

    brief_text = run_codex_text(
        build_brief_prompt(project_dir, read_text(prompt_path)),
        workdir=project_dir,
        output_dir=log_dir,
    )
    brief_path.write_text(brief_text.strip() + "\n", encoding="utf-8")
    return brief_path


def normalize_scene(scene: dict[str, Any], scene_id: int, project_dir: Path) -> dict[str, Any]:
    narration = str(scene.get("narration", "")).strip()
    prompt = str(scene.get("visual_prompt", "")).strip()
    source_path = str(scene.get("image_source_path", "")).strip()
    video_source_path = str(scene.get("video_source_path", "")).strip()
    title = str(scene.get("title", f"Scene {scene_id + 1}")).strip()
    if not narration or (not prompt and not source_path and not video_source_path):
        raise ValueError(
            f"Scene {scene_id} is missing narration and visual path "
            f"(need visual_prompt, image_source_path, or video_source_path)"
        )
    image_path = resolve_plan_path(project_dir, source_path) or (
        project_dir / "images" / f"scene_{scene_id:03d}.png"
    )
    audio_path = project_dir / "audio" / f"scene_{scene_id:03d}.wav"
    resolved_source_path = resolve_plan_path(project_dir, source_path)
    resolved_video_source_path = resolve_plan_path(project_dir, video_source_path)
    return {
        "id": scene_id,
        "uid": uuid4().hex[:8],
        "title": title,
        "narration": narration,
        "visual_prompt": prompt,
        "image_source_path": plan_path_string(project_dir, resolved_source_path),
        "video_source_path": plan_path_string(project_dir, resolved_video_source_path),
        "video_start_seconds": scene.get("video_start_seconds"),
        "refinement_history": [],
        "image_path": plan_path_string(project_dir, image_path),
        "audio_path": plan_path_string(project_dir, audio_path),
    }


def estimate_seconds_from_words(text: str, words_per_minute: int = 145) -> float:
    words = len([w for w in text.split() if w.strip()])
    return words / words_per_minute * 60.0


def create_plan(project_dir: Path, *, force: bool = False) -> Path:
    brief_path = project_dir / "BRIEF.md"
    plan_path = project_dir / "plan.json"
    log_dir = project_dir / "logs"
    if plan_path.exists() and not force:
        return plan_path
    if not brief_path.exists():
        create_brief(project_dir, force=False)
    if not brief_path.exists():
        raise FileNotFoundError(f"Missing BRIEF.md at {brief_path}")

    raw = run_codex_text(
        build_planner_prompt(project_dir, read_text(brief_path)),
        workdir=project_dir,
        output_dir=log_dir,
    )
    data = json.loads(raw)
    scenes_in = data.get("scenes")
    if not isinstance(scenes_in, list) or not scenes_in:
        raise ValueError("Planner returned no scenes")

    normalized_scenes = [
        normalize_scene(scene, idx, project_dir)
        for idx, scene in enumerate(scenes_in)
    ]
    est_seconds = sum(estimate_seconds_from_words(scene["narration"]) for scene in normalized_scenes)
    payload = {
        "meta": {
            "project_name": project_dir.name,
            "created_utc": utc_now(),
            "llm_provider": "codex_exec_markdown_factory",
            "image_model": "gpt-image-2",
            "tts_provider": str(data.get("meta", {}).get("tts_provider", "elevenlabs")),
            "voice": str(data.get("meta", {}).get("voice", "Antoni")),
            "audio_speed": 1.0,
            "title": str(data.get("meta", {}).get("title", project_dir.name)),
            "summary": str(data.get("meta", {}).get("summary", "")),
            "brief_path": str(brief_path),
            "estimated_duration_seconds": round(est_seconds, 1),
            "elevenlabs_model_id": str(data.get("meta", {}).get("elevenlabs_model_id", "eleven_multilingual_v2")),
        },
        "scenes": normalized_scenes,
    }
    save_json(plan_path, payload)
    return plan_path


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _fingerprint_path(asset_path: Path) -> Path:
    return asset_path.with_name(asset_path.name + ".src.sha256")


def _validate_asset(path: Path) -> None:
    if path.suffix.lower() == ".png":
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    elif path.suffix.lower() == ".wav":
        import wave
        with wave.open(str(path), "rb") as audio:
            if audio.getnframes() <= 0:
                raise ReceiptConflict(f"Empty audio: {path}")
            expected = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
            if len(audio.readframes(audio.getnframes())) != expected:
                raise ReceiptConflict(f"Truncated audio: {path}")
    elif ffprobe_duration(path) <= 0:
        raise ReceiptConflict(f"Empty media: {path}")


def _asset_current(asset_path: Path, digest: str, settings: dict | None = None) -> bool:
    """Reuse exact validated assets; retain legacy source-fingerprint compatibility."""
    if not asset_path.exists():
        return False
    try:
        current = _fingerprint_path(asset_path).read_text(encoding="utf-8").strip() == digest
    except OSError:
        return False
    if not current:
        return False
    record_path = asset_path.with_name(asset_path.name + ".receipt.json")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("sha256") != digest_bytes(asset_path.read_bytes()):
            raise ReceiptConflict(f"Accepted asset hash mismatch: {asset_path}")
        if record.get("source_digest") != digest:
            raise ReceiptConflict(f"Accepted asset source mismatch: {asset_path}")
        if settings is not None and not record.get("legacy_source_verified") and record.get("settings") != settings:
            return False
    _validate_asset(asset_path)
    if not record_path.exists():
        atomic_json(record_path, {"source_digest": digest, "sha256": digest_bytes(asset_path.read_bytes()),
                                  "legacy_source_verified": True})
    return True


def _record_asset(asset_path: Path, digest: str, settings: dict | None = None) -> None:
    atomic_json(asset_path.with_name(asset_path.name + ".receipt.json"),
                {"source_digest": digest, "sha256": digest_bytes(asset_path.read_bytes()),
                 "settings": settings or {}})
    atomic_bytes(_fingerprint_path(asset_path), (digest + "\n").encode())


def _generate_asset(operation_dir: Path, attempt_id: str, asset_id: str,
                    canonical: Path, digest: str, settings: dict,
                    generate, *, recovery: bool) -> None:
    """Convert/reconcile within the owned operation, then atomically promote."""
    directory = operation_dir / attempt_id / asset_id
    if recovery and not directory.exists():
        raise UnknownDispatch(f"Asset has no recovery evidence: {asset_id}")
    with AssetOperation(operation_dir, attempt_id, asset_id) as operation:
        manifest = operation.directory / "asset.json"
        identity = {"source_digest": digest, "settings": settings}
        prepared = operation.directory / "prepared.json"
        if prepared.exists():
            if json.loads(prepared.read_text()) != identity:
                raise ReceiptConflict(f"Prepared asset intent mismatch: {prepared}")
        elif recovery:
            raise UnknownDispatch(f"Asset has no prepared recovery intent: {asset_id}")
        if manifest.exists():
            record = json.loads(manifest.read_text())
            if any(record.get(key) != value for key, value in identity.items()):
                raise ReceiptConflict(f"Incompatible asset operation: {manifest}")
        else:
            record = dict(identity)
            atomic_json(manifest, record)
        staging = operation.directory / "staging"
        exact = staging / canonical.parent.name / canonical.name
        # A completed operation-owned file is free to reconcile after a crash.
        if record.get("output_sha256"):
            if not exact.exists() or digest_bytes(exact.read_bytes()) != record["output_sha256"]:
                raise ReceiptConflict(f"Operation output hash mismatch: {exact}")
        else:
            result = Path(generate(staging))
            if result.resolve() != exact.resolve() or not exact.is_file():
                raise ReceiptConflict(f"Generator did not return exact new asset: {exact}")
            _validate_asset(exact)
            record["output_sha256"] = digest_bytes(exact.read_bytes())
            atomic_json(manifest, record)
        _validate_asset(exact)
        if canonical.exists():
            previous = canonical.read_bytes()
            archive_dir = operation_dir / "accepted-assets" / digest_bytes(previous)
            archive = archive_dir / canonical.name
            if not archive.exists():
                atomic_bytes(archive, previous)
                for sidecar in (_fingerprint_path(canonical), canonical.with_name(canonical.name + ".receipt.json")):
                    if sidecar.exists():
                        atomic_bytes(archive_dir / sidecar.name, sidecar.read_bytes())
        atomic_bytes(canonical, exact.read_bytes())
        _record_asset(canonical, digest, settings)


def _scene_image_fingerprint(scene: dict[str, Any], width: int, height: int) -> str:
    return _fingerprint(
        {
            "visual_prompt": str(scene.get("visual_prompt") or ""),
            "on_screen_text": scene.get("on_screen_text"),
            "title": str(scene.get("title") or ""),
            "width": width,
            "height": height,
        }
    )


def _scene_audio_fingerprint(scene: dict[str, Any], voice_settings: dict[str, Any]) -> str:
    return _fingerprint({"narration": str(scene.get("narration") or ""), **voice_settings})


def render_project(project_dir: Path, *, force_images: bool = False, force_audio: bool = False,
                   attempt_id: str | None = None, operation_dir: Path | None = None,
                   recovery: bool = False) -> Path:
    project_dir = Path(project_dir).resolve()
    operations = Path(operation_dir) if operation_dir is not None else project_dir / ".render-operations"
    if recovery and not attempt_id:
        raise ValueError("Recovery requires the original attempt_id")
    with exclusive_lock(project_dir / ".render-operations" / "project.lock"), codex_runtime_scope():
        return _render_project(project_dir, force_images=force_images, force_audio=force_audio,
                               attempt_id=attempt_id or uuid4().hex, operation_dir=operations,
                               recovery=recovery)


def _render_project(project_dir: Path, *, force_images: bool, force_audio: bool,
                    attempt_id: str, operation_dir: Path, recovery: bool) -> Path:
    plan_path = project_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing plan.json at {plan_path}")

    if not attempt_id or Path(attempt_id).name != attempt_id or attempt_id in {".", ".."}:
        raise ValueError("Invalid attempt_id")
    plan = load_plan(plan_path)
    scenes = plan.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("plan.json has no scenes")
    target_width, target_height = render_dimensions(plan)

    meta = plan.get("meta", {})
    tts_provider = str(meta.get("tts_provider", "elevenlabs"))
    voice = str(meta.get("voice", "Antoni"))
    audio_speed = float(meta.get("audio_speed", 1.0))
    tts_model = str(meta.get("tts_model", "tts-1-hd")).strip()
    voice_instructions = str(meta.get("voice_instructions", "")).strip()
    elevenlabs_model_id = str(meta.get("elevenlabs_model_id", "")).strip() or "eleven_multilingual_v2"
    elevenlabs_text_normalization = str(
        meta.get("elevenlabs_apply_text_normalization", "auto")
    ).strip() or "auto"
    elevenlabs_stability = float(meta.get("elevenlabs_stability", 0.4))
    elevenlabs_similarity_boost = float(meta.get("elevenlabs_similarity_boost", 0.75))
    elevenlabs_style = float(meta.get("elevenlabs_style", 0.4))
    elevenlabs_use_speaker_boost = bool(meta.get("elevenlabs_use_speaker_boost", True))
    if tts_provider == "kokoro":
        raise ValueError("Kokoro narration is disabled for clinic renders; use ElevenLabs or OpenAI tts-1-hd.")
    if tts_model.startswith("gpt-realtime") or tts_model.startswith("gpt-4o"):
        raise ValueError("OpenAI realtime and 4o-based voices are disabled for clinic narration; use ElevenLabs or OpenAI tts-1-hd.")

    image_settings = {
        "provider": os.getenv("LOCAL_EXPLAINER_IMAGE_PROVIDER") or "codex",
        "model": os.getenv("LOCAL_EXPLAINER_IMAGE_MODEL") or "gpt-image-1",
        "codex_image_model": "gpt-image-2",
        "quality": os.getenv("LOCAL_EXPLAINER_IMAGE_QUALITY") or "high",
        "base_url": os.getenv("OPENAI_IMAGE_BASE_URL") or "https://api.openai.com/v1",
        "runner_model": os.getenv("LOCAL_EXPLAINER_CODEX_IMAGE_RUNNER_MODEL") or os.getenv("CODEX_IMAGE_RUNNER_MODEL") or "",
        "width": target_width, "height": target_height,
    }
    voice_settings = {
        "tts_provider": tts_provider,
        "voice": voice,
        "speed": audio_speed,
        "tts_model": tts_model,
        "voice_instructions": voice_instructions,
        "elevenlabs_model_id": elevenlabs_model_id,
        "elevenlabs_text_normalization": elevenlabs_text_normalization,
        "elevenlabs_stability": elevenlabs_stability,
        "elevenlabs_similarity_boost": elevenlabs_similarity_boost,
        "elevenlabs_style": elevenlabs_style,
        "elevenlabs_use_speaker_boost": elevenlabs_use_speaker_boost,
    }
    manifest_path = operation_dir / attempt_id / "attempt.json"
    local_sources = {}
    asset_specs = {}
    for idx, source_scene in enumerate(scenes):
        scene = dict(source_scene)
        apply_render_dimensions(scene, target_width, target_height)
        scene_id = int(scene.get("id", idx))
        for key in ("image_source_path", "video_source_path"):
            if scene.get(key):
                source = resolve_plan_path(project_dir, str(scene[key]))
                local_sources[str(source)] = digest_bytes(source.read_bytes())
        if f"audio-{scene_id}" in asset_specs:
            raise ValueError("Scene ids must be unique within an attempt")
        asset_specs[f"audio-{scene_id}"] = {
            "source_digest": _scene_audio_fingerprint(scene, voice_settings), "settings": voice_settings}
        if not scene.get("image_source_path") and not scene.get("video_source_path"):
            asset_specs[f"image-{scene_id}"] = {
                "source_digest": _scene_image_fingerprint(scene, target_width, target_height), "settings": image_settings}
    manifest = {"plan_sha256": digest_bytes(plan_path.read_bytes()), "local_sources": local_sources,
                "force_images": force_images, "force_audio": force_audio, "assets": asset_specs}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ReceiptConflict("Attempt intent differs from saved sources or effective settings")
    elif recovery:
        raise UnknownDispatch("Attempt has no recovery manifest")
    else:
        atomic_json(manifest_path, manifest)
    if not recovery:
        # Commit every expected asset's exact intent before the first paid call.
        for asset_id, spec in asset_specs.items():
            prepared = operation_dir / attempt_id / asset_id / "prepared.json"
            if prepared.exists():
                if json.loads(prepared.read_text()) != spec:
                    raise ReceiptConflict(f"Incompatible prepared asset: {asset_id}")
            else:
                atomic_json(prepared, spec)

    failures = {}
    def generate_one(asset_id, canonical, digest, settings, generate, *, force):
        try:
            # A durable current-attempt output can repair an interrupted canonical
            # promotion even while its old sidecars disagree with the new bytes.
            manifest = operation_dir / attempt_id / asset_id / "asset.json"
            completed = manifest.exists() and json.loads(manifest.read_text()).get("output_sha256")
            if completed or force or not _asset_current(canonical, digest, settings):
                _generate_asset(operation_dir, attempt_id, asset_id, canonical, digest,
                                settings, generate, recovery=recovery)
        except Exception as exc:
            failures[asset_id] = exc

    for idx, scene in enumerate(scenes):
        apply_render_dimensions(scene, target_width, target_height)
        scene_id = int(scene.get("id", idx))
        source_path = (
            resolve_plan_path(project_dir, str(scene["image_source_path"]))
            if scene.get("image_source_path")
            else None
        )
        video_source_path = (
            resolve_plan_path(project_dir, str(scene["video_source_path"]))
            if scene.get("video_source_path")
            else None
        )

        # Canonicalize generated artifact paths on every render. This keeps the
        # project stable even after manual scene insert/delete work in plan.json.
        if video_source_path is not None:
            if not video_source_path.exists():
                raise FileNotFoundError(f"Missing video_source_path asset: {video_source_path}")
            image_path = None
            scene["video_source_path"] = plan_path_string(project_dir, video_source_path)
        elif source_path is not None:
            if not source_path.exists():
                raise FileNotFoundError(f"Missing image_source_path asset: {source_path}")
            image_path = source_path.resolve()
        else:
            image_path = project_dir / "images" / f"scene_{scene_id:03d}.png"
        audio_path = project_dir / "audio" / f"scene_{scene_id:03d}.wav"
        scene["image_path"] = plan_path_string(project_dir, image_path)
        scene["audio_path"] = plan_path_string(project_dir, audio_path)

        image_digest = _scene_image_fingerprint(scene, target_width, target_height)
        if (
            video_source_path is None
            and source_path is None
            and image_path is not None
        ):
            generate_one(f"image-{scene_id}", image_path,
                            image_digest, image_settings,
                            lambda staging: generate_scene_image(
                                dict(scene), staging, model="gpt-image-2", target_width=target_width,
                                target_height=target_height,
                                orientation="portrait" if target_height > target_width else "landscape"),
                            force=force_images)
        audio_digest = _scene_audio_fingerprint(scene, voice_settings)
        generate_one(f"audio-{scene_id}", audio_path,
                            audio_digest, voice_settings,
                            lambda staging: generate_scene_audio(
                                dict(scene),
                                staging,
                                tts_provider=tts_provider,
                                voice=voice,
                                speed=audio_speed,
                                openai_model=tts_model,
                                openai_instructions=voice_instructions,
                                openrouter_model=tts_model,
                                elevenlabs_model_id=elevenlabs_model_id,
                                elevenlabs_apply_text_normalization=elevenlabs_text_normalization,
                                elevenlabs_stability=elevenlabs_stability,
                                elevenlabs_similarity_boost=elevenlabs_similarity_boost,
                                elevenlabs_style=elevenlabs_style,
                                elevenlabs_use_speaker_boost=elevenlabs_use_speaker_boost,
                            ), force=force_audio)

    if failures:
        raise AssetFailures(failures)

    output_name = f".{project_dir.name}.{uuid4().hex}.mp4"
    if any(scene.get("video_source_path") for scene in scenes):
        video_path = assemble_mixed_video(scenes, project_dir, output_filename=output_name)
    else:
        video_path = assemble_video(
            scenes,
            project_dir,
            output_filename=output_name,
            fps=24,
            target_width=target_width,
            target_height=target_height,
        )
    expected_output = project_dir / output_name
    if Path(video_path).resolve() != expected_output.resolve() or not expected_output.is_file():
        raise ReceiptConflict("Assembly did not produce the exact new MP4")
    _validate_asset(expected_output)
    video_path = project_dir / f"{project_dir.name}.mp4"
    if video_path.exists():
        previous = video_path.read_bytes()
        archive = project_dir / ".v1-videos" / f"{project_dir.name}.{digest_bytes(previous)}.mp4"
        if not archive.exists():
            atomic_bytes(archive, previous)
    os.replace(expected_output, video_path)
    plan.setdefault("meta", {})["video_path"] = plan_path_string(project_dir, video_path)
    plan["meta"]["rendered_utc"] = utc_now()
    plan["meta"]["actual_duration_seconds"] = round(ffprobe_duration(video_path), 2)
    atomic_json(project_dir / "rendered-plan.json", plan)
    return video_path


def summarize(project_dir: Path) -> None:
    plan = load_plan(project_dir / "plan.json")
    print(json.dumps(plan.get("meta", {}), indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown-driven still video maker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("project_dir")
    p_plan.add_argument("--force", action="store_true")

    p_brief = sub.add_parser("brief")
    p_brief.add_argument("project_dir")
    p_brief.add_argument("--force", action="store_true")

    p_render = sub.add_parser("render")
    p_render.add_argument("project_dir")
    p_render.add_argument("--force-images", action="store_true")
    p_render.add_argument("--force-audio", action="store_true")

    p_make = sub.add_parser("make")
    p_make.add_argument("project_dir")
    p_make.add_argument("--force-brief", action="store_true")
    p_make.add_argument("--force-plan", action="store_true")
    p_make.add_argument("--force-images", action="store_true")
    p_make.add_argument("--force-audio", action="store_true")

    p_summary = sub.add_parser("summary")
    p_summary.add_argument("project_dir")

    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()

    if args.cmd == "brief":
        path = create_brief(project_dir, force=args.force)
        print(path)
        return 0
    if args.cmd == "plan":
        path = create_plan(project_dir, force=args.force)
        print(path)
        return 0
    if args.cmd == "render":
        path = render_project(
            project_dir,
            force_images=args.force_images,
            force_audio=args.force_audio,
        )
        print(path)
        return 0
    if args.cmd == "make":
        create_brief(project_dir, force=args.force_brief)
        create_plan(project_dir, force=args.force_plan)
        path = render_project(
            project_dir,
            force_images=args.force_images,
            force_audio=args.force_audio,
        )
        print(path)
        return 0
    if args.cmd == "summary":
        summarize(project_dir)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
