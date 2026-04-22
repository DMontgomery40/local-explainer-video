#!/usr/bin/env python3
"""Poll an Anthropic storyboard batch, save the finished plan, and optionally render stills."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.director import MESSAGE_BATCHES_BETA, OUTPUT_300K_BETA, _parse_scenes_json
from core.image_gen import DEFAULT_IMAGE_GEN_MODEL, generate_scene_image
from core.scene_modes import plan_has_cathode_motion_scenes
from core.video_assembly import assemble_video
from core.voice_gen import DEFAULT_SPEED, DEFAULT_VOICE, generate_scene_audio

DEFAULT_TTS_PROVIDER = "kokoro"


def _client() -> anthropic.Anthropic:
    load_dotenv(REPO_ROOT / ".env", override=False)
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is required")
    return anthropic.Anthropic(api_key=api_key)


def _fetch_batch(client: anthropic.Anthropic, batch_id: str):
    return client.beta.messages.batches.retrieve(
        batch_id,
        betas=[MESSAGE_BATCHES_BETA, OUTPUT_300K_BETA],
    )


def _extract_scenes_from_result(item: Any) -> list[dict[str, Any]]:
    result_type = getattr(item.result, "type", None)
    if result_type != "succeeded":
        raise RuntimeError(f"Batch result not succeeded: {result_type}")

    message = item.result.message
    text_parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    raw = "".join(text_parts).strip()
    return _parse_scenes_json(raw)


def _normalize_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_scene in enumerate(scenes, start=1):
        scene = dict(raw_scene)
        scene["id"] = int(scene.get("id") or index)
        scene["uid"] = str(scene.get("uid") or uuid.uuid4())[:8]
        normalized.append(scene)
    return normalized


def _default_project_name(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}__api_batch_{ts}"


def _save_plan(
    *,
    batch_id: str,
    project_name: str,
    source_markdown_path: str | None,
    scenes: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    project_dir = REPO_ROOT / "projects" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    if source_markdown_path:
        source_path = Path(source_markdown_path).expanduser()
        if source_path.exists():
            (project_dir / "source_input.md").write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    plan_path = project_dir / "plan.json"
    plan = {
        "meta": {
            "project_name": project_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "llm_provider": "api-batch",
            "anthropic_model": "claude-sonnet-4-6",
            "anthropic_effort": "high",
            "anthropic_thinking_type": "adaptive",
            "batch_id": batch_id,
            "source_markdown_path": source_markdown_path,
        },
        "scenes": _normalize_scenes(scenes),
    }
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan_path, plan


def _render_saved_plan(
    *,
    plan_path: Path,
    plan: dict[str, Any],
    image_model: str,
    tts_provider: str,
    voice: str,
    speed: float,
) -> Path:
    scenes = [scene for scene in (plan.get("scenes") or []) if isinstance(scene, dict)]
    project_dir = plan_path.parent

    if plan_has_cathode_motion_scenes(scenes):
        raise RuntimeError(
            "Saved batch contains Cathode motion scenes. This helper only supports the still-image path; "
            "render motion scenes downstream in Cathode."
        )

    for scene in scenes:
        image_path = scene.get("image_path")
        if not image_path or not Path(str(image_path)).exists():
            generate_scene_image(scene, project_dir, model=image_model)

        audio_path = scene.get("audio_path")
        if not audio_path or not Path(str(audio_path)).exists():
            generate_scene_audio(
                scene,
                project_dir,
                tts_provider=tts_provider,
                voice=voice,
                speed=speed,
            )

    video_filename = f"{project_dir.name}.mp4"
    output_path = assemble_video(
        scenes,
        project_dir,
        output_filename=video_filename,
    )
    meta = plan.setdefault("meta", {})
    meta["video_path"] = str(output_path)
    meta["rendered_utc"] = datetime.now(timezone.utc).isoformat()
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poll an Anthropic storyboard batch and save the result locally")
    parser.add_argument("batch_id", help="Anthropic message batch id")
    parser.add_argument("--project-prefix", required=True, help="Prefix for the local projects/ folder name")
    parser.add_argument("--source-markdown-path", default="", help="Original patient/source markdown path for plan metadata")
    parser.add_argument("--wait", action="store_true", help="Keep polling until the batch reaches ended")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval when --wait is set")
    parser.add_argument(
        "--render",
        action="store_true",
        help="After saving the plan, render still images locally with Codex gpt-image-2 plus narration audio",
    )
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_GEN_MODEL, help="Image model for prompt-bearing stills")
    parser.add_argument("--tts-provider", default=DEFAULT_TTS_PROVIDER, help="TTS provider for optional render")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Narration voice for optional render")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Narration speed for optional render")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    client = _client()

    while True:
        batch = _fetch_batch(client, args.batch_id)
        print(f"status={batch.processing_status}")
        print(f"request_counts={batch.request_counts}")

        if str(batch.processing_status).lower() == "ended":
            results = list(
                client.beta.messages.batches.results(
                    args.batch_id,
                    betas=[MESSAGE_BATCHES_BETA, OUTPUT_300K_BETA],
                )
            )
            if not results:
                raise SystemExit("Batch ended but returned no results")

            succeeded = next(
                (item for item in results if getattr(getattr(item, "result", None), "type", None) == "succeeded"),
                None,
            )
            if succeeded is None:
                raise SystemExit("Batch ended but returned no succeeded results")

            scenes = _extract_scenes_from_result(succeeded)
            project_name = _default_project_name(args.project_prefix)
            plan_path, plan = _save_plan(
                batch_id=args.batch_id,
                project_name=project_name,
                source_markdown_path=args.source_markdown_path or None,
                scenes=scenes,
            )
            print(f"saved_plan_path={plan_path}")
            print(f"scene_count={len(plan['scenes'])}")
            if args.render:
                output_path = _render_saved_plan(
                    plan_path=plan_path,
                    plan=plan,
                    image_model=args.image_model,
                    tts_provider=args.tts_provider,
                    voice=args.voice,
                    speed=args.speed,
                )
                print(f"rendered_output_path={output_path}")
            return 0

        if not args.wait:
            return 0

        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    raise SystemExit(main())
