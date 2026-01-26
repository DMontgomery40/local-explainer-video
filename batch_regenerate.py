#!/usr/bin/env python3
"""
Batch regenerate videos for all valid patient ID projects.

Usage:
    python3.10 batch_regenerate.py [--dry-run] [--projects PROJECT1,PROJECT2,...]

Valid patient ID format: MM-DD-YYYY-N (e.g., 01-01-1991-0)
"""

import argparse
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from core.director import generate_storyboard
from core.image_gen import generate_scene_image
from core.voice_gen import (
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_ELEVENLABS_SIMILARITY_BOOST,
    DEFAULT_ELEVENLABS_SPEED,
    DEFAULT_ELEVENLABS_STABILITY,
    DEFAULT_ELEVENLABS_STYLE,
    DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
    DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST,
    DEFAULT_ELEVENLABS_VOICE,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    ElevenLabsTextNormalization,
    generate_scene_audio,
)
from core.video_assembly import assemble_video

PROJECTS_DIR = Path(__file__).parent / "projects"
ARGS: dict = {}

# Pattern for valid patient IDs: MM-DD-YYYY-N
PATIENT_ID_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}-\d+$")


def get_valid_patient_projects() -> list[Path]:
    """Find all projects with valid patient ID names that have plan.json."""
    projects = []
    for p in PROJECTS_DIR.iterdir():
        if p.is_dir() and PATIENT_ID_PATTERN.match(p.name):
            if (p / "plan.json").exists():
                projects.append(p)
    return sorted(projects)


def load_plan(project_dir: Path) -> dict | None:
    """Load plan.json from a project directory."""
    plan_path = project_dir / "plan.json"
    if plan_path.exists():
        try:
            return json.loads(plan_path.read_text())
        except json.JSONDecodeError as e:
            print(f"ERROR: Corrupted plan.json in {project_dir.name}: {e}")
            return None
    return None


def save_plan(project_dir: Path, plan: dict) -> None:
    """Save plan.json to a project directory."""
    project_dir.mkdir(parents=True, exist_ok=True)
    plan_path = project_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))


def regenerate_project(project_dir: Path, dry_run: bool = False) -> bool:
    """
    Regenerate all assets for a project.

    Steps:
    1. Load existing plan.json to get input_text
    2. Generate new storyboard with updated prompts
    3. Generate all images
    4. Generate all audio
    5. Assemble final video

    Returns True on success, False on failure.
    """
    project_name = project_dir.name
    print(f"\n{'='*60}")
    print(f"Processing: {project_name}")
    print(f"{'='*60}")

    # Load existing plan
    old_plan = load_plan(project_dir)
    if not old_plan:
        print(f"ERROR: Could not load plan.json for {project_name}")
        return False

    # Get input text from old plan
    input_text = old_plan.get("meta", {}).get("input_text")
    if not input_text:
        print(f"ERROR: No input_text found in {project_name}/plan.json")
        return False

    provider = old_plan.get("meta", {}).get("llm_provider", "anthropic")
    meta = old_plan.get("meta", {}) if isinstance(old_plan.get("meta"), dict) else {}

    # TTS settings: prefer CLI args (if provided), else prefer plan meta, else defaults.
    tts_provider = (ARGS.get("tts_provider") or meta.get("tts_provider") or "kokoro").strip()
    if tts_provider not in {"kokoro", "elevenlabs", "openai"}:
        print(f"  WARNING: Unknown tts_provider={tts_provider!r}; falling back to 'kokoro'")
        tts_provider = "kokoro"

    if tts_provider == "elevenlabs":
        tts_voice = (ARGS.get("tts_voice") or meta.get("tts_voice") or DEFAULT_ELEVENLABS_VOICE).strip()
        tts_speed = float(ARGS.get("tts_speed") or meta.get("tts_speed") or float(DEFAULT_ELEVENLABS_SPEED))
        elevenlabs_model_id = (ARGS.get("elevenlabs_model_id") or meta.get("elevenlabs_model_id") or DEFAULT_ELEVENLABS_MODEL).strip()
        elevenlabs_text_norm: ElevenLabsTextNormalization = str(
            ARGS.get("elevenlabs_text_normalization")
            or meta.get("elevenlabs_apply_text_normalization")
            or DEFAULT_ELEVENLABS_TEXT_NORMALIZATION
        )  # type: ignore[assignment]
        if elevenlabs_text_norm not in {"auto", "on", "off"}:
            elevenlabs_text_norm = DEFAULT_ELEVENLABS_TEXT_NORMALIZATION
        elevenlabs_stability = float(ARGS.get("elevenlabs_stability") or meta.get("elevenlabs_stability") or DEFAULT_ELEVENLABS_STABILITY)
        elevenlabs_similarity = float(
            ARGS.get("elevenlabs_similarity_boost")
            or meta.get("elevenlabs_similarity_boost")
            or DEFAULT_ELEVENLABS_SIMILARITY_BOOST
        )
        elevenlabs_style = float(ARGS.get("elevenlabs_style") or meta.get("elevenlabs_style") or DEFAULT_ELEVENLABS_STYLE)
        elevenlabs_use_speaker_boost = bool(
            ARGS.get("elevenlabs_use_speaker_boost")
            if ARGS.get("elevenlabs_use_speaker_boost") is not None
            else meta.get("elevenlabs_use_speaker_boost", DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST)
        )
    else:
        # Kokoro/OpenAI defaults (voice/speed are only meaningful for Kokoro here)
        tts_voice = (ARGS.get("tts_voice") or meta.get("tts_voice") or DEFAULT_VOICE).strip()
        tts_speed = float(ARGS.get("tts_speed") or meta.get("tts_speed") or DEFAULT_SPEED)
        elevenlabs_model_id = DEFAULT_ELEVENLABS_MODEL
        elevenlabs_text_norm = DEFAULT_ELEVENLABS_TEXT_NORMALIZATION
        elevenlabs_stability = DEFAULT_ELEVENLABS_STABILITY
        elevenlabs_similarity = DEFAULT_ELEVENLABS_SIMILARITY_BOOST
        elevenlabs_style = DEFAULT_ELEVENLABS_STYLE
        elevenlabs_use_speaker_boost = DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST

    if dry_run:
        print(f"  [DRY RUN] Would regenerate with provider: {provider}")
        print(f"  [DRY RUN] Input text length: {len(input_text)} chars")
        print(f"  [DRY RUN] TTS: provider={tts_provider} voice={tts_voice} speed={tts_speed}")
        return True

    # Step 1: Generate new storyboard
    print(f"\n[1/4] Generating storyboard...")
    try:
        scenes = generate_storyboard(input_text, provider=provider)
        print(f"  Generated {len(scenes)} scenes")
    except Exception as e:
        print(f"ERROR: Storyboard generation failed: {e}")
        return False

    # Add UIDs to scenes
    for scene in scenes:
        scene["uid"] = str(uuid.uuid4())[:8]

    # Create new plan
    new_plan = {
        "meta": {
            "project_name": project_name,
            "created_utc": datetime.utcnow().isoformat(),
            "regenerated_utc": datetime.utcnow().isoformat(),
            "llm_provider": provider,
            "image_model": "qwen/qwen-image-2512",
            "input_text": input_text,
            "tts_provider": tts_provider,
            "tts_voice": tts_voice,
            "tts_speed": tts_speed,
            "elevenlabs_model_id": elevenlabs_model_id,
            "elevenlabs_apply_text_normalization": elevenlabs_text_norm,
            "elevenlabs_stability": elevenlabs_stability,
            "elevenlabs_similarity_boost": elevenlabs_similarity,
            "elevenlabs_style": elevenlabs_style,
            "elevenlabs_use_speaker_boost": elevenlabs_use_speaker_boost,
        },
        "scenes": scenes,
    }

    # Save plan
    save_plan(project_dir, new_plan)
    print(f"  Saved new plan.json")

    # Step 2: Generate images
    print(f"\n[2/4] Generating images...")
    for i, scene in enumerate(scenes):
        print(f"  Scene {i+1}/{len(scenes)}: {scene.get('title', 'Untitled')[:40]}...")
        try:
            path = generate_scene_image(scene, project_dir)
            scene["image_path"] = str(path)
            save_plan(project_dir, new_plan)
        except Exception as e:
            print(f"  ERROR generating image for scene {i}: {e}")
            # Continue with other scenes

    # Step 3: Generate audio
    print(f"\n[3/4] Generating audio...")
    for i, scene in enumerate(scenes):
        print(f"  Scene {i+1}/{len(scenes)}...")
        try:
            path = generate_scene_audio(
                scene, project_dir,
                tts_provider=tts_provider,  # type: ignore[arg-type]
                voice=tts_voice,
                speed=tts_speed,
                elevenlabs_model_id=elevenlabs_model_id,
                elevenlabs_apply_text_normalization=elevenlabs_text_norm,
                elevenlabs_stability=elevenlabs_stability,
                elevenlabs_similarity_boost=elevenlabs_similarity,
                elevenlabs_style=elevenlabs_style,
                elevenlabs_use_speaker_boost=elevenlabs_use_speaker_boost,
            )
            scene["audio_path"] = str(path)
            save_plan(project_dir, new_plan)
        except Exception as e:
            print(f"  ERROR generating audio for scene {i}: {e}")
            # Continue with other scenes

    # Step 4: Assemble video
    print(f"\n[4/4] Assembling video...")
    try:
        output_filename = f"{project_name}.mp4"
        video_path = assemble_video(scenes, project_dir, output_filename=output_filename)
        new_plan["meta"]["video_path"] = str(video_path)
        save_plan(project_dir, new_plan)
        print(f"  Video saved: {video_path}")
    except Exception as e:
        print(f"ERROR: Video assembly failed: {e}")
        return False

    print(f"\n✓ Successfully regenerated {project_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch regenerate patient videos")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--projects", type=str, help="Comma-separated list of specific projects to process")
    parser.add_argument("--tts-provider", type=str, default="", help="TTS provider: kokoro, elevenlabs, openai (default: from plan meta or kokoro)")
    parser.add_argument("--tts-voice", type=str, default="", help="TTS voice (Kokoro voice id or ElevenLabs voice name)")
    parser.add_argument("--tts-speed", type=float, default=0.0, help="TTS speed (Kokoro or ElevenLabs). 0 means default/from meta.")
    parser.add_argument("--elevenlabs-model-id", type=str, default="", help="ElevenLabs model_id (default eleven_flash_v2_5)")
    parser.add_argument(
        "--elevenlabs-text-normalization",
        type=str,
        default="",
        help='ElevenLabs apply_text_normalization: "auto"|"on"|"off" (default auto)',
    )
    parser.add_argument("--elevenlabs-stability", type=float, default=0.0, help="ElevenLabs stability 0..1 (0 means default/from meta)")
    parser.add_argument("--elevenlabs-similarity-boost", type=float, default=0.0, help="ElevenLabs similarity_boost 0..1 (0 means default/from meta)")
    parser.add_argument("--elevenlabs-style", type=float, default=0.0, help="ElevenLabs style 0..1 (0 means default/from meta)")
    parser.add_argument(
        "--elevenlabs-use-speaker-boost",
        action="store_true",
        help="Enable ElevenLabs speaker boost (if omitted, use plan meta or default).",
    )
    args = parser.parse_args()

    # Normalize CLI args into a simple dict for use inside regenerate_project()
    global ARGS
    ARGS = {
        "tts_provider": (args.tts_provider or "").strip(),
        "tts_voice": (args.tts_voice or "").strip(),
        "tts_speed": float(args.tts_speed) if args.tts_speed else 0.0,
        "elevenlabs_model_id": (args.elevenlabs_model_id or "").strip(),
        "elevenlabs_text_normalization": (args.elevenlabs_text_normalization or "").strip(),
        "elevenlabs_stability": float(args.elevenlabs_stability) if args.elevenlabs_stability else 0.0,
        "elevenlabs_similarity_boost": float(args.elevenlabs_similarity_boost) if args.elevenlabs_similarity_boost else 0.0,
        "elevenlabs_style": float(args.elevenlabs_style) if args.elevenlabs_style else 0.0,
        # Note: presence of the flag is meaningful; absence means "use meta/default"
        "elevenlabs_use_speaker_boost": True if args.elevenlabs_use_speaker_boost else None,
    }

    if args.projects:
        # Process specific projects
        project_names = [p.strip() for p in args.projects.split(",")]
        projects = [PROJECTS_DIR / name for name in project_names if (PROJECTS_DIR / name).exists()]
    else:
        # Find all valid patient ID projects
        projects = get_valid_patient_projects()

    if not projects:
        print("No valid patient ID projects found.")
        print(f"Looking in: {PROJECTS_DIR}")
        print("Valid format: MM-DD-YYYY-N (e.g., 01-01-1991-0)")
        sys.exit(1)

    print(f"Found {len(projects)} project(s) to process:")
    for p in projects:
        print(f"  - {p.name}")

    if args.dry_run:
        print("\n[DRY RUN MODE - no changes will be made]")

    # Process each project
    successes = 0
    failures = 0

    for project_dir in projects:
        try:
            if regenerate_project(project_dir, dry_run=args.dry_run):
                successes += 1
            else:
                failures += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nERROR processing {project_dir.name}: {e}")
            failures += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Successes: {successes}")
    print(f"Failures:  {failures}")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
