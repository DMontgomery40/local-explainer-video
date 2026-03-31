"""Remotion pipeline: two-pass narration-synced scene generation.

Pass 1: Generate narration + scene structure (what data to show per scene)
Pass 2: After TTS + Whisper timestamps, generate scene_code with cue point timing

Usage:
  python3.10 -m core.pipeline_remotion projects/<patient>/ --input-text <path_to_report>
  python3.10 -m core.pipeline_remotion projects/<patient>/  # uses existing plan
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FPS = 30


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _phase(msg: str) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'=' * 60}", flush=True)


def run_pipeline(
    project_dir: Path,
    *,
    input_text: str | None = None,
    tts_provider: str = "elevenlabs_replicate",
    voice: str = "Antoni",
    speed: float = 1.15,
    skip_tts: bool = False,
    skip_whisper: bool = False,
) -> Path:
    """Run the full two-pass Remotion pipeline."""
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    project_dir = Path(project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    plan_path = project_dir / "plan.json"

    # ── Pass 1: Generate narration + scene structure ──
    if input_text:
        _phase("Pass 1: Generate storyboard (narration + scene data)")
        from .director import generate_storyboard_api
        scenes = generate_storyboard_api(input_text)
        plan = {"meta": {"render_backend": "remotion_dynamic"}, "scenes": scenes}
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        _log(f"Generated {len(scenes)} scenes")
    else:
        plan = json.loads(plan_path.read_text())
        scenes = plan.get("scenes", [])
        _log(f"Loaded {len(scenes)} scenes from existing plan.json")

    # ── TTS ──
    _phase("TTS: Generate audio for all scenes")
    from .voice_gen import generate_audio

    audio_dir = project_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    if not skip_tts:
        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "").strip()
            if not narration:
                continue
            audio_path = audio_dir / f"scene_{i:03d}.wav"
            if audio_path.exists() and audio_path.stat().st_size > 0:
                _log(f"[{i:2d}] exists")
                scene["audio_path"] = str(audio_path)
                continue
            _log(f"[{i:2d}] generating ({len(narration.split())}w)...")
            generate_audio(text=narration, output_path=audio_path,
                           tts_provider=tts_provider, voice=voice, speed=speed)
            scene["audio_path"] = str(audio_path)
    else:
        for i, scene in enumerate(scenes):
            p = audio_dir / f"scene_{i:03d}.wav"
            if p.exists():
                scene["audio_path"] = str(p)

    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))

    # ── Whisper timestamps + cue points ──
    _phase("Whisper: Extract word timestamps and cue points")
    from .whisper_timestamps import get_word_timestamps
    from .cue_points import extract_cue_points

    if not skip_whisper:
        for i, scene in enumerate(scenes):
            audio_path = scene.get("audio_path")
            if not audio_path or not Path(audio_path).exists():
                continue
            _log(f"[{i:2d}] whisper...")
            result = get_word_timestamps(Path(audio_path))
            scene["audio_duration"] = result.duration
            cues = extract_cue_points(result, scene)
            scene["cue_points"] = [asdict(c) for c in cues]
            _log(f"[{i:2d}] {result.duration:.1f}s, {len(cues)} cue points")
    else:
        _log("Skipping Whisper (--skip-whisper)")

    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))

    # ── Pass 2: Generate scene_code with timing data ──
    _phase("Pass 2: Generate scene visuals with cue point timing")

    has_scene_code = any(s.get("scene_code", "").strip() for s in scenes)
    if has_scene_code:
        _log("Scenes already have scene_code — skipping Pass 2")
    else:
        _generate_scene_code_with_timing(scenes, plan, plan_path)

    # ── Render ──
    _phase("Render: Remotion dynamic scenes")
    from .remotion_bridge import render_dynamic_scene, duration_frames

    clips_dir = project_dir / "remotion_renders"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for i, scene in enumerate(scenes):
        code = scene.get("scene_code", "").strip()
        if not code:
            _log(f"[{i:2d}] SKIP — no scene_code")
            continue
        audio_p = Path(scene["audio_path"]) if scene.get("audio_path") else None
        frames = duration_frames(audio_p)
        clip = clips_dir / f"scene_{i:03d}.mp4"
        _log(f"[{i:2d}] rendering ({frames}fr)...")
        try:
            render_dynamic_scene(scene_code=code, output_path=clip, duration_in_frames=frames)
            scene["clip_path"] = str(clip)
        except Exception as e:
            _log(f"[{i:2d}] FAIL: {e}")

    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))

    # ── Assemble ──
    _phase("Assemble: Mux clips + audio, concatenate")
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

    segments_dir = project_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []

    for i, scene in enumerate(scenes):
        clip = scene.get("clip_path")
        audio = scene.get("audio_path")
        if not clip or not audio or not Path(clip).exists() or not Path(audio).exists():
            continue
        seg = segments_dir / f"seg_{i:03d}.mp4"
        cmd = [ffmpeg, "-y", "-i", clip, "-i", audio,
               "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-c:a", "aac", "-b:a", "128k",
               "-map", "0:v:0", "-map", "1:a:0",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-shortest", str(seg)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            segments.append(seg)
        else:
            _log(f"[{i:2d}] mux fail")

    if not segments:
        _log("No segments to assemble!")
        return plan_path

    concat_file = project_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for s in segments:
            f.write(f"file '{s.resolve()}'\n")

    output = project_dir / f"{project_dir.name}.mp4"
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if r.returncode == 0:
        probe = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(output)],
            capture_output=True, text=True, timeout=15)
        dur = float(probe.stdout.strip())
        mb = output.stat().st_size / (1024 * 1024)
        mins, secs = int(dur // 60), int(dur % 60)
        _phase(f"DONE: {output.name}")
        _log(f"Duration: {mins}:{secs:02d}")
        _log(f"Size: {mb:.1f} MB")
        _log(f"Scenes: {len(segments)}/{len(scenes)}")
        return output
    else:
        _log(f"Assembly failed: {r.stderr[-200:]}")
        return plan_path


def _generate_scene_code_with_timing(
    scenes: list[dict[str, Any]],
    plan: dict[str, Any],
    plan_path: Path,
) -> None:
    """Second-pass: generate scene_code for each scene with cue point timing."""
    import anthropic

    client = anthropic.Anthropic()
    system = _build_scene_code_prompt()

    for i, scene in enumerate(scenes):
        cue_points = scene.get("cue_points", [])
        audio_duration = scene.get("audio_duration", 5.0)
        total_frames = int(audio_duration * FPS)

        user_msg = json.dumps({
            "scene_id": i,
            "title": scene.get("title", ""),
            "narration": scene.get("narration", ""),
            "scene_data": scene.get("scene_data", scene.get("composition", {}).get("props", {})),
            "audio_duration_seconds": audio_duration,
            "total_frames": total_frames,
            "fps": FPS,
            "cue_points": cue_points,
        }, indent=2, ensure_ascii=False)

        _log(f"[{i:2d}] generating scene_code ({len(cue_points)} cues, {total_frames}fr)...")

        text = ""
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for event in stream:
                if hasattr(event, "type") and event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        text += event.delta.text

        code = _extract_code(text)
        if code:
            scene["scene_code"] = code
            _log(f"[{i:2d}] OK ({len(code)} chars)")
        else:
            _log(f"[{i:2d}] WARN: no code extracted")

    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))


def _extract_code(text: str) -> str:
    """Extract scene_code from model output — handles raw code or JSON wrapper."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "scene_code" in parsed:
                return parsed["scene_code"]
        except json.JSONDecodeError:
            pass

    if "useCurrentFrame" in text or "AbsoluteFill" in text or "return (" in text:
        return text

    return ""


def _build_scene_code_prompt() -> str:
    """System prompt for Pass 2: generate one scene's Remotion code with cue timing."""
    return """You generate Remotion React component code for a single scene of a patient explainer video.

You receive a JSON object with:
- title, narration: what this scene is about
- scene_data: the metrics, values, labels to display
- audio_duration_seconds, total_frames, fps: timing info
- cue_points: array of {word, time, cue_type, value} — EXACT moments specific words are spoken

YOUR JOB: Write the component BODY (no imports, no export, no function wrapper).

ANIMATION TIMING — SCENE TYPE MATTERS:

Title cards / roadmaps / closing scenes:
  - Animate in during the first 1-3 seconds, then HOLD for the rest of the scene.
  - No cue-point sync needed. The narrator talks over a beautiful static scene.
  - Don't keep things moving the whole time — reveal once, hold.

Data scenes WITH cue points:
  - Background/layout animates in during the first 1-2 seconds.
  - Data values, metrics, chart elements reveal when the narrator mentions them.
  - Convert cue_point.time to frames: Math.round(time * fps)
  - Use spring() with delay = cue frame. Element appears, animates for ~1s, then holds.
  - If there are NO cue_points, just do a staggered reveal over the first 3-4 seconds.

CRITICAL: Once an element has fully animated in, it HOLDS in place for the rest of the scene. Nothing should keep moving, bouncing, or pulsing continuously. A particle background is fine. But data values, charts, text — they animate in and stay put. The scene should look beautiful as a screenshot at any point after the reveals complete.

AVAILABLE APIS (already in scope — NO imports):
  React, useState, useEffect, useMemo, useRef, useCallback
  AbsoluteFill, Sequence, Img, interpolate, spring, staticFile
  useCurrentFrame, useVideoConfig, Easing

QUALITY BAR:
- Ring charts, arc gauges, bar charts, line graphs, SVG paths — real data viz
- Color-coded: red=#ef4444 (concerning), amber=#fbbf24 (transitional), teal=#4fd1c5 (improved)
- Dark premium gradient backgrounds: linear-gradient(160deg, #0a0e1a, #060a14)
- Spring physics: {damping: 200} for smooth, {damping: 20, stiffness: 200} for snappy
- Fill the 1664x928 canvas purposefully — no massive dead space
- Font: 'Inter', 'Helvetica Neue', sans-serif
- Text shadows for readability over backgrounds
- Each scene visually unique

Return ONLY the component body code. No explanation. No markdown fences. No imports.
Start with: const frame = useCurrentFrame();
End with: return (...);"""


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Two-pass Remotion pipeline")
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--input-text", type=Path, default=None)
    parser.add_argument("--voice", default="Antoni")
    parser.add_argument("--tts-provider", default="elevenlabs_replicate")
    parser.add_argument("--speed", type=float, default=1.15)
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-whisper", action="store_true")
    args = parser.parse_args()

    text = None
    if args.input_text and args.input_text.exists():
        text = args.input_text.read_text()[:6000]

    run_pipeline(
        args.project_dir,
        input_text=text,
        tts_provider=args.tts_provider,
        voice=args.voice,
        speed=args.speed,
        skip_tts=args.skip_tts,
        skip_whisper=args.skip_whisper,
    )
