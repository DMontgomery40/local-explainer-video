"""V2 Pipeline: Audio-first Remotion render with narration-synced timing.

Code may validate and execute. Code may NOT decide what scene to make.

Usage:
  python3.10 pipeline_v2.py projects/<patient>/
  python3.10 pipeline_v2.py projects/<patient>/ --skip-qc
  python3.10 pipeline_v2.py projects/<patient>/ --skip-tts --skip-render

Pipeline stages (code-only, no creative decisions):
  1. Load plan.json + scene_claims.json
  2. Validate narration claims against _data_pack.json
  3. Generate TTS (ElevenLabs via Replicate)
  4. Get word timestamps (Whisper via Replicate)
  5. Extract cue points
  6. Write scene_timing.json
  7. Validate scene claims (if scene_claims.json exists)
  8. Render scenes via Remotion (composition + props from plan.json)
  9. Assemble final video (h264_videotoolbox on Mac)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")


def _log(msg: str) -> None:
    print(f"  {msg}")


def _phase(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _load_plan(project_dir: Path) -> dict:
    plan_path = project_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"plan.json not found in {project_dir}")
    return json.loads(plan_path.read_text(encoding="utf-8"))


def _save_plan(project_dir: Path, plan: dict) -> None:
    plan_path = project_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")


def _infer_patient_id(project_dir: Path) -> str:
    """Read the clinic patient ID off a project directory name."""
    from core.qc_publish import infer_patient_id

    patient_id = infer_patient_id(project_dir.name)
    if patient_id is None:
        raise ValueError(
            f"Cannot read a clinic patient ID from directory name: {project_dir.name}"
        )
    return patient_id


def run_pipeline(
    project_dir: Path,
    *,
    tts_provider: str = "elevenlabs_replicate",
    voice: str = "Antoni",
    speed: float = 1.15,
    skip_qc: bool = False,
    skip_tts: bool = False,
    skip_record: bool = False,
    fps: int = 30,
    qeeg_dir: Path | None = None,
) -> Path:
    """Run the full v2 pipeline."""

    project_dir = Path(project_dir).resolve()
    plan = _load_plan(project_dir)
    scenes = plan.get("scenes", [])
    _log(f"Project: {project_dir.name}")
    _log(f"Scenes: {len(scenes)}")

    # Resolve qEEG analysis directory
    if qeeg_dir is None:
        qeeg_dir = Path(os.getenv("QEEG_ANALYSIS_DIR", str(REPO_ROOT.parent / "qEEG-analysis")))

    # --- Stage 1: Validate narration claims ---
    if not skip_qc:
        _phase("Stage 1: Validate narration claims against data pack")
        claims_path = project_dir / "scene_claims.json"
        if claims_path.exists():
            from core.qc_claims import validate_claims_file, write_claims_report
            from core.qc_publish import load_qeeg_ground_truth

            patient_id = _infer_patient_id(project_dir)
            gt = load_qeeg_ground_truth(qeeg_dir=qeeg_dir, patient_label=patient_id)
            result = validate_claims_file(claims_path, gt.data_pack)
            report_path = project_dir / "qc_claims_report.json"
            write_claims_report(result, report_path)

            _log(f"Claims: {len(result.results)} checked, {len(result.errors)} failed")
            if not result.passed:
                for err in result.errors:
                    _log(f"  FAIL: {err}")
                raise RuntimeError(
                    f"Narration claims QC failed ({len(result.errors)} errors). "
                    f"Fix scene_claims.json or the narration, then re-run."
                )
            _log("All narration claims passed.")
        else:
            _log(f"No scene_claims.json found — skipping narration claims QC")

    # --- Stage 2: Generate TTS ---
    _phase("Stage 2: Generate TTS audio")
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    if not skip_tts:
        from core.voice_gen import generate_audio

        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "")
            if not narration:
                continue
            audio_path = audio_dir / f"scene_{i:03d}.wav"
            _log(f"[tts] scene_{i:03d} ({len(narration.split())} words)...")
            generate_audio(
                text=narration,
                output_path=audio_path,
                tts_provider=tts_provider,
                voice=voice,
                speed=speed,
            )
            _log(f"[done] {audio_path.name}")
            scene["audio_path"] = str(audio_path)
        _save_plan(project_dir, plan)
    else:
        _log("Skipping TTS (--skip-tts)")
        # Ensure audio_path is set for existing files
        for i, scene in enumerate(scenes):
            audio_path = audio_dir / f"scene_{i:03d}.wav"
            if audio_path.exists():
                scene["audio_path"] = str(audio_path)

    # --- Stage 3: Whisper timestamps ---
    _phase("Stage 3: Get word timestamps (Whisper)")
    from core.whisper_timestamps import get_word_timestamps

    timing_data: list[dict] = []
    for i, scene in enumerate(scenes):
        audio_path = Path(str(scene.get("audio_path", "")))
        if not audio_path.exists():
            _log(f"[skip] scene_{i:03d} — no audio")
            timing_data.append({"id": i, "title": scene.get("title", ""), "audio_duration": 0, "cue_points": []})
            continue

        _log(f"[whisper] scene_{i:03d}...")
        whisper_result = get_word_timestamps(audio_path)
        scene["audio_duration"] = whisper_result.duration

        # --- Stage 4: Extract cue points ---
        from core.cue_points import extract_cue_points, cue_points_to_json
        cues = extract_cue_points(whisper_result, scene)
        scene["cue_points"] = [asdict(c) for c in cues]
        _log(f"  duration={whisper_result.duration:.1f}s, cues={len(cues)}")

        timing_data.append({
            "id": i,
            "title": scene.get("title", ""),
            "audio_duration": whisper_result.duration,
            "cue_points": [asdict(c) for c in cues],
        })

    # Write scene_timing.json (code → agent bridge)
    timing_path = project_dir / "scene_timing.json"
    timing_path.write_text(json.dumps({"scenes": timing_data}, indent=2), encoding="utf-8")
    _log(f"Wrote {timing_path.name}")
    _save_plan(project_dir, plan)

    # --- Stage 5: Validate HTML scene claims (if present) ---
    if not skip_qc:
        html_claims_path = project_dir / "scene_claims_html.json"
        if html_claims_path.exists():
            _phase("Stage 5: Validate HTML scene claims")
            from core.qc_claims import validate_claims_file, write_claims_report
            from core.qc_publish import load_qeeg_ground_truth

            patient_id = _infer_patient_id(project_dir)
            gt = load_qeeg_ground_truth(qeeg_dir=qeeg_dir, patient_label=patient_id)
            result = validate_claims_file(html_claims_path, gt.data_pack)
            report_path = project_dir / "qc_claims_html_report.json"
            write_claims_report(result, report_path)

            if not result.passed:
                for err in result.errors:
                    _log(f"  FAIL: {err}")
                raise RuntimeError(
                    f"HTML scene claims QC failed ({len(result.errors)} errors). "
                    f"Fix the HTML scenes or scene_claims_html.json, then re-run."
                )
            _log("All HTML scene claims passed.")

    # --- Stage 6: Render scenes via Remotion ---
    if not skip_record:
        _phase("Stage 6: Render scenes via Remotion")
        from core.remotion_bridge import render_scene, duration_frames, VALID_FAMILIES

        clips_dir = project_dir / "remotion_renders"
        clips_dir.mkdir(parents=True, exist_ok=True)

        failures = {}
        for i, scene in enumerate(scenes):
            composition = scene.get("composition", {})
            family = composition.get("family", "narration_slide")
            props = composition.get("props", {})

            if family not in VALID_FAMILIES:
                _log(f"[skip] scene_{i:03d} — unknown family {family!r}, falling back to narration_slide")
                family = "narration_slide"
                props = {"headline": scene.get("title", ""), "body": ""}

            if not props:
                props = {"headline": scene.get("title", ""), "body": ""}

            audio_path = scene.get("audio_path")
            audio_p = Path(str(audio_path)) if audio_path else None
            frames = duration_frames(audio_p)

            clip_path = clips_dir / f"scene_{i:03d}.mp4"

            _log(f"[render] scene_{i:03d} ({family}, {frames} frames)...")
            try:
                render_scene(
                    family=family,
                    props=props,
                    output_path=clip_path,
                    duration_in_frames=frames,
                )
                scene["clip_path"] = str(clip_path)
                _log(f"[done] {clip_path.name}")
            except Exception as exc:
                scene.pop("clip_path", None)
                failures[str(i)] = exc
                _log(f"[FAIL] scene_{i:03d}: {exc}")
            _save_plan(project_dir, plan)

        if failures:
            from core.generation_receipts import AssetFailures
            raise AssetFailures(failures)
    else:
        _log("Skipping rendering (--skip-render)")

    # --- Stage 7: Assemble ---
    _phase("Stage 7: Assemble final video")
    from core.video_assembly import assemble_v2_video

    # Only assemble scenes that have both clip and audio
    ready_scenes = [
        s for s in scenes
        if s.get("clip_path") and Path(str(s["clip_path"])).exists()
        and s.get("audio_path") and Path(str(s["audio_path"])).exists()
    ]
    _log(f"Ready scenes: {len(ready_scenes)} / {len(scenes)}")

    if len(ready_scenes) != len(scenes) or not scenes:
        raise RuntimeError("Every planned scene needs its current clip and audio before assembly")

    output_path = assemble_v2_video(ready_scenes, project_dir, fps=fps)

    _phase("Done")
    _log(f"Video: {output_path}")
    _log(f"Timing: {timing_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 pipeline: audio-first recording with narration-synced cue points")
    parser.add_argument("project_dir", type=Path, help="Project directory with plan.json")
    parser.add_argument("--voice", default="Antoni", help="ElevenLabs voice name")
    parser.add_argument("--tts-provider", default="elevenlabs_replicate",
                        help="TTS provider (elevenlabs_replicate, elevenlabs, kokoro, openai)")
    parser.add_argument("--speed", type=float, default=1.15, help="Speech speed")
    parser.add_argument("--skip-qc", action="store_true", help="Skip claims validation")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS (use existing audio)")
    parser.add_argument("--skip-render", action="store_true", help="Skip Remotion rendering (use existing clips)")
    parser.add_argument("--skip-record", action="store_true", dest="skip_render", help="(alias for --skip-render)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--qeeg-dir", type=Path, default=None, help="qEEG analysis directory")
    args = parser.parse_args()

    run_pipeline(
        args.project_dir,
        tts_provider=args.tts_provider,
        voice=args.voice,
        speed=args.speed,
        skip_qc=args.skip_qc,
        skip_tts=args.skip_tts,
        skip_record=args.skip_render,
        fps=args.fps,
        qeeg_dir=args.qeeg_dir,
    )


if __name__ == "__main__":
    main()
