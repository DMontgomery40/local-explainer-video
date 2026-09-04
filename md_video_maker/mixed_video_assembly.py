"""Assemble narrated scenes that mix still images and short source videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

TARGET_WIDTH = 1664
TARGET_HEIGHT = 928
TARGET_AUDIO_SAMPLE_RATE = 48_000
TARGET_AUDIO_CHANNELS = 2


def _bin(name: str) -> str:
    candidate = Path("/opt/homebrew/bin") / name
    if candidate.exists():
        return str(candidate)
    return name


FFMPEG = _bin("ffmpeg")
FFPROBE = _bin("ffprobe")


def _resolve_plan_path(project_dir: Path, raw: str | None) -> Path:
    candidate = Path(raw or "").expanduser()
    if not candidate.is_absolute():
        candidate = project_dir / candidate
    return candidate.resolve()


def _plan_path_string(project_dir: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_dir.resolve()))
    except ValueError:
        return str(resolved)


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            FFPROBE,
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


def _archive_existing_video(output_path: Path, project_dir: Path) -> Path | None:
    if not output_path.exists():
        return None

    archive_dir = project_dir / ".v1-videos"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / f"{project_dir.name}{output_path.suffix or '.mp4'}"
    if dest.exists():
        n = 2
        while True:
            candidate = archive_dir / f"{project_dir.name} v{n}{output_path.suffix or '.mp4'}"
            if not candidate.exists():
                dest = candidate
                break
            n += 1

    output_path.replace(dest)
    return dest


def _filter() -> str:
    return (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1,fps=24,format=yuv420p"
    )


def _make_still_segment(
    scene: dict[str, Any], project_dir: Path, audio_path: Path, out_path: Path, seconds: float
) -> None:
    image_path = _resolve_plan_path(
        project_dir,
        str(scene.get("image_path") or scene.get("image_source_path") or ""),
    )
    if not image_path.exists():
        raise FileNotFoundError(f"Missing scene image: {image_path}")

    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            "24",
            "-i",
            str(image_path),
            "-i",
            str(audio_path),
            "-t",
            f"{seconds:.3f}",
            "-vf",
            _filter(),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(TARGET_AUDIO_SAMPLE_RATE),
            "-ac",
            str(TARGET_AUDIO_CHANNELS),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            str(out_path),
        ],
        check=True,
    )


def _make_video_segment(
    scene: dict[str, Any], project_dir: Path, audio_path: Path, out_path: Path, seconds: float
) -> None:
    video_path = _resolve_plan_path(project_dir, str(scene.get("video_source_path") or ""))
    if not video_path.exists():
        raise FileNotFoundError(f"Missing scene video: {video_path}")

    cmd = [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-stream_loop",
        "-1",
    ]
    start = scene.get("video_start_seconds")
    if start not in (None, ""):
        cmd.extend(["-ss", f"{float(start):.3f}"])
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-t",
            f"{seconds:.3f}",
            "-vf",
            _filter(),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-analyzeduration",
            "100M",
            "-probesize",
            "100M",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(TARGET_AUDIO_SAMPLE_RATE),
            "-ac",
            str(TARGET_AUDIO_CHANNELS),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            str(out_path),
        ]
    )
    subprocess.run(cmd, check=True)


def assemble_mixed_video(
    scenes: list[dict[str, Any]],
    project_dir: Path,
    output_filename: str = "final_video.mp4",
) -> Path:
    project_dir = Path(project_dir)
    output_path = project_dir / output_filename
    archived = _archive_existing_video(output_path, project_dir)
    if archived:
        print(f"Archived previous video to: {archived}")

    segment_dir = project_dir / "_mixed_segments"
    if segment_dir.exists():
        shutil.rmtree(segment_dir)
    segment_dir.mkdir(parents=True, exist_ok=True)

    segment_paths: list[Path] = []
    try:
        for idx, scene in enumerate(scenes):
            audio_path = _resolve_plan_path(project_dir, str(scene.get("audio_path") or ""))
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing scene audio: {audio_path}")
            seconds = duration(audio_path)
            segment_path = segment_dir / f"segment_{idx:03d}.mp4"
            if scene.get("video_source_path"):
                _make_video_segment(scene, project_dir, audio_path, segment_path, seconds)
            else:
                _make_still_segment(scene, project_dir, audio_path, segment_path, seconds)
            segment_paths.append(segment_path)

        list_path = project_dir / "_mixed_segments.txt"
        with list_path.open("w", encoding="utf-8") as handle:
            for segment_path in segment_paths:
                handle.write(f"file '{segment_path}'\n")

        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                "-use_editlist",
                "0",
                "-movflags",
                "+faststart+negative_cts_offsets",
                str(output_path),
            ],
            check=True,
        )
        list_path.unlink(missing_ok=True)
    finally:
        shutil.rmtree(segment_dir, ignore_errors=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError(f"Video assembly failed: {output_path}")

    return output_path


def update_plan_render_metadata(project_dir: Path, video_path: Path) -> None:
    plan_path = project_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.setdefault("meta", {})["video_path"] = _plan_path_string(project_dir, video_path)
    plan["meta"]["actual_duration_seconds"] = round(duration(video_path), 2)
    plan["meta"]["assembly"] = "mixed stills + source video clips"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
