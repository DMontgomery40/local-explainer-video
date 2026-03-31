"""Video assembly using MoviePy (v1) and ffmpeg (v2)."""

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from moviepy import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)

# Target dimensions for video output (must match image_gen.py)
TARGET_WIDTH = 1664
TARGET_HEIGHT = 928


def _ensure_dimensions(clip: ImageClip, scene_id: int = 0) -> ImageClip:
    """
    Resize clip to target dimensions if mismatched.

    Prevents video assembly failures from dimension inconsistencies
    (e.g., edited images returning different sizes).
    """
    w, h = clip.size
    if w != TARGET_WIDTH or h != TARGET_HEIGHT:
        print(f"  Scene {scene_id}: resizing {w}x{h} -> {TARGET_WIDTH}x{TARGET_HEIGHT}")
        return clip.resized((TARGET_WIDTH, TARGET_HEIGHT))
    return clip


def _archive_existing_video(output_path: Path, project_dir: Path) -> Path | None:
    """
    If output_path exists, move it to project_dir/.v1-videos with versioned naming.

    Example:
      - .v1-videos/<project>.mp4
      - .v1-videos/<project> v2.mp4
      - .v1-videos/<project> v3.mp4
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return None

    archive_dir = Path(project_dir) / ".v1-videos"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the project folder name as the patient/version base.
    base = Path(project_dir).name.split("__")[0]
    ext = output_path.suffix or ".mp4"

    dest = archive_dir / f"{base}{ext}"
    if dest.exists():
        n = 2
        while True:
            candidate = archive_dir / f"{base} v{n}{ext}"
            if not candidate.exists():
                dest = candidate
                break
            n += 1

    output_path.replace(dest)
    return dest


def assemble_video(
    scenes: list[dict],
    project_dir: Path,
    output_filename: str = "final_video.mp4",
    fps: int = 24,
    default_duration: float = 5.0,
) -> Path:
    """
    Assemble scenes into a final video.

    Args:
        scenes: List of scene dictionaries with image_path and audio_path
        project_dir: Project directory containing assets
        output_filename: Name of the output video file
        fps: Frames per second for the output video
        default_duration: Duration for scenes without audio

    Returns:
        Path to the assembled video
    """
    project_dir = Path(project_dir)
    output_path = project_dir / output_filename

    archived_path = _archive_existing_video(output_path, project_dir)
    if archived_path:
        print(f"Archived previous video to: {archived_path}")

    clips = []
    audio_clips = []  # Track for cleanup

    try:
        for i, scene in enumerate(scenes):
            image_path = scene.get("image_path")
            audio_path = scene.get("audio_path")

            # Skip scenes without images
            if not image_path or not Path(image_path).exists():
                print(f"Skipping scene {scene.get('id', i)}: no image")
                continue

            # Create image clip and ensure correct dimensions
            image_clip = ImageClip(str(image_path))
            image_clip = _ensure_dimensions(image_clip, scene.get('id', i))

            # Add audio if available
            if audio_path and Path(audio_path).exists():
                audio_clip = AudioFileClip(str(audio_path))
                audio_clips.append(audio_clip)  # Keep reference for cleanup
                duration = audio_clip.duration
                image_clip = image_clip.with_duration(duration)
                image_clip = image_clip.with_audio(audio_clip)
            else:
                # Use default duration if no audio
                image_clip = image_clip.with_duration(default_duration)

            clips.append(image_clip)

        if not clips:
            raise ValueError("No valid scenes to assemble")

        # Concatenate all clips (hard cuts, no transitions)
        final_video = concatenate_videoclips(clips, method="compose")

        # Write output video using CPU encoder (faster for slideshow content)
        final_video.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",  # CPU encoding - faster for slideshow content
            audio_codec="aac",
            temp_audiofile=str(project_dir / "temp_audio.m4a"),
            remove_temp=True,
            logger="bar",  # Progress bar
            ffmpeg_params=[
                "-preset", "ultrafast",
                "-crf", "35",
                "-pix_fmt", "yuv420p",  # Compatible pixel format
                "-movflags", "+faststart",  # Web-friendly streaming
            ],
        )

    finally:
        # Clean up ALL clips after encoding
        for clip in clips:
            clip.close()
        for audio_clip in audio_clips:
            audio_clip.close()
        if 'final_video' in locals():
            final_video.close()

    # Validate output
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError(f"Video assembly failed - output not created: {output_path}")

    return output_path


def preview_scene(
    scene: dict,
    project_dir: Path,
    output_filename: str | None = None,
    fps: int = 24,
) -> Path | None:
    """
    Create a preview video for a single scene.

    Args:
        scene: Scene dictionary with image_path and audio_path
        project_dir: Project directory
        output_filename: Name of preview file (auto-generated if None)
        fps: Frames per second

    Returns:
        Path to the preview video, or None if scene has no assets
    """
    project_dir = Path(project_dir)

    image_path = scene.get("image_path")
    audio_path = scene.get("audio_path")

    if not image_path or not Path(image_path).exists():
        return None

    scene_id = scene.get("id", 0)
    if output_filename is None:
        output_filename = f"preview_scene_{scene_id:03d}.mp4"

    output_path = project_dir / "previews" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_clip = None
    audio_clip = None

    try:
        # Create video from single scene and ensure correct dimensions
        image_clip = ImageClip(str(image_path))
        image_clip = _ensure_dimensions(image_clip, scene_id)

        if audio_path and Path(audio_path).exists():
            audio_clip = AudioFileClip(str(audio_path))
            image_clip = image_clip.with_duration(audio_clip.duration)
            image_clip = image_clip.with_audio(audio_clip)
        else:
            image_clip = image_clip.with_duration(5.0)

        image_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",  # CPU encoding - faster for slideshow content
            audio_codec="aac",
            ffmpeg_params=[
                "-preset", "ultrafast",
                "-crf", "35",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
            logger=None,  # Quiet for previews
        )

    finally:
        if image_clip:
            image_clip.close()
        if audio_clip:
            audio_clip.close()

    # Validate output
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError(f"Preview generation failed: {output_path}")

    return output_path


def get_video_duration(scenes: list[dict]) -> float:
    """
    Calculate total video duration from scenes.

    Args:
        scenes: List of scene dictionaries

    Returns:
        Total duration in seconds
    """
    total_duration = 0.0

    for scene in scenes:
        audio_path = scene.get("audio_path")

        if audio_path and Path(audio_path).exists():
            audio_clip = AudioFileClip(str(audio_path))
            try:
                total_duration += audio_clip.duration
            finally:
                audio_clip.close()
        else:
            # Default duration for scenes without audio
            total_duration += 5.0

    return total_duration


# --- V2 Assembly: ffmpeg-based, clip + audio mux, h264_videotoolbox on Mac ---

_FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
_PAD_COLOR = "0a0a0f"


def _get_media_duration(path: Path) -> float:
    """Get media duration in seconds via ffprobe."""
    result = subprocess.run(
        [_FFPROBE, "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def _pick_encoder() -> tuple[list[str], str]:
    """Pick the best H.264 encoder: h264_videotoolbox on Mac, libx264 fallback."""
    if platform.system() == "Darwin":
        # Try videotoolbox first
        result = subprocess.run(
            [_FFMPEG, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_videotoolbox" in result.stdout:
            return ["-c:v", "h264_videotoolbox", "-b:v", "4M"], "h264_videotoolbox"
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "23"], "libx264"


def _mux_segment(
    clip_path: Path,
    audio_path: Path,
    output_path: Path,
    encoder_args: list[str],
    fps: int = 30,
) -> Path:
    """Mux one video clip + audio into a segment. No tpad -- clip is already correct duration."""
    audio_dur = _get_media_duration(audio_path)

    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color={_PAD_COLOR}"
    )

    cmd = [
        _FFMPEG, "-y",
        "-i", str(clip_path),
        "-i", str(audio_path),
        *encoder_args,
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", vf,
        "-r", str(fps),
        "-t", str(audio_dur),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Mux failed for {clip_path.name}: {result.stderr[-300:]}")
    return output_path


def assemble_v2_video(
    scenes: list[dict[str, Any]],
    project_dir: Path,
    output_filename: str | None = None,
    fps: int = 30,
) -> Path:
    """Assemble v2 video from pre-recorded clips + audio.

    Each scene dict must have clip_path and audio_path.
    No tpad needed -- clips are already recorded to match audio duration.
    Uses h264_videotoolbox on Mac for GPU acceleration.

    Args:
        scenes: List of scene dicts with clip_path and audio_path
        project_dir: Project directory
        output_filename: Output filename (default: <project_name>.mp4)
        fps: Frames per second

    Returns:
        Path to the assembled video
    """
    project_dir = Path(project_dir)
    tmp_dir = project_dir / "tmp_segments"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    encoder_args, encoder_name = _pick_encoder()
    print(f"  Encoder: {encoder_name}")

    # Mux each scene
    segments: list[Path] = []
    for i, scene in enumerate(scenes):
        clip_path = Path(str(scene.get("clip_path", "")))
        audio_path = Path(str(scene.get("audio_path", "")))

        if not clip_path.exists():
            print(f"  [SKIP] Scene {i}: clip not found: {clip_path}")
            continue
        if not audio_path.exists():
            print(f"  [SKIP] Scene {i}: audio not found: {audio_path}")
            continue

        seg_path = tmp_dir / f"seg_{i:03d}.mp4"
        audio_dur = _get_media_duration(audio_path)
        clip_dur = _get_media_duration(clip_path)
        print(f"  [mux] Scene {i}: audio={audio_dur:.1f}s clip={clip_dur:.1f}s")

        _mux_segment(clip_path, audio_path, seg_path, encoder_args, fps)
        segments.append(seg_path)

    if not segments:
        raise RuntimeError("No segments to concatenate")

    # Concatenate
    concat_file = tmp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for seg in segments:
            f.write(f"file '{seg.resolve()}'\n")

    name = output_filename or f"{project_dir.name}.mp4"
    output_path = project_dir / name

    # Archive existing if present
    _archive_existing_video(output_path, project_dir)

    cmd = [
        _FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        *encoder_args,
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Concat failed: {result.stderr[-500:]}")

    total_dur = _get_media_duration(output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Final: {output_path.name} ({total_dur:.1f}s, {size_mb:.1f}MB, {encoder_name})")
    return output_path
