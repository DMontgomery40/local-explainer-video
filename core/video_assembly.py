"""Video assembly using MoviePy."""

from pathlib import Path

from moviepy import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)


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

    clips = []

    for scene in scenes:
        image_path = scene.get("image_path")
        audio_path = scene.get("audio_path")

        # Skip scenes without images
        if not image_path or not Path(image_path).exists():
            print(f"Skipping scene {scene['id']}: no image")
            continue

        # Create image clip
        image_clip = ImageClip(str(image_path))

        # Add audio if available
        if audio_path and Path(audio_path).exists():
            audio_clip = AudioFileClip(str(audio_path))
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

    # Write output video using Apple Silicon hardware encoder
    final_video.write_videofile(
        str(output_path),
        fps=fps,
        codec="h264_videotoolbox",  # Apple Silicon GPU acceleration
        audio_codec="aac",
        temp_audiofile=str(project_dir / "temp_audio.m4a"),
        remove_temp=True,
        logger="bar",  # Progress bar
        ffmpeg_params=[
            "-allow_sw", "0",  # Disable software fallback - force GPU
            "-q:v", "65",  # Quality (0-100, higher = better)
            "-pix_fmt", "yuv420p",  # Compatible pixel format
        ],
    )

    # Clean up clips
    for clip in clips:
        clip.close()
    final_video.close()

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

    scene_id = scene["id"]
    if output_filename is None:
        output_filename = f"preview_scene_{scene_id:03d}.mp4"

    output_path = project_dir / "previews" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create video from single scene
    image_clip = ImageClip(str(image_path))

    if audio_path and Path(audio_path).exists():
        audio_clip = AudioFileClip(str(audio_path))
        image_clip = image_clip.with_duration(audio_clip.duration)
        image_clip = image_clip.with_audio(audio_clip)
    else:
        image_clip = image_clip.with_duration(5.0)

    image_clip.write_videofile(
        str(output_path),
        fps=fps,
        codec="h264_videotoolbox",  # Apple Silicon GPU acceleration
        audio_codec="aac",
        ffmpeg_params=[
            "-allow_sw", "0",  # Force GPU
            "-q:v", "65",
            "-pix_fmt", "yuv420p",
        ],
        logger=None,  # Quiet for previews
    )

    image_clip.close()

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
            total_duration += audio_clip.duration
            audio_clip.close()
        else:
            # Default duration for scenes without audio
            total_duration += 5.0

    return total_duration
