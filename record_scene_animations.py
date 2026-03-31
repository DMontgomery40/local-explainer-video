"""
Record animated HTML scene files as MP4 clips via headless Playwright.

Batch-captures every .html file in a directory at 1664x928, waits for the
animation to finish, then transcodes WebM -> MP4 (H.264, CRF 18, 30fps).

Output clips drop into a sibling `clips/` directory, named to match the
source HTML file.  These are ready for either pipeline:
  - local-explainer-video hybrid assemble.py (VIDEO_SCENES dict)
  - Cathode scene_type: "video" with videoUrl

Prerequisites:
  python3 -m pip install playwright
  python3 -m playwright install chromium
  ffmpeg installed (brew install ffmpeg)

Usage:
  # Record all HTML files in a scene_artifacts/ directory
  python3 record_scene_animations.py projects/<project>/scene_artifacts/

  # Override duration (seconds per scene, default auto-detect or 8s)
  python3 record_scene_animations.py projects/<project>/scene_artifacts/ --duration 10

  # Override output directory
  python3 record_scene_animations.py scenes/ --out clips/

  # Record a single file
  python3 record_scene_animations.py scenes/scene_05_p300_amplitude.html
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1664, "height": 928}
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# Per-scene duration hints: if the filename contains one of these substrings,
# use the corresponding duration.  Otherwise fall back to --duration flag.
DURATION_HINTS: dict[str, float] = {
    "title_card":           6.0,
    "roadmap":              5.0,
    "p300":                 6.0,
    "trail_making":         6.0,
    "trail_comparison":     6.0,
    "orchestra":            8.0,   # p5.js transition needs time
    "coherence":            6.0,
    "beta_coherence":       6.0,
    "brain_pathways":       8.0,   # p5.js transition needs time
    "full_picture":         5.0,
}

DEFAULT_DURATION = 8.0


def guess_duration(html_path: Path, override: float | None) -> float:
    """Pick recording duration from override > filename hint > default."""
    if override is not None:
        return override
    stem = html_path.stem.lower()
    for hint, dur in DURATION_HINTS.items():
        if hint in stem:
            return dur
    return DEFAULT_DURATION


def transcode(source: Path, output: Path) -> None:
    """WebM -> MP4 (H.264, CRF 18, 30fps, no audio)."""
    cmd = [
        FFMPEG, "-y",
        "-i", str(source),
        "-an",                     # no audio track (narration added later)
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-r", "30",
        "-pix_fmt", "yuv420p",    # broad compatibility
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {source.name}: {result.stderr[-500:]}")


def record_one(html_path: Path, out_dir: Path, raw_dir: Path,
               duration: float | None,
               cue_points_json: str | None = None) -> Path:
    """Open one HTML file in headless Chromium, record, transcode to MP4.

    If duration and cue_points_json are provided (v2 mode), they are passed
    as query parameters so the HTML scene can read them and sync animations.
    """
    dur = guess_duration(html_path, duration)
    clip_name = html_path.stem  # e.g. scene_05_p300_amplitude

    print(f"\n{'=' * 60}")
    print(f"  File:     {html_path.name}")
    print(f"  Duration: {dur:.1f}s")
    print(f"  Output:   {clip_name}.mp4")
    print(f"{'=' * 60}")

    existing = {p.name for p in raw_dir.glob("*.webm")}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(raw_dir),
            record_video_size=VIEWPORT,
        )
        page = context.new_page()

        file_url = html_path.resolve().as_uri()
        # v2: pass duration and cue points as query params
        import urllib.parse
        query_parts = []
        if duration is not None:
            query_parts.append(f"duration={dur:.1f}")
        if cue_points_json:
            query_parts.append(f"cues={urllib.parse.quote(cue_points_json)}")
        if query_parts:
            file_url += "?" + "&".join(query_parts)
        print(f"  Loading {file_url[:120]}{'...' if len(file_url) > 120 else ''}")
        page.goto(file_url, wait_until="load")

        # Small settle time for fonts / CDN scripts (Tailwind, p5.js)
        page.wait_for_timeout(1500)

        # Let the animation play
        print(f"  Recording for {dur:.1f}s...")
        time.sleep(dur)

        print("  Closing browser context...")
        page.close()
        context.close()
        browser.close()

    # Find the new recording
    new_files = [p for p in raw_dir.glob("*.webm") if p.name not in existing]
    if not new_files:
        raise FileNotFoundError(f"No Playwright recording found for {html_path.name}")

    newest = max(new_files, key=lambda p: p.stat().st_mtime)
    raw_target = raw_dir / f"{clip_name}.webm"
    if raw_target.exists():
        raw_target.unlink()
    newest.rename(raw_target)

    final = out_dir / f"{clip_name}.mp4"
    print(f"  Transcoding -> {final.name}")
    transcode(raw_target, final)
    print(f"  Done: {final}")
    return final


def record_scene_with_audio_duration(
    html_path: Path,
    out_dir: Path,
    audio_duration: float,
    cue_points_json: str | None = None,
    buffer: float = 1.5,
) -> Path:
    """V2 convenience function: record a scene for exact audio duration + buffer.

    Called by pipeline_v2.py. Passes duration and cues as query params
    so the HTML scene can sync animations to narration.
    """
    raw_dir = out_dir.parent / "raw_recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    total_duration = audio_duration + buffer
    return record_one(html_path, out_dir, raw_dir, total_duration, cue_points_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record animated HTML scenes as MP4 clips via Playwright")
    parser.add_argument("source", type=Path,
                        help="Directory of .html files, or a single .html file")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for MP4 clips (default: <source>/../clips)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override recording duration in seconds (default: auto per scene)")
    args = parser.parse_args()

    if not Path(FFMPEG).exists():
        print(f"ERROR: ffmpeg not found at {FFMPEG}", file=sys.stderr)
        sys.exit(1)

    # Resolve source files
    if args.source.is_file():
        html_files = [args.source]
        base_dir = args.source.parent
    elif args.source.is_dir():
        html_files = sorted(args.source.glob("*.html"))
        base_dir = args.source
    else:
        print(f"ERROR: {args.source} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    if not html_files:
        print(f"No .html files found in {args.source}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or base_dir.parent / "clips"
    raw_dir = base_dir.parent / "raw_recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"HTML Scene Animation Recorder")
    print(f"{'=' * 60}")
    print(f"  Source:    {args.source}")
    print(f"  Files:     {len(html_files)}")
    print(f"  Output:    {out_dir}")
    print(f"  Viewport:  {VIEWPORT['width']}x{VIEWPORT['height']}")
    print(f"  ffmpeg:    {FFMPEG}")

    outputs = []
    for html_file in html_files:
        outputs.append(record_one(html_file, out_dir, raw_dir, args.duration))

    print(f"\n{'=' * 60}")
    print(f"Recorded {len(outputs)} clips:")
    for o in outputs:
        print(f"  {o}")
    print(f"\nNext steps:")
    print(f"  - Use clips in local-explainer-video hybrid assemble.py (VIDEO_SCENES dict)")
    print(f"  - Or upload to Cathode as scene_type: 'video' scenes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
