from __future__ import annotations

import json
import math
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from mixed_video_assembly import FFMPEG, FFPROBE, assemble_mixed_video  # noqa: E402


def _write_pcm_wav(path: Path, channels: int) -> None:
    sample_rate = 44_100
    frame_count = sample_rate // 5
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for frame in range(frame_count):
            sample = int(8_000 * math.sin(2 * math.pi * 440 * frame / sample_rate))
            for _ in range(channels):
                frames.extend(struct.pack("<h", sample))
        handle.writeframes(frames)


class MixedVideoAssemblyAudioFormatTest(unittest.TestCase):
    def test_mixed_input_channels_produce_one_stable_youtube_audio_format(self) -> None:
        for input_channels in ((1, 2), (2, 1)):
            with self.subTest(input_channels=input_channels), tempfile.TemporaryDirectory() as raw:
                project_dir = Path(raw)
                image_path = project_dir / "frame.ppm"
                image_path.write_bytes(b"P6\n2 2\n255\n" + bytes([20, 80, 140]) * 4)

                scenes = []
                for index, channels in enumerate(input_channels):
                    audio_path = project_dir / f"scene-{index}.wav"
                    _write_pcm_wav(audio_path, channels)
                    scenes.append(
                        {
                            "image_path": image_path.name,
                            "audio_path": audio_path.name,
                        }
                    )

                output = assemble_mixed_video(scenes, project_dir, "mixed.mp4")
                probe = subprocess.run(
                    [
                        FFPROBE,
                        "-v",
                        "error",
                        "-select_streams",
                        "a:0",
                        "-show_entries",
                        "stream=codec_name,sample_rate,channels,channel_layout",
                        "-of",
                        "json",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                stream = json.loads(probe.stdout)["streams"][0]
                self.assertEqual(
                    {
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                        "channel_layout": "stereo",
                    },
                    stream,
                )

                decoded = subprocess.run(
                    [
                        FFMPEG,
                        "-hide_banner",
                        "-i",
                        str(output),
                        "-map",
                        "0:a:0",
                        "-af",
                        "ashowinfo",
                        "-f",
                        "null",
                        "-",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                decoded_channels = set(re.findall(r"channels:(\d+)", decoded.stderr))
                self.assertEqual({"2"}, decoded_channels)

                # YouTube explicitly warns that MP4 edit lists can prevent correct processing.
                self.assertNotIn(b"edts", output.read_bytes())


if __name__ == "__main__":
    unittest.main()
