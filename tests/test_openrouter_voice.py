from __future__ import annotations

import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from core import voice_gen


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class OpenRouterVoiceTest(unittest.TestCase):
    def test_gemini_pcm_is_written_as_24khz_mono_wav_with_exact_profile(self) -> None:
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _Response(b"\x01\x00\x02\x00")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene.wav"
            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
                patch.object(voice_gen.requests, "post", side_effect=fake_post),
                patch.object(
                    voice_gen.openai_limiter,
                    "call_with_retry",
                    side_effect=lambda fn: fn(),
                ),
            ):
                result = voice_gen.generate_audio(
                    "The exact narration.",
                    output,
                    tts_provider="openrouter",
                    voice="Charon",
                    speed=1.0,
                    openrouter_model="google/gemini-3.1-flash-tts-preview",
                )

            self.assertEqual(result, output)
            with wave.open(str(output), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 24000)
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.readframes(2), b"\x01\x00\x02\x00")

        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/audio/speech")
        self.assertEqual(captured["json"]["model"], "google/gemini-3.1-flash-tts-preview")
        self.assertEqual(captured["json"]["voice"], "Charon")
        self.assertEqual(captured["json"]["speed"], 1.0)
        self.assertEqual(captured["json"]["response_format"], "pcm")

    def test_openrouter_voice_requires_its_own_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                    voice_gen.generate_audio(
                        "Narration.",
                        Path(tmp) / "scene.wav",
                        tts_provider="openrouter",
                        voice="Charon",
                    )


if __name__ == "__main__":
    unittest.main()
