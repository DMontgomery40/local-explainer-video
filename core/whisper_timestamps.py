"""Extract word-level timestamps from audio via Whisper on Replicate.

Dumb API call. Returns structured data. No interpretation.
Cost: ~$0.003 per 30s of audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .generation_receipts import (
    AssetOperation, atomic_json, current_asset_directory, digest_bytes,
    paid_bytes, request_digest,
)


WHISPER_MODEL = "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class WhisperResult:
    text: str
    words: list[WordTimestamp] = field(default_factory=list)
    duration: float = 0.0  # total audio duration in seconds


def get_word_timestamps(
    audio_path: Path,
    *,
    model: str = WHISPER_MODEL,
    task: str = "transcribe",
    batch_size: int = 64,
) -> WhisperResult:
    """Run Whisper on Replicate to get word-level timestamps.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        model: Replicate model identifier
        task: "transcribe" or "translate"
        batch_size: Whisper batch size

    Returns:
        WhisperResult with text, word-level timestamps, and duration
    """
    import replicate

    audio_path = Path(audio_path)
    request = {"provider": "replicate", "model": model,
               "audio_sha256": digest_bytes(audio_path.read_bytes()),
               "task": task, "timestamp": "word", "batch_size": batch_size}
    # Timing is keyed to the audio bytes and inference settings across both pipelines.
    with AssetOperation(audio_path.parent / '.generation', request_digest(request),
                        audio_path.name + '.whisper'):
        prediction_path = current_asset_directory() / 'prediction.json'

        def collect(prediction):
            if prediction.status not in {'succeeded', 'failed', 'canceled'}:
                prediction.wait()
            if prediction.status != 'succeeded':
                raise RuntimeError(f'Whisper prediction {prediction.id} {prediction.status}')
            if not isinstance(prediction.output, dict):
                raise ValueError('Whisper prediction returned invalid timing data')
            return json.dumps(prediction.output).encode()

        def dispatch():
            with audio_path.open('rb') as handle:
                inputs = {'audio': handle, 'task': task, 'timestamp': 'word', 'batch_size': batch_size}
                selector = {'version': model.split(':', 1)[1]} if ':' in model else {'model': model}
                prediction = replicate.predictions.create(**selector, input=inputs)
            # Save before any polling. A lost create response remains unknown, never replayed.
            atomic_json(prediction_path, {'id': prediction.id})
            return collect(prediction)

        def recover():
            if not prediction_path.exists():
                return None
            return collect(replicate.predictions.get(json.loads(prediction_path.read_text())['id']))

        output = json.loads(paid_bytes(request, dispatch, key='whisper', recover=recover))

    return _parse_whisper_output(output)


def _parse_whisper_output(output: Any) -> WhisperResult:
    """Parse Whisper Replicate output into structured WhisperResult."""
    if not isinstance(output, dict):
        return WhisperResult(text=str(output))

    text = str(output.get("text", "")).strip()
    chunks = output.get("chunks", [])

    words: list[WordTimestamp] = []
    for chunk in chunks:
        word_text = str(chunk.get("text", "")).strip()
        ts = chunk.get("timestamp", [0, 0])
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start
        else:
            start = end = 0.0
        if word_text:
            words.append(WordTimestamp(word=word_text, start=start, end=end))

    duration = words[-1].end if words else 0.0

    return WhisperResult(text=text, words=words, duration=duration)
