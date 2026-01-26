"""Text-to-speech generation using Kokoro, ElevenLabs, Chatterbox, or OpenAI."""

import os
from pathlib import Path
from typing import Literal

import requests
import soundfile as sf

from core.rate_limiter import elevenlabs_limiter, openai_limiter, image_limiter

# ElevenLabs voices - curated selection for narration
# Full library at: https://elevenlabs.io/voice-library
ELEVENLABS_VOICES = {
    # Female voices
    "Rachel": ("21m00Tcm4TlvDq8ikWAM", "Warm, calm female - great for explainers"),
    "Bella": ("EXAVITQu4vr4xnSDxMaL", "Friendly, conversational female"),
    "Elli": ("MF3mGyEYCl7XYWbV9V6O", "Young, energetic female"),
    "Domi": ("AZnzlk1XvdvUeBnXmlld", "Strong, confident female"),
    # Male voices
    "Antoni": ("ErXwobaYiN019PkySvjV", "Calm, professional male"),
    "Josh": ("TxGEqnHWrfWFTfGW9XjX", "Deep, authoritative male"),
    "Adam": ("pNInz6obpgDQGcFmaJgB", "Deep, warm male"),
    "Arnold": ("VR6AewLTigWG4xSOukaG", "Bold, energetic male"),
}

DEFAULT_ELEVENLABS_VOICE = "Antoni"
DEFAULT_ELEVENLABS_MODEL = "eleven_flash_v2_5"  # Fast + affordable; requires good number spelling in narration

# Default ElevenLabs voice settings for warm, engaging narration
DEFAULT_ELEVENLABS_STABILITY = 0.4
DEFAULT_ELEVENLABS_SIMILARITY_BOOST = 0.75
DEFAULT_ELEVENLABS_STYLE = 0.4
DEFAULT_ELEVENLABS_SPEED = 1.15
DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST = True

ElevenLabsTextNormalization = Literal["auto", "on", "off"]
DEFAULT_ELEVENLABS_TEXT_NORMALIZATION: ElevenLabsTextNormalization = "auto"

# Available Kokoro voices
# American English (lang_code='a'):
#   Female: af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
#   Male: am_adam, am_michael
# British English (lang_code='b'):
#   Female: bf_emma, bf_isabella
#   Male: bm_george, bm_lewis

KOKORO_VOICES = {
    # American female - sorted by energy/upbeat quality
    "af_bella": "Warm, friendly, upbeat female",
    "af_sarah": "Clear, enthusiastic female",
    "af_heart": "Gentle, reassuring female (default)",
    "af_nicole": "Professional, confident female",
    "af_jessica": "Bright, energetic female",
    "af_nova": "Modern, dynamic female",
    "af_sky": "Light, airy female",
    "af_alloy": "Neutral, clear female",
    "af_aoede": "Melodic, expressive female",
    "af_kore": "Youthful, fresh female",
    "af_river": "Smooth, flowing female",
    # American male
    "am_adam": "Warm, friendly male",
    "am_michael": "Clear, professional male",
    # British female
    "bf_emma": "Warm British female",
    "bf_isabella": "Elegant British female",
    # British male
    "bm_george": "Classic British male",
    "bm_lewis": "Modern British male",
}

# Default settings for upbeat, engaging narration
DEFAULT_VOICE = "af_bella"  # More upbeat than af_heart
DEFAULT_SPEED = 1.1  # Slightly faster than normal
DEFAULT_EXAGGERATION = 0.6  # Slightly more expressive than neutral (0.5)

# TTS Provider type
TTSProvider = Literal["kokoro", "elevenlabs", "chatterbox", "openai"]


def generate_audio(
    text: str,
    output_path: str | Path,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    tts_provider: TTSProvider = "kokoro",
    exaggeration: float = DEFAULT_EXAGGERATION,
    # ElevenLabs settings (used when tts_provider=="elevenlabs")
    elevenlabs_model_id: str = DEFAULT_ELEVENLABS_MODEL,
    elevenlabs_apply_text_normalization: ElevenLabsTextNormalization = DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
    elevenlabs_stability: float = DEFAULT_ELEVENLABS_STABILITY,
    elevenlabs_similarity_boost: float = DEFAULT_ELEVENLABS_SIMILARITY_BOOST,
    elevenlabs_style: float = DEFAULT_ELEVENLABS_STYLE,
    elevenlabs_use_speaker_boost: bool = DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST,
) -> Path:
    """
    Generate speech audio from text.

    Args:
        text: The text to convert to speech
        output_path: Where to save the audio file
        voice: Voice identifier (provider-dependent: Kokoro voice id like "af_bella", or ElevenLabs voice name like "Rachel")
        speed: Speech speed multiplier (provider-dependent: Kokoro pipeline speed, or ElevenLabs voice_settings.speed)
        tts_provider: TTS provider ("kokoro", "elevenlabs", "chatterbox", or "openai")
        exaggeration: Emotion intensity 0.25-2.0 (0.5=neutral, higher=more expressive) - Chatterbox only
        elevenlabs_model_id: ElevenLabs model id (e.g., "eleven_flash_v2_5")
        elevenlabs_apply_text_normalization: ElevenLabs text normalization mode ("auto"|"on"|"off")
        elevenlabs_stability: ElevenLabs voice_settings.stability (0-1)
        elevenlabs_similarity_boost: ElevenLabs voice_settings.similarity_boost (0-1)
        elevenlabs_style: ElevenLabs voice_settings.style (0-1)
        elevenlabs_use_speaker_boost: ElevenLabs voice_settings.use_speaker_boost (bool)

    Returns:
        Path to the saved audio file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tts_provider == "kokoro":
        try:
            return _generate_with_kokoro(text, output_path, voice, speed)
        except Exception as e:
            # Fallback to OpenAI if Kokoro fails
            print(f"Kokoro TTS failed ({e}), falling back to OpenAI...")
            return _generate_with_openai(text, output_path)
    elif tts_provider == "elevenlabs":
        return _generate_with_elevenlabs(
            text=text,
            output_path=output_path,
            voice=voice,
            model_id=elevenlabs_model_id,
            stability=elevenlabs_stability,
            similarity_boost=elevenlabs_similarity_boost,
            style=elevenlabs_style,
            speed=speed,
            use_speaker_boost=elevenlabs_use_speaker_boost,
            apply_text_normalization=elevenlabs_apply_text_normalization,
        )
    elif tts_provider == "chatterbox":
        return _generate_with_chatterbox(text, output_path, exaggeration)
    elif tts_provider == "openai":
        return _generate_with_openai(text, output_path)
    else:
        raise ValueError(f"Unknown TTS provider: {tts_provider}")


def _generate_with_kokoro(text: str, output_path: Path, voice: str, speed: float = 1.0) -> Path:
    """Generate audio using Kokoro local TTS."""
    import numpy as np
    from kokoro import KPipeline

    # Determine language code from voice prefix
    # 'a' = American English, 'b' = British English
    lang_code = "b" if voice.startswith("b") else "a"

    # Initialize pipeline
    pipeline = KPipeline(lang_code=lang_code)

    # Generate audio - new API returns (graphemes, phonemes, audio) tuples
    # speed parameter controls speech rate (1.0 = normal, 1.2 = 20% faster)
    generator = pipeline(text, voice=voice, speed=speed)

    # Collect all audio chunks
    all_audio = []
    sample_rate = 24000  # Kokoro default

    for graphemes, phonemes, audio in generator:
        all_audio.append(audio)

    # Concatenate all chunks and save
    if all_audio:
        audio_data = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
        sf.write(str(output_path), audio_data, sample_rate)
    else:
        raise ValueError("No audio generated")

    return output_path


def _generate_with_chatterbox(
    text: str,
    output_path: Path,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
) -> Path:
    """
    Generate audio using Chatterbox on Replicate.

    Chatterbox supports emotion control and tags like [laugh], [cough], [chuckle].

    Args:
        text: Text to synthesize (can include emotion tags)
        output_path: Where to save the audio file
        exaggeration: Emotion intensity 0.25-2.0 (0.5=neutral, higher=more expressive)
        cfg_weight: Pace/CFG weight 0.2-1.0
        temperature: Variability 0.05-5.0

    Returns:
        Path to the saved audio file
    """
    import replicate

    def _call_chatterbox():
        return replicate.run(
            "resemble-ai/chatterbox",
            input={
                "prompt": text,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "seed": 0,  # Random seed for variety
            }
        )

    # Use image_limiter since it's also Replicate
    output_url = image_limiter.call_with_retry(_call_chatterbox)

    # Download the audio file
    response = requests.get(output_url)
    response.raise_for_status()

    # Chatterbox returns WAV, save directly
    # If output_path expects WAV, write directly; otherwise handle format
    temp_path = output_path.with_suffix(".wav")
    temp_path.write_bytes(response.content)

    # If caller wanted WAV, we're done
    if output_path.suffix == ".wav":
        if temp_path != output_path:
            temp_path.rename(output_path)
        return output_path

    # Otherwise convert (though WAV is preferred)
    return temp_path


def _generate_with_elevenlabs(
    *,
    text: str,
    output_path: Path,
    voice: str = DEFAULT_ELEVENLABS_VOICE,
    model_id: str = DEFAULT_ELEVENLABS_MODEL,
    stability: float = DEFAULT_ELEVENLABS_STABILITY,
    similarity_boost: float = DEFAULT_ELEVENLABS_SIMILARITY_BOOST,
    style: float = DEFAULT_ELEVENLABS_STYLE,
    speed: float = 1.0,
    use_speaker_boost: bool = DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST,
    apply_text_normalization: ElevenLabsTextNormalization = DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
) -> Path:
    """
    Generate audio using ElevenLabs Text-to-Speech.

    Docs: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
    """
    api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set. Add it to your .env to use ElevenLabs TTS.")

    # Allow passing a raw voice_id, but prefer curated voice names
    if voice in ELEVENLABS_VOICES:
        voice_id = ELEVENLABS_VOICES[voice][0]
    else:
        voice_id = voice  # assume caller passed an actual voice_id

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(stability),
            "similarity_boost": float(similarity_boost),
            "style": float(style),
            "use_speaker_boost": bool(use_speaker_boost),
            "speed": float(speed),
        },
        "apply_text_normalization": apply_text_normalization,
    }

    def _call_elevenlabs() -> requests.Response:
        resp = requests.post(
            url,
            params={"output_format": "mp3_44100_128"},
            headers=headers,
            json=payload,
            timeout=(10, 180),
        )
        resp.raise_for_status()
        return resp

    # Use dedicated ElevenLabs limiter (configurable via ELEVENLABS_MIN_DELAY_S / ELEVENLABS_MAX_RETRIES)
    response = elevenlabs_limiter.call_with_retry(_call_elevenlabs)

    mp3_path = output_path.with_suffix(".mp3")
    mp3_path.write_bytes(response.content)

    if output_path.suffix == ".wav":
        _convert_mp3_to_wav(mp3_path, output_path)
        mp3_path.unlink(missing_ok=True)
        return output_path

    return mp3_path


def _generate_with_openai(text: str, output_path: Path, voice: str = "nova") -> Path:
    """Generate audio using OpenAI TTS with rate limiting."""
    import openai

    client = openai.OpenAI()

    def _call_openai():
        return client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3",
        )

    response = openai_limiter.call_with_retry(_call_openai)

    # Save the audio
    mp3_path = output_path.with_suffix(".mp3")
    response.stream_to_file(str(mp3_path))

    # Convert to WAV for consistency (MoviePy works better with WAV)
    if output_path.suffix == ".wav":
        _convert_mp3_to_wav(mp3_path, output_path)
        mp3_path.unlink()  # Remove temporary MP3
        return output_path

    return mp3_path


def _convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    """Convert MP3 to WAV using pydub or ffmpeg."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio.export(str(wav_path), format="wav")
    except ImportError:
        # Fallback to ffmpeg
        import subprocess
        subprocess.run(
            ["ffmpeg", "-i", str(mp3_path), "-y", str(wav_path)],
            capture_output=True,
            check=True,
        )


def generate_scene_audio(
    scene: dict,
    project_dir: Path,
    tts_provider: TTSProvider = "kokoro",
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    exaggeration: float = DEFAULT_EXAGGERATION,
    # ElevenLabs passthrough
    elevenlabs_model_id: str = DEFAULT_ELEVENLABS_MODEL,
    elevenlabs_apply_text_normalization: ElevenLabsTextNormalization = DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
    elevenlabs_stability: float = DEFAULT_ELEVENLABS_STABILITY,
    elevenlabs_similarity_boost: float = DEFAULT_ELEVENLABS_SIMILARITY_BOOST,
    elevenlabs_style: float = DEFAULT_ELEVENLABS_STYLE,
    elevenlabs_use_speaker_boost: bool = DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST,
) -> Path:
    """
    Generate audio for a specific scene.

    Args:
        scene: Scene dictionary with 'id' and 'narration'
        project_dir: Project directory for saving assets
        tts_provider: TTS provider ("kokoro", "elevenlabs", "chatterbox", or "openai")
        voice: Voice identifier (provider-dependent)
        speed: Speed multiplier (provider-dependent)
        exaggeration: Emotion intensity 0.25-2.0 (Chatterbox only)
        elevenlabs_*: ElevenLabs settings (used when tts_provider=="elevenlabs")

    Returns:
        Path to the generated audio
    """
    scene_id = scene["id"]
    narration = scene["narration"]

    output_path = project_dir / "audio" / f"scene_{scene_id:03d}.wav"

    return generate_audio(
        narration,
        output_path,
        voice=voice,
        speed=speed,
        tts_provider=tts_provider,
        exaggeration=exaggeration,
        elevenlabs_model_id=elevenlabs_model_id,
        elevenlabs_apply_text_normalization=elevenlabs_apply_text_normalization,
        elevenlabs_stability=elevenlabs_stability,
        elevenlabs_similarity_boost=elevenlabs_similarity_boost,
        elevenlabs_style=elevenlabs_style,
        elevenlabs_use_speaker_boost=elevenlabs_use_speaker_boost,
    )
