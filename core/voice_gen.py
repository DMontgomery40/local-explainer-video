"""Text-to-speech generation using premium narration providers."""

import base64
import os
import wave
from pathlib import Path
from typing import Literal

import requests

from core.generation_receipts import paid_bytes, atomic_bytes, image_generation_action
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
    "George - Warm, Captivating Storyteller": ("JBFqnCBsd6RMkjVDRZzb", "British male storyteller"),
    "Daniel - Steady Broadcaster": ("onwK4e9ZLuTAKqWW03F9", "British male broadcaster"),
}

DEFAULT_ELEVENLABS_VOICE = "Antoni"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"

# Default ElevenLabs voice settings for warm, engaging narration
DEFAULT_ELEVENLABS_STABILITY = 0.4
DEFAULT_ELEVENLABS_SIMILARITY_BOOST = 0.75
DEFAULT_ELEVENLABS_STYLE = 0.4
DEFAULT_ELEVENLABS_SPEED = 1.15
DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST = True

ElevenLabsTextNormalization = Literal["auto", "on", "off"]
DEFAULT_ELEVENLABS_TEXT_NORMALIZATION: ElevenLabsTextNormalization = "auto"

# Default settings for premium clinic narration
DEFAULT_VOICE = DEFAULT_ELEVENLABS_VOICE
DEFAULT_SPEED = 1.0
DEFAULT_EXAGGERATION = 0.6  # Slightly more expressive than neutral (0.5)
DEFAULT_OPENAI_VOICE = "onyx"
DEFAULT_OPENAI_MODEL = "tts-1-hd"
DEFAULT_OPENROUTER_VOICE = "Charon"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-tts-preview"
DEFAULT_OPENAI_INSTRUCTIONS = (
    "Adult male voice. Natural American English. Warm, direct, confident, and conversational. "
    "Read the narration with a premium documentary-clinic tone, natural pauses, grounded pacing, "
    "and clear emphasis on the numbers."
)

# TTS Provider type
TTSProvider = Literal["elevenlabs", "elevenlabs_replicate", "chatterbox", "openai", "openrouter"]


def generate_audio(
    text: str,
    output_path: str | Path,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    tts_provider: TTSProvider = "elevenlabs",
    exaggeration: float = DEFAULT_EXAGGERATION,
    # ElevenLabs settings (used when tts_provider=="elevenlabs")
    elevenlabs_model_id: str = DEFAULT_ELEVENLABS_MODEL,
    elevenlabs_apply_text_normalization: ElevenLabsTextNormalization = DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
    elevenlabs_stability: float = DEFAULT_ELEVENLABS_STABILITY,
    elevenlabs_similarity_boost: float = DEFAULT_ELEVENLABS_SIMILARITY_BOOST,
    elevenlabs_style: float = DEFAULT_ELEVENLABS_STYLE,
    elevenlabs_use_speaker_boost: bool = DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_instructions: str = DEFAULT_OPENAI_INSTRUCTIONS,
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
) -> Path:
    """
    Generate speech audio from text.

    Args:
        text: The text to convert to speech
        output_path: Where to save the audio file
        voice: Voice identifier (provider-dependent: ElevenLabs voice name like "Antoni", or provider-specific id)
        speed: Speech speed multiplier (provider-dependent)
        tts_provider: TTS provider ("elevenlabs", "chatterbox", or "openai")
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
        raise ValueError("Kokoro narration is disabled for clinic renders; use ElevenLabs or OpenAI tts-1-hd.")
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
    elif tts_provider == "elevenlabs_replicate":
        return _generate_with_elevenlabs_replicate(
            text=text,
            output_path=output_path,
            voice=voice,
            speed=speed,
            stability=elevenlabs_stability,
            similarity_boost=elevenlabs_similarity_boost,
            style=elevenlabs_style,
        )
    elif tts_provider == "chatterbox":
        return _generate_with_chatterbox(text, output_path, exaggeration)
    elif tts_provider == "openai":
        if openai_model.startswith("gpt-realtime") or openai_model.startswith("gpt-4o"):
            raise ValueError("OpenAI realtime and 4o-based audio models are disabled for clinic narration.")
        return _generate_with_openai(
            text,
            output_path,
            voice=voice or DEFAULT_OPENAI_VOICE,
            model=openai_model,
            instructions=openai_instructions,
        )
    elif tts_provider == "openrouter":
        return _generate_with_openrouter(
            text,
            output_path,
            voice=voice or DEFAULT_OPENROUTER_VOICE,
            model=openrouter_model,
            speed=speed,
        )
    else:
        raise ValueError(f"Unknown TTS provider: {tts_provider}")


def _replicate_audio_output(model: str, inputs: dict, output_path: Path):
    """Save the creation acknowledgement before polling or downloading."""
    import replicate
    client = replicate.Client()
    prediction_id = image_limiter.call_with_retry(lambda: paid_bytes(
        {"provider": "replicate", "base_url": os.getenv("REPLICATE_BASE_URL") or "https://api.replicate.com",
         "model": model, "input": inputs},
        lambda: client.models.predictions.create(model=model, input=inputs, wait=False).id.encode(),
        output_path=output_path))
    prediction = client.predictions.get(prediction_id.decode())
    if prediction.status not in {"succeeded", "failed", "canceled"}:
        prediction.wait()
    if prediction.status != "succeeded":
        raise RuntimeError(f"Acknowledged Replicate prediction {prediction_id.decode()} is {prediction.status}")
    return prediction.output


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
    output_url = _replicate_audio_output("resemble-ai/chatterbox", {
        "prompt": text, "exaggeration": exaggeration, "cfg_weight": cfg_weight,
        "temperature": temperature, "seed": 0}, output_path)

    # Download the audio file
    response = requests.get(output_url, timeout=(10, 180))
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


def _generate_with_elevenlabs_replicate(
    *,
    text: str,
    output_path: Path,
    voice: str = DEFAULT_ELEVENLABS_VOICE,
    model_slug: str = "elevenlabs/flash-v2.5",
    speed: float = 1.15,
    stability: float = DEFAULT_ELEVENLABS_STABILITY,
    similarity_boost: float = DEFAULT_ELEVENLABS_SIMILARITY_BOOST,
    style: float = DEFAULT_ELEVENLABS_STYLE,
) -> Path:
    """Generate audio using ElevenLabs on Replicate.

    Avoids direct ElevenLabs API quota issues; billing goes through Replicate.
    Follows the same pattern as _generate_with_chatterbox().
    """
    import replicate

    # Replicate ElevenLabs has its own voice set (different from direct API)
    REPLICATE_ELEVENLABS_VOICES = {
        "Rachel", "Drew", "Clyde", "Paul", "Aria", "Domi", "Dave", "Roger",
        "Fin", "Sarah", "James", "Jane", "Juniper", "Arabella", "Hope",
        "Bradford", "Reginald", "Gaming", "Austin", "Kuon", "Blondie",
        "Priyanka", "Alexandra", "Monika", "Mark", "Grimblewood",
    }
    # Map direct-API voice names to closest Replicate equivalents
    VOICE_MAP = {
        "Antoni": "Drew",     # Calm professional male
        "Josh": "Dave",       # Deep authoritative male
        "Adam": "Mark",       # Deep warm male
        "Arnold": "Austin",   # Bold energetic male
        "Rachel": "Rachel",   # Warm calm female
        "Bella": "Aria",      # Friendly conversational female
        "Elli": "Jane",       # Young energetic female
    }
    replicate_voice = VOICE_MAP.get(voice, voice)
    if replicate_voice not in REPLICATE_ELEVENLABS_VOICES:
        replicate_voice = "Drew"  # safe fallback

    output_url = _replicate_audio_output(model_slug, {
        "prompt": text, "voice": replicate_voice, "speed": speed, "stability": stability,
        "similarity_boost": similarity_boost, "style": style}, output_path)

    # Download the audio file (Replicate returns a URL)
    if hasattr(output_url, "url"):
        url = output_url.url
    elif isinstance(output_url, str):
        url = output_url
    else:
        url = str(output_url)

    response = requests.get(url, timeout=(10, 180))
    response.raise_for_status()

    # Save as the output format (likely mp3 from ElevenLabs)
    temp_path = output_path.with_suffix(".mp3")
    temp_path.write_bytes(response.content)

    # Convert to WAV if needed
    if output_path.suffix == ".wav":
        wav_path = output_path
        _convert_mp3_to_wav(temp_path, wav_path)
        temp_path.unlink(missing_ok=True)
        return wav_path

    if temp_path != output_path:
        temp_path.rename(output_path)
    return output_path


def _convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    """Convert MP3 to WAV using soundfile or ffmpeg fallback."""
    import subprocess
    import shutil

    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(mp3_path), "-ar", "24000", "-ac", "1", str(wav_path)],
        capture_output=True, timeout=30, check=True,
    )


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
    raw = elevenlabs_limiter.call_with_retry(lambda: paid_bytes(
        {"provider": "elevenlabs", "url": url, "output_format": "mp3_44100_128", **payload},
        lambda: _call_elevenlabs().content, output_path=output_path))

    mp3_path = output_path.with_suffix(".mp3")
    atomic_bytes(mp3_path, raw)

    if output_path.suffix == ".wav":
        _convert_mp3_to_wav(mp3_path, output_path)
        mp3_path.unlink(missing_ok=True)
        return output_path

    return mp3_path


def _generate_with_openai(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_OPENAI_VOICE,
    model: str = DEFAULT_OPENAI_MODEL,
    instructions: str = DEFAULT_OPENAI_INSTRUCTIONS,
) -> Path:
    """Generate audio using OpenAI TTS with rate limiting."""
    import openai

    if model.startswith("gpt-realtime") or model.startswith("gpt-4o"):
        raise ValueError("OpenAI realtime and 4o-based audio models are disabled for clinic narration.")

    client = openai.OpenAI(max_retries=0)

    # The general audio model family works through chat-completions audio output.
    if model.startswith("gpt-audio") or "audio-preview" in model:
        def _call_openai_chat_audio():
            return client.chat.completions.create(
                model=model,
                modalities=["audio"],
                audio={
                    "voice": voice,
                    "format": "wav",
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{instructions}\n\n"
                            "Read the user's script exactly as written. "
                            "Do not add a preamble, conclusion, or extra commentary. "
                            "Do not summarize. Do not rewrite. Speak the script verbatim."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                temperature=0.2,
            )

        audio_b64 = openai_limiter.call_with_retry(lambda: paid_bytes(
            {"provider": "openai-chat-audio", "base_url": str(client.base_url), "model": model,
             "voice": voice, "instructions": instructions, "text": text, "format": "wav", "temperature": 0.2},
            lambda: _call_openai_chat_audio().choices[0].message.audio.data.encode("ascii"), output_path=output_path))
        audio_bytes = base64.b64decode(audio_b64)

        wav_path = output_path if output_path.suffix == ".wav" else output_path.with_suffix(".wav")
        wav_path.write_bytes(audio_bytes)
        if output_path.suffix == ".wav":
            return wav_path
        return wav_path

    def _call_openai():
        return client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="mp3",
            instructions=instructions,
        )

    raw = openai_limiter.call_with_retry(lambda: paid_bytes(
        {"provider": "openai-tts", "base_url": str(client.base_url), "model": model, "voice": voice,
         "input": text, "response_format": "mp3", "instructions": instructions},
        lambda: _call_openai().content, output_path=output_path))
    mp3_path = output_path.with_suffix(".mp3")
    atomic_bytes(mp3_path, raw)

    # Convert to WAV for consistency (MoviePy works better with WAV)
    if output_path.suffix == ".wav":
        _convert_mp3_to_wav(mp3_path, output_path)
        mp3_path.unlink()  # Remove temporary MP3
        return output_path

    return mp3_path


def _generate_with_openrouter(
    text: str,
    output_path: Path,
    *,
    voice: str = DEFAULT_OPENROUTER_VOICE,
    model: str = DEFAULT_OPENROUTER_MODEL,
    speed: float = 1.0,
) -> Path:
    """Generate Gemini narration through OpenRouter's TTS endpoint."""
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    resolved_model = str(model or DEFAULT_OPENROUTER_MODEL).strip()
    resolved_voice = str(voice or DEFAULT_OPENROUTER_VOICE).strip()
    response_format = (
        "pcm"
        if resolved_model.startswith("google/gemini-") and "tts" in resolved_model
        else "mp3"
    )
    payload = {
        "model": resolved_model,
        "input": (
            "Read the following text exactly as a natural, measured, conversational documentary narrator. "
            "Use a normal speaking pace. Do not add, omit, summarize, or comment on any words.\n\n"
            f"{text}"
        ),
        "voice": resolved_voice,
        "response_format": response_format,
        "speed": float(speed or 1.0),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost/qeeg-clinic-workbench",
        "X-Title": "qEEG Clinic Workbench",
    }

    def _call_openrouter() -> requests.Response:
        response = requests.post(
            "https://openrouter.ai/api/v1/audio/speech",
            headers=headers,
            json=payload,
            timeout=(10, 300),
        )
        response.raise_for_status()
        return response

    raw = openai_limiter.call_with_retry(lambda: paid_bytes(
        {"provider": "openrouter", "url": "https://openrouter.ai/api/v1/audio/speech", **payload},
        lambda: _call_openrouter().content, output_path=output_path))
    if response_format == "pcm":
        wav_path = output_path if output_path.suffix == ".wav" else output_path.with_suffix(".wav")
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(raw)
        return wav_path

    mp3_path = output_path.with_suffix(".mp3")
    mp3_path.write_bytes(raw)
    if output_path.suffix == ".wav":
        _convert_mp3_to_wav(mp3_path, output_path)
        mp3_path.unlink(missing_ok=True)
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
    tts_provider: TTSProvider = "elevenlabs",
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
    openai_model: str = DEFAULT_OPENAI_MODEL,
    openai_instructions: str = DEFAULT_OPENAI_INSTRUCTIONS,
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    action_id: str | None = None,
) -> Path:
    """
    Generate audio for a specific scene.

    Args:
        scene: Scene dictionary with 'id' and 'narration'
        project_dir: Project directory for saving assets
        tts_provider: TTS provider ("elevenlabs", "openai", or an explicitly requested fallback)
        voice: Voice identifier (provider-dependent)
        speed: Speed multiplier (provider-dependent)
        exaggeration: Emotion intensity 0.25-2.0 (Chatterbox only)
        elevenlabs_*: ElevenLabs settings (used when tts_provider=="elevenlabs")
        action_id: Retain for retries of one scene action; omit for a new generation.

    Returns:
        Path to the generated audio
    """
    scene_id = scene["id"]
    narration = scene["narration"]

    output_path = project_dir / "audio" / f"scene_{scene_id:03d}.wav"

    # The same receipt scope serves image and audio actions and preserves an
    # enclosing render attempt. Low-level pipeline calls retain request receipts.
    return image_generation_action(generate_audio)(
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
        openai_model=openai_model,
        openai_instructions=openai_instructions,
        openrouter_model=openrouter_model,
        action_id=action_id,
    )
