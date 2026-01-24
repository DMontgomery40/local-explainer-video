# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

qEEG Explainer Video Generator - A Streamlit app that converts qEEG analysis text into patient-friendly slideshow videos (static AI-generated images + AI voiceover).

## Commands

```bash
# Run the app (recommended - handles everything)
./start.sh

# Manual run (IMPORTANT: must use Python 3.10 for Kokoro TTS)
/opt/homebrew/bin/python3.10 -m streamlit run app.py

# Install Python dependencies (use Python 3.10)
/opt/homebrew/bin/python3.10 -m pip install -r requirements.txt

# System dependencies (macOS)
brew install python@3.10 ffmpeg espeak-ng

# System dependencies (Linux)
sudo apt-get install python3.10 ffmpeg espeak-ng
```

## Architecture

**Pipeline**: Input text → Director Agent → plan.json → Per-scene asset generation → Video assembly

**Source of truth**: `projects/{project_name}/plan.json` - All scene data, paths, and refinement history live here.

**Core modules**:
- `core/director.py` - LLM-based storyboard generation (5-15 scenes from clinical text)
- `core/image_gen.py` - Replicate API (Qwen Image), with style suffix auto-appended
- `core/voice_gen.py` - Kokoro local TTS with configurable voice/speed, OpenAI TTS fallback
- `core/video_assembly.py` - MoviePy + ffmpeg, hard cuts only, 24fps, GPU acceleration

**Prompt files** (`prompts/`): Loaded at runtime, not hardcoded.

## Voice Configuration

Kokoro TTS supports multiple voices and speed settings:

**Voices** (set in sidebar):
- `af_bella` - Warm, friendly, upbeat (default - best for explainers)
- `af_sarah` - Clear, enthusiastic
- `af_heart` - Gentle, reassuring
- `am_adam` - Warm, friendly male
- British options: `bf_emma`, `bm_george`, etc.

**Speed** (set in sidebar):
- `1.0` = normal
- `1.1` = slightly faster (default)
- `1.2` = 20% faster
- Range: 0.8 to 1.5

## Key Design Decisions

- **Editability**: Regenerate single scene assets without rebuilding entire video
- **Widget key stability**: Use `scene['uid']` not list index for Streamlit keys
- **Error handling**: Scene-scoped - one failure shouldn't crash the app
- **Caching**: Assets persist on disk; reuse unless user explicitly regenerates
- **Project naming**: Auto-increment folder name on collision unless "Overwrite" checkbox is enabled
- **Python 3.10**: Required for Kokoro TTS compatibility

## Environment Variables

Required in `.env`:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `REPLICATE_API_TOKEN`

## Constraints

- No PHI - only de-identified metrics
- Minimal V1 - no music, no transitions beyond hard cuts
- Default aspect ratio: 16:9
- Warm, reassuring tone - no diagnosis claims or fear language
