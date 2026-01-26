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
- `core/voice_gen.py` - Kokoro (local) / ElevenLabs / OpenAI TTS
- `core/video_assembly.py` - MoviePy + ffmpeg, hard cuts only, 24fps, GPU acceleration

**Prompt files** (`prompts/`): Loaded at runtime, not hardcoded.

## QC + Publish (qEEG Council Integration)

This repo can optionally run a final QA gate against **qEEG Council** ground truth before publishing an MP4 into the
clinician portal sync folder.

**Ground truth**
- Narrative truth: qEEG Council **Stage 4 consolidation** (markdown)
- Numeric truth: qEEG Council **Stage 1 `_data_pack.json`**

**Models**
- Narrative judge: **Claude Opus 4.5** (Anthropic API)
- Visual judge: **Gemini 3 Flash** via **CLIProxyAPI** (vision over rendered slide PNGs)
- Fixes: **Qwen Image Edit** (Replicate) — surgical edits only

**Non-negotiables**
- QC must be **ELI5-liberal**: ignore imperfect analogies; be strict only on contradictions + wrong patient-data numbers.
- For slide text issues, **never regenerate images**; use image edit on the existing PNG.
- Only change `visual_prompt` when the prompt itself contains a wrong patient number (surgical string replace).
- Default behavior is **check-only visual QC** (no automated image edits). When issues are found, it writes:
  - `projects/<PROJECT>/qc_visual_issues.json`
  Enable auto-fix explicitly in the UI or with `--auto-fix-images`.

**How it works**
1. Loads qEEG Council ground truth for the patient ID (`MM-DD-YYYY-N`) from `qEEG-analysis/data/app.db`
2. Runs Opus narrative QC on `plan.json` (may apply high-confidence string replacements; blocks on critical issues)
3. Runs Gemini visual QC on each rendered slide PNG and blocks if issues are found (optionally applies fixes via image edit)
4. Re-renders the MP4, then publishes it to:
   - `qEEG-analysis/data/portal_patients/<PATIENT_ID>/<PATIENT_ID>.mp4`
   - qEEG Council backend `POST /api/patients/{patient_uuid}/files` (DB-tracked upload; non-fatal if backend is down)

Run it:
- Streamlit: Step 3 → **QC + Publish**
- CLI (check-only): `python3.10 qc_publish.py --project 09-23-1982-0`
- CLI (auto-fix images): `python3.10 qc_publish.py --project 09-23-1982-0 --auto-fix-images`
- Batch: `python3.10 qc_publish_batch.py` (latest version per patient, valid patient IDs only)

## Image Action Gotchas (Generate vs Edit)

These are intentionally different codepaths/models:
- **Generate/Regenerate Image** → `core/image_gen.generate_image()` → `qwen/qwen-image-2512` (new image)
- **Edit Image** → `core/image_gen.edit_image()` → DashScope `qwen-image-edit-max` (if `DASHSCOPE_API_KEY` is set) or Replicate `qwen/qwen-image-edit-2511` (fallback). Override via sidebar **Image Edit** or `IMAGE_EDIT_MODEL`.
- **Refine Prompt** → prompt rewrite step; avoid for QC automation

## DashScope Image Edit API (qwen-image-edit-*)

DashScope (Alibaba Model Studio) provides higher-quality image editing with additional parameters.

### Models Available

| Model | Quality | Outputs | Notes |
|-------|---------|---------|-------|
| `qwen-image-edit-max` | Highest | 1-6 | Best for production |
| `qwen-image-edit-plus` | Mid-tier | 1-6 | Faster, lower cost |
| `qwen-image-edit` | Base | 1 only | Single output only |

### API Parameters

| Parameter | Type | Range/Values | Default | Notes |
|-----------|------|--------------|---------|-------|
| `n` | int | 1-6 | 1 | Number of output variants (max/plus only) |
| `size` | string | "512*512" to "2048*2048" | auto from input | Explicit output dimensions |
| `prompt_extend` | bool | true/false | true | Let API expand prompt for better results |
| `negative_prompt` | string | max 500 chars | " " | Things to avoid in output |
| `watermark` | bool | true/false | false | Add watermark (disabled by default) |
| `seed` | int | 0-2147483647 | random | For reproducible edits |

### Input Capabilities

- **1-3 images** per request (multi-image fusion supported)
- Formats: JPG, JPEG, PNG, BMP, TIFF, WEBP, GIF
- Resolution: 384-3072 pixels per dimension
- Max file size: 10MB per image
- Prompt max: 800 characters

### Supported Edit Types

1. Text editing (modify text, font, color)
2. Object add/remove/move
3. Subject pose changes
4. Style transfer
5. Background replacement
6. Viewpoint transformation
7. Portrait modification
8. Old photo restoration

### UI Controls (Sidebar)

When a DashScope model is selected in the sidebar, additional controls appear:
- **Variants (n)**: Slider 1-6 for generating multiple output options
- **Seed**: Text input for reproducible edits (leave empty for random)
- **Prompt extend**: Checkbox to let DashScope expand your prompt
- **Negative prompt**: Text input for things to avoid

### Environment Variables

```bash
DASHSCOPE_API_KEY=sk-...         # Required for DashScope models
DASHSCOPE_REGION=SINGAPORE       # Optional: SINGAPORE (default) or BEIJING
DASHSCOPE_ENDPOINT=https://...   # Optional: Override endpoint URL
IMAGE_EDIT_MODEL=qwen-image-edit-max  # Optional: Default model
```

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

ElevenLabs (Flash v2.5) notes:
- Narration should spell out numbers as words (no digits). The storyboard + narration refiner prompts enforce this.
- Visual prompts should keep digit labels for slide text/QC (e.g., \"42%\", \"3.5 µV\").

## Key Design Decisions

- **Editability**: Regenerate single scene assets without rebuilding entire video
- **Widget key stability**: Use `scene['uid']` not list index for Streamlit keys
- **Error handling**: Scene-scoped - one failure shouldn't crash the app
- **Caching**: Assets persist on disk; reuse unless user explicitly regenerates
- **Project naming**: Auto-increment folder name on collision unless "Overwrite" checkbox is enabled
- **Python 3.10**: Required for Kokoro TTS compatibility

## Environment Variables

Base pipeline (`.env`):
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (director)
- `REPLICATE_API_TOKEN` (image gen/edit)
- `ELEVENLABS_API_KEY` (optional; required if selecting ElevenLabs TTS)

QC + Publish (optional):
- `CLIPROXY_BASE_URL` / `CLIPROXY_API_KEY` (Gemini visual QC)
- `QEEG_ANALYSIS_DIR` (defaults to `../qEEG-analysis`)
- `QEEG_BACKEND_URL` (defaults to `http://127.0.0.1:8000`)

## Constraints

- No PHI - only de-identified metrics
- Minimal V1 - no music, no transitions beyond hard cuts
- Default aspect ratio: 16:9
- Warm, reassuring tone - no diagnosis claims or fear language
