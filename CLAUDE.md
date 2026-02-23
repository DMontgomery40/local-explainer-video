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
- `core/render.py` - Calls `qeeg-blender render` as subprocess, passes qEEGSession JSON, returns PNG path. Deterministic Blender output — no AI image generation.
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
- Narrative judge: **Claude Opus 4.6** (Anthropic API)
- Visual judge: **Gemini 3 Flash** via **CLIProxyAPI** (vision over rendered slide PNGs)

**Non-negotiables**
- QC must be **ELI5-liberal**: ignore imperfect analogies; be strict only on contradictions + wrong patient-data numbers.
- Slide text and numbers are rendered deterministically by Blender from patient data — visual QC issues are structural/layout problems only, never text fidelity. **Do not attempt image edits.** Block and alert instead.
- Default behavior is **check-only visual QC**. When issues are found, it writes:
  - `projects/<PROJECT>/qc_visual_issues.json`

**How it works**
1. Loads qEEG Council ground truth for the patient ID (`MM-DD-YYYY-N`) from `qEEG-analysis/data/app.db`
2. Runs Opus narrative QC on `plan.json` (may apply high-confidence string replacements; blocks on critical issues)
3. Runs Gemini visual QC on each rendered slide PNG and blocks if issues are found (optionally applies fixes via image edit)
4. Re-renders the MP4, then publishes it to:
   - `qEEG-analysis/data/portal_patients/<PATIENT_ID>/<PATIENT_ID>.mp4`
   - qEEG Council backend `POST /api/patients/{patient_uuid}/files` (DB-tracked upload; non-fatal if backend is down)

**Artifacts**
- Narrative trace: `projects/<PROJECT>/qc_narrative_report.json`
- Visual issues (check-only mode): `projects/<PROJECT>/qc_visual_issues.json`

**Rate limits**
- If image-edit hits a provider quota/rate-limit (e.g., “reset after 48s”), QC auto-waits and continues.

Run it:
- Streamlit: Step 3 → **QC + Publish**
- CLI (check-only): `python3.10 qc_publish.py --project 09-23-1982-0`
- CLI (auto-fix images): `python3.10 qc_publish.py --project 09-23-1982-0 --auto-fix-images`
- Batch: `python3.10 qc_publish_batch.py` (latest version per patient, valid patient IDs only)

## Rendering Architecture

Slides are rendered deterministically by **Blender** via the `qeeg-blender` CLI (see `../qeeg-blender/`). There is no AI image generation in this pipeline.

- `core/render.py` calls `qeeg-blender render --patient <session.json> --output <slide.png>`
- All text, electrode positions, μV values, Hz labels, and clinical notation come directly from patient data — no prompting, no hallucination risk
- To change visual style, edit the Blender template (`qeeg-blender build-template`), not prompts
- `BLENDER_BIN` env var overrides the `blender` binary path if not on PATH


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
- `BLENDER_BIN` (optional; path to blender binary if not on PATH)
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
