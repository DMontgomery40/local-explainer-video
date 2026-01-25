# local-explainer-video

A local, offline-capable alternative to NotebookLM's "Audio Overview" and video generation features. Built for my specific use case (converting qEEG brain scan reports into patient-friendly explainer videos), but easily adaptable for:

- **Podcasts** – Skip image generation and export audio-only from the MP4
- **Slide decks** – Use the generated images and script
- **Any text-to-video workflow** – Educational content, documentation walkthroughs, etc.

Essentially a local version of NotebookLM Studio, minus the RAG part.

## Full Disclosure

I didn't really create anything here. It's just pieces of things taped together:
- Streamlit for the UI
- Kokoro for local TTS (or OpenAI as fallback)
- Replicate for image generation
- MoviePy + ffmpeg for video assembly
- Claude/GPT for the "director" that breaks text into scenes

If it saves y'all a few minutes, feel free to grab or fork. 🤷

---

## What It Does

**Pipeline**: Input text → Director Agent → `plan.json` → Per-scene asset generation → Video assembly

1. Paste any text (clinical reports, documentation, scripts, whatever)
2. An LLM breaks it into 5-15 scenes with narration and image prompts
3. Generate images locally or via API
4. Generate voiceover locally with Kokoro TTS
5. Stitch it all together into an MP4

Everything is editable scene-by-scene. Regenerate just one image, tweak the narration, whatever you need.

---

## Quick Start

```bash
./start.sh
```

That's it. The script handles Python version checks and dependencies.

---

## Setup

### 1. System Dependencies

**macOS:**
```bash
brew install python@3.10 ffmpeg espeak-ng
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3.10 ffmpeg espeak-ng
```

### 2. Python Dependencies

```bash
/opt/homebrew/bin/python3.10 -m pip install -r requirements.txt
```

### 3. API Keys

Copy the example and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...          # Optional: for GPT-based director or TTS fallback
ANTHROPIC_API_KEY=sk-ant-...   # Optional: for Claude-based director
REPLICATE_API_TOKEN=r8_...     # Required: for image generation
```

You need at least one of OpenAI or Anthropic for the director agent.

### 4. Run

```bash
./start.sh
# or manually:
/opt/homebrew/bin/python3.10 -m streamlit run app.py
```

Opens at http://localhost:8501

---

## Adapting for Other Uses

### Podcast Mode

Skip image generation entirely. Generate just the audio scenes, then:

```bash
ffmpeg -i final_video.mp4 -vn -acodec libmp3lame podcast.mp3
```

Or modify `video_assembly.py` to export audio-only.

### Slide Deck

The `plan.json` contains all scene data. Pull the image paths and narration text to build slides in your preferred format.

### Different Content Types

Edit `prompts/director_system.txt` to change how the LLM breaks down your input text. The current version is optimized for medical explainers, but you can adapt it for:
- Technical documentation
- Product demos
- Educational lectures
- Whatever

---

## Voice Options

Uses Kokoro TTS locally. Configure in the sidebar:

**Voices:**
- `af_bella` – Warm, friendly (default)
- `af_sarah` – Clear, enthusiastic
- `af_heart` – Gentle, reassuring
- `am_adam` – Warm male voice
- British options: `bf_emma`, `bm_george`, etc.

**Speed:** 0.8 (slow) to 1.5 (fast), default 1.1

---

## Project Structure

```
local-explainer-video/
├── start.sh                 # One-command startup
├── app.py                   # Streamlit UI
├── core/
│   ├── director.py          # LLM storyboarding
│   ├── image_gen.py         # Replicate image generation
│   ├── voice_gen.py         # Kokoro/OpenAI TTS
│   └── video_assembly.py    # MoviePy video stitching
├── prompts/
│   ├── director_system.txt  # Storyboard prompt (edit this!)
│   └── refiner_system.txt   # Image prompt refinement
└── projects/                # Generated projects (gitignored)
```

---

## Troubleshooting

**Kokoro won't load:** Make sure you're on Python 3.10 and espeak-ng is installed.

**Images failing:** Check your Replicate API token and account credits.

**Video won't play:** Uses H.264 + AAC. Try VLC or a browser.

---

## QC + Publish (qEEG Council Integration)

If you also have the qEEG Council repo (`qEEG-analysis`) on the same machine, the app can run a final verification gate
before publishing an MP4 to the clinician portal sync folder.

**What it does**
- Loads ground truth from qEEG Council (Stage 4 consolidation + Stage 1 `_data_pack.json`)
- Uses a judge model (Claude Opus 4.5) to flag contradictions/wrong patient-data numbers (ELI5-friendly, liberal on analogies)
- Uses Gemini vision to find misspelled words / wrong patient numbers *in the rendered slide images*
- Fixes slide text via **Qwen Image Edit** (no regeneration), re-renders the MP4, then publishes to:
  - `qEEG-analysis/data/portal_patients/<PATIENT_ID>/<PATIENT_ID>.mp4`
  - qEEG Council backend `POST /api/patients/{patient_uuid}/files` (DB-tracked)

**Requirements**
- The project already has scene assets on disk (images + audio). QC does *not* generate missing assets.
- qEEG Council repo directory exists (default assumes `../qEEG-analysis`, override with `QEEG_ANALYSIS_DIR`)
- CLIProxyAPI is running and logged in (for Gemini vision checks)
- qEEG Council backend is running (for the upload step)

Run it from the UI:
- Open the patient project
- Go to **Step 3: Render Video**
- Click **Run QC + Publish**

Or run it from the CLI:

```bash
python3.10 qc_publish.py --project 09-23-1982-0
```

Common env vars:

```
QEEG_ANALYSIS_DIR=../qEEG-analysis
QEEG_BACKEND_URL=http://127.0.0.1:8000
CLIPROXY_BASE_URL=http://127.0.0.1:8317
CLIPROXY_API_KEY=
```

---

## License

MIT – Do whatever you want with it.
