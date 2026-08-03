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
4. Generate voiceover with Kokoro (local) or ElevenLabs (API)
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

Create a `.env` (or export env vars) and add your keys:
```
OPENAI_API_KEY=sk-...          # Optional: for GPT-based director or TTS fallback
ANTHROPIC_API_KEY=sk-ant-...   # Optional: for Claude-based director
REPLICATE_API_TOKEN=r8_...     # Required: for image generation
DASHSCOPE_API_KEY=sk-...       # Optional: for DashScope Qwen image editing (qwen-image-edit-max/plus)
DASHSCOPE_REGION=SINGAPORE     # Optional: SINGAPORE (default) or BEIJING (keys/endpoints are region-specific)
ELEVENLABS_API_KEY=...         # Optional: required only if you select ElevenLabs TTS
REMOTION_SKILL_ID=skill_...    # Optional: attach your Anthropic custom Remotion skill in director.py
REMOTION_SKILL_VERSION=latest  # Optional: pin a specific skill version instead of latest

# Optional overrides for which model is used when editing existing images (UI "Edit Image" + QC auto-fix)
IMAGE_EDIT_MODEL=qwen-image-edit-max   # or: qwen/qwen-image-edit-2511
```

You need at least one of OpenAI or Anthropic for the director agent.

If you want the direct Claude API path to have access to your full Anthropic-hosted Remotion custom skill, set `REMOTION_SKILL_ID` to the remote `skill_...` id from Anthropic. A local Codex/Claude skill install helps us follow the right API pattern, but the API call itself still needs the remote Anthropic custom skill id.

To list existing custom skills or upload a local skill directory and print the resulting `skill_...` id:

```bash
python3 scripts/anthropic_skills.py list
python3 scripts/anthropic_skills.py create .claude/skills/remotion-best-practices --display-title "Remotion Best Practices"
```

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

Choose the TTS provider in the sidebar:
- Kokoro (local, free)
- ElevenLabs (Flash v2.5)
- OpenAI TTS

Note for ElevenLabs Flash v2.5:
- Narration should spell out numbers as words (no digits) for best results.
- Visual prompts can (and should) keep digit labels for slide text (e.g., \"42%\", \"3.5 µV\").

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
│   ├── image_gen.py         # Image generation (Replicate) + image editing (Replicate or DashScope)
│   ├── voice_gen.py         # Kokoro/ElevenLabs/OpenAI TTS
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

**DashScope image edit failing:** Make sure `DASHSCOPE_API_KEY` matches your region endpoint (`DASHSCOPE_REGION=SINGAPORE` for `dashscope-intl`).

**Video won't play:** Uses H.264 + AAC. Try VLC or a browser.

---

## QC + Publish (qEEG Council Integration)

If you also have the qEEG Council repo (`qEEG-analysis`) on the same machine, the app can run a final verification gate
before publishing an MP4 to the clinician portal sync folder.

**What it does**
- Loads ground truth from qEEG Council (Stage 4 consolidation + Stage 1 `_data_pack.json`)
- Uses a judge model (Claude Opus 4.5) to flag contradictions/wrong patient-data numbers (ELI5-friendly, liberal on analogies)
- Uses Gemini vision to find misspelled words / wrong patient numbers *in the rendered slide images*
- By default, visual QC runs in **check-only mode** (no automated image edits). When issues are found it writes:
  - `projects/<PROJECT>/qc_visual_issues.json`
- Optionally fixes slide text via **Qwen Image Edit** (no regeneration), re-renders the MP4, then publishes to:
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

Notes:
- If Streamlit restarts, re-load the project; if an MP4 exists, the app jumps straight to Step 3 so you can resume QC without re-rendering first.
- If image-edit hits a provider quota/rate-limit (e.g., “reset after 48s”), QC auto-waits and continues.

Or run it from the CLI:

```bash
python3.10 qc_publish.py --project ZZ_01-01-1900            # check-only
python3.10 qc_publish.py --project ZZ_01-01-1900 --auto-fix-images
python3.10 qc_publish.py --project ZZ_01-01-1900 --auto-fix-images --image-edit-model qwen-image-edit-max
```

Batch mode (latest version per patient, valid patient IDs only):

```bash
python3.10 qc_publish_batch.py
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
