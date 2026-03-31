# Repository Agents (local-explainer-video)

This file is for **AI agents working in this repo**. For architecture, read `CLAUDE.md`.

## Start here (don’t skip)

1. Read `CLAUDE.md` for the pipeline and constraints.
2. `projects/<project>/plan.json` is the source of truth for narration + image prompts.
3. Understand the three different “image” actions (they are NOT interchangeable):
   - **Generate/Regenerate Image** → *new* image from prompt via `qwen/qwen-image-2512`
   - **Edit Image** → *surgical* edits to an existing PNG via `qwen/qwen-image-edit-2511`
   - **Refine Prompt** → LLM rewrites the prompt (avoid for QC automation)

## QC + Publish (qEEG Council integration)

If `../qEEG-analysis` exists on the same machine, Step 3 includes a **QC + Publish** gate:

- Narrative ground truth: qEEG Council **Stage 4 consolidation**
- Numeric ground truth: qEEG Council **Stage 1 `_data_pack.json`**
- Narrative judge: **Claude Opus 4.5** (liberal ELI5; strict on contradictions + wrong patient-data numbers)
- Visual judge: **Gemini vision via CLIProxyAPI** (find misspelled words / wrong patient numbers in rendered slides)
- Fixes: **Qwen Image Edit only** (never regenerate images for text fixes)
- If the prompt text is correct but the rendered slide text is wrong, **do not rewrite the prompt** — fix the PNG via edit.
- If a patient-data number is wrong *in the prompt*, change **only that number** (surgical string replace), then re-run QC.
- By default, visual QC runs in **check-only mode** (no automated image edits). When issues are found it writes:
  - `projects/<PROJECT>/qc_visual_issues.json`
  Enable auto-fix in the UI by checking **Auto-fix slide text (image edit)** or via CLI `--auto-fix-images`.
- Narrative QC writes a full trace to `projects/<PROJECT>/qc_narrative_report.json` (what issues were found, what “safe fixes” were applied).
- If image-edit hits a provider quota/rate-limit (e.g., “reset after 48s”), QC will **auto-wait and continue**.
- Publish targets:
  - `qEEG-analysis/data/portal_patients/<PATIENT_ID>/<PATIENT_ID>.mp4`
  - qEEG Council backend `POST /api/patients/{patient_uuid}/files` (DB-tracked)

## Quick commands

- Run app: `./start.sh`
- Manual: `/opt/homebrew/bin/python3.10 -m streamlit run app.py`
- CLI QC (check-only): `python3.10 qc_publish.py --project 09-23-1982-0`
- CLI QC (auto-fix images): `python3.10 qc_publish.py --project 09-23-1982-0 --auto-fix-images`
- Batch (latest version per patient, valid patient IDs only): `python3.10 qc_publish_batch.py`

## Non-qEEG Hybrid Video Workflow

For explainer videos that mix AI images with pre-recorded demo clips (no qEEG, no director, no QC gate).

### Agent responsibilities

1. **Write `plan.json` by hand** — same schema, but skip `director.py`. Define which scenes are image scenes (AI-generated) and which are video scenes (demo clips). Use a `VIDEO_SCENES` dict in `assemble.py` to map scene IDs to clip files.

2. **Record demo clips** — use Playwright (headless=false, video recording) or QuickTime screen recording. Cut clips with ffmpeg. Verify each clip covers the narration window.

3. **Generate assets through Streamlit** — the normal pipeline UI handles audio (ElevenLabs) and images (Qwen) for all scenes. The agent sets up plan.json so the user just clicks through voice selection and image approval.

4. **Protect pre-made images** — Qwen overwrites ALL image slots during batch generation. If a scene uses a hand-crafted image (research figure, screenshot composite), keep a backup copy and restore it before assembly:
   ```
   cp scene_003_research_backup.png images/scene_003.png
   ```

5. **Run standalone `assemble.py`** — NOT the pipeline's `core/video_assembly.py`. Each hybrid project has its own `assemble.py` that handles the image/video mix via ffmpeg subprocess calls.

6. **Timing verification** — after assembly, check:
   - Total duration matches expected narration sum
   - No video scene has > 5s of frozen frame (last-frame freeze via `tpad=stop_mode=clone`)
   - If frozen frame is too long, re-record a longer clip or find additional footage

### Narration rules (ElevenLabs)

- Spell out numbers as words in narration text
- Use commas and `...` (ellipses) for pacing cues
- In visual prompts: color names not hex, double-quote literal text
- Keep digit labels in visual prompts for on-screen accuracy

### Gotchas

- **Qwen overwrites pre-made images** every time you regenerate — always backup/restore
- **Race conditions** — if Streamlit is still writing an image when you assemble, you get a stale frame. Check file modification timestamps before running `assemble.py`
- **Frozen frames** — `tpad=stop_mode=clone` freezes the last video frame when narration exceeds clip length. Acceptable for < 5s, noticeable above that
- **generate_images.py SKIP_SCENES** — must match `VIDEO_SCENES` in `assemble.py` plus any pre-made image scenes. If you add/remove scenes, update both

### Project structure

```
projects/<name>/
  plan.json              # Hand-written storyboard
  assemble.py            # Standalone ffmpeg hybrid assembly
  generate_images.py     # Optional batch Qwen gen (skips video + pre-made scenes)
  images/                # AI images + pre-made images
  audio/                 # ElevenLabs narration (all scenes)
  clips/                 # Demo video clips (cut from recordings)
  tmp_segments/          # Intermediate (auto-created by assemble.py)
  <name>.mp4             # Final output
```

## Environment variables (common)

- `ANTHROPIC_API_KEY` (required for Opus narrative judge)
- `REPLICATE_API_TOKEN` (required for image generation/edit)
- `ELEVENLABS_API_KEY` (required for ElevenLabs TTS)
- `CLIPROXY_BASE_URL` / `CLIPROXY_API_KEY` (required for Gemini visual QC)
- `QEEG_ANALYSIS_DIR` (defaults to `../qEEG-analysis`)
- `QEEG_BACKEND_URL` (defaults to `http://127.0.0.1:8000`)
