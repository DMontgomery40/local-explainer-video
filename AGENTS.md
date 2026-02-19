# Repository Agents (local-explainer-video)

This file is for **AI agents working in this repo**. For architecture, read `CLAUDE.md`.

## Start here (don’t skip)

1. Read `CLAUDE.md` for the pipeline and constraints.
2. `projects/<project>/plan.json` is the source of truth for narration + image prompts.
3. Understand the three different “image” actions (they are NOT interchangeable):
   - **Generate/Regenerate Image** → *new* image from prompt via the selected Replicate model (`qwen/qwen-image-2512` or `google/imagen-4`)
   - **Edit Image** → *surgical* edits to an existing PNG via `qwen/qwen-image-edit-2511`
   - **Refine Prompt** → LLM rewrites the prompt (avoid for QC automation)

## QC + Publish (qEEG Council integration)

If `../qEEG-analysis` exists on the same machine, Step 3 includes a **QC + Publish** gate:

- Narrative ground truth: qEEG Council **Stage 4 consolidation**
- Numeric ground truth: qEEG Council **Stage 1 `_data_pack.json`**
- Narrative judge: **Claude Opus 4.6** (liberal ELI5; strict on contradictions + wrong patient-data numbers)
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

## Environment variables (common)

- `ANTHROPIC_API_KEY` (required for Opus narrative judge)
- `REPLICATE_API_TOKEN` (required for image generation/edit)
- `CLIPROXY_BASE_URL` / `CLIPROXY_API_KEY` (required for Gemini visual QC)
- `QEEG_ANALYSIS_DIR` (defaults to `../qEEG-analysis`)
- `QEEG_BACKEND_URL` (defaults to `http://127.0.0.1:8000`)
