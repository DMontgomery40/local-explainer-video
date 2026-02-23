# Repository Agents (local-explainer-video)

This file is for **AI agents working in this repo**. For architecture, read `CLAUDE.md`.

## Start here (don't skip)

1. Read `CLAUDE.md` for the pipeline and constraints.
2. `projects/<project>/plan.json` is the source of truth for narration + scene metadata.
3. Slides are rendered by **Blender** via `core/render.py` → `qeeg-blender render`. There is no AI image generation. Text, numbers, and electrode positions are deterministic — they come from patient data, not prompts.

## QC + Publish (qEEG Council integration)

If `../qEEG-analysis` exists on the same machine, Step 3 includes a **QC + Publish** gate:

- Narrative ground truth: qEEG Council **Stage 4 consolidation**
- Numeric ground truth: qEEG Council **Stage 1 `_data_pack.json`**
- Narrative judge: **Claude Opus 4.6** (liberal ELI5; strict on contradictions + wrong patient-data numbers)
- Visual judge: **Gemini vision via CLIProxyAPI** (structural/layout issues in rendered slides only)
- Since text and numbers are rendered deterministically by Blender from patient data, **visual QC will never find text fidelity errors**. If Gemini flags wrong numbers, that indicates a data binding bug in `qeeg-blender` — block and alert, do not attempt image edits.
- Visual QC runs in **check-only mode**. When issues are found it writes:
  - `projects/<PROJECT>/qc_visual_issues.json`
- Narrative QC writes a full trace to `projects/<PROJECT>/qc_narrative_report.json`.
- If a patient-data number is wrong in the narrative, apply a surgical string replace to `plan.json`, then re-run QC.
- Publish targets:
  - `qEEG-analysis/data/portal_patients/<PATIENT_ID>/<PATIENT_ID>.mp4`
  - qEEG Council backend `POST /api/patients/{patient_uuid}/files` (DB-tracked)

## Quick commands

- Run app: `./start.sh`
- Manual: `/opt/homebrew/bin/python3.10 -m streamlit run app.py`
- CLI QC (check-only): `python3.10 qc_publish.py --project 09-23-1982-0`
- Batch (latest version per patient, valid patient IDs only): `python3.10 qc_publish_batch.py`

## Environment variables (common)

- `ANTHROPIC_API_KEY` (required for Opus narrative judge)
- `BLENDER_BIN` (optional; path to blender binary if not on PATH)
- `CLIPROXY_BASE_URL` / `CLIPROXY_API_KEY` (required for Gemini visual QC)
- `QEEG_ANALYSIS_DIR` (defaults to `../qEEG-analysis`)
- `QEEG_BACKEND_URL` (defaults to `http://127.0.0.1:8000`)
