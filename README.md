# local-explainer-video

Deterministic qEEG explainer video pipeline with curated Qwen template assets.

## Canonical Scope
Active implementation scope is defined by:
1. `/Users/davidmontgomery/local-explainer-video/HANDOFF-deterministic-template-pipeline.md`
2. `/Users/davidmontgomery/local-explainer-video/HANDOFF_NEXT_AGENT_DETERMINISTIC_TEMPLATE_PIPELINE_2026-02-22.md`
3. `/Users/davidmontgomery/local-explainer-video/.codex/REPO_CONTRACT.md`

## Runtime Pipeline (default)
Input text -> Director (`scene_type + structured_data`) -> template selector -> deterministic compositor -> audio -> MP4 assembly

For data-bearing slides, production rendering is template-driven and deterministic.

## Core Rules
- `projects/<project>/plan.json` is the source of truth.
- Data-bearing slides in production must not depend on runtime generative image creation.
- `generic_data_panel_v1` is development-only scaffolding.
- `origin=scaffold_only` templates remain `dev_only` until curated/approved.
- Production mode hard-fails when archetype coverage is missing.
- Runtime Qwen fallback is emergency-only and must be auditable and QC-gated.

## Setup

### 1) System dependencies

macOS:
```bash
brew install python@3.10 ffmpeg espeak-ng
```

Linux (Ubuntu/Debian):
```bash
sudo apt-get install python3.10 ffmpeg espeak-ng
```

### 2) Python dependencies

```bash
/opt/homebrew/bin/python3.10 -m pip install -r requirements.txt
```

### 3) Environment variables

```bash
# LLM + rendering
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
REPLICATE_API_TOKEN=r8_...

# Optional DashScope image edit
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_REGION=SINGAPORE
IMAGE_EDIT_MODEL=qwen-image-edit-max

# Deterministic template pipeline
USE_TEMPLATE_PIPELINE=true
TEMPLATE_PIPELINE_MODE=development   # or production
ALLOW_TEMPLATE_FALLBACK_TO_QWEN=false
ALLOW_DOWNSTREAM_SCENE_TYPER_FALLBACK=true
TEMPLATE_COMPOSITOR_BACKEND=pillow

# Scene typer resilience
SCENE_TYPER_REQUEST_TIMEOUT_SECONDS=120
SCENE_TYPER_MAX_RETRIES=2
SCENE_TYPER_RETRY_BACKOFF_SECONDS=1.5

# QC + publish integration
QEEG_ANALYSIS_DIR=../qEEG-analysis
QEEG_BACKEND_URL=http://127.0.0.1:8000
CLIPROXY_BASE_URL=http://127.0.0.1:8317
CLIPROXY_API_KEY=
```

## Run

```bash
./start.sh
# or
/opt/homebrew/bin/python3.10 -m streamlit run app.py
```

## TDD Recovery Loop

```bash
python3 scripts/tdd_loop.py status
python3 scripts/tdd_loop.py cycle

python3 -m pytest tests/unit tests/contract -q --maxfail=1
python3 -m pytest tests/execution -q

python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --provider openai --template-mode development --name integration_gate
python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --provider openai --template-mode production --name production_gate
```

Single manual stop in the loop: after Phase 2 template package generation for human approval.

## Image Operations (scope)
- Template authoring (Phase 2): generate/regenerate text-free Qwen templates.
- Emergency fallback remediation only: edit existing PNGs with Qwen image edit.
- Prompt refinement: authoring aid only; avoid for QC automation.
- Production data-slide runtime: deterministic template rendering path.

## QC + Publish (qEEG Council)

Run checks/publish from UI Step 3 or CLI:

```bash
python3.10 qc_publish.py --project 09-23-1982-0
python3.10 qc_publish.py --project 09-23-1982-0 --auto-fix-images
python3.10 qc_publish_batch.py
```

Behavior:
- Narrative QC uses Stage 4 consolidation as truth.
- Visual QC uses rendered PNG checks for wrong text/numbers.
- For rendered text errors, fix via image edit on existing PNG (no regenerate).

## Project Structure

```text
local-explainer-video/
├── app.py
├── core/
│   ├── director.py
│   ├── image_gen.py
│   ├── template_pipeline/
│   ├── voice_gen.py
│   └── video_assembly.py
├── prompts/
│   ├── director_system.txt
│   ├── scene_typer_system.txt
│   └── refiner_system.txt
├── templates/
│   ├── backgrounds/
│   ├── anchors/
│   └── manifest.json
├── scripts/
│   ├── run_template_pipeline_e2e.py
│   └── tdd_loop.py
└── projects/
```

## Troubleshooting
- If template render fails in production mode, inspect `projects/<project>/artifacts/template_render_audit.jsonl`.
- If E2E hangs or aborts, inspect `projects/<project>/artifacts/template_e2e_status.json`.
- If fallback happened, inspect `projects/<project>/artifacts/template_fallback_audit.jsonl`.
