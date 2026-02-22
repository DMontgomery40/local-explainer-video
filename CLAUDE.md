# CLAUDE.md

Operational guidance for coding agents working in this repository.

## Canonical Precedence (strict)
1. `/Users/davidmontgomery/local-explainer-video/HANDOFF-deterministic-template-pipeline.md`
2. `/Users/davidmontgomery/local-explainer-video/HANDOFF_NEXT_AGENT_DETERMINISTIC_TEMPLATE_PIPELINE_2026-02-22.md`
3. `/Users/davidmontgomery/local-explainer-video/.codex/REPO_CONTRACT.md`
4. Runtime contracts in code/tests under `core/template_pipeline/*` and `tests/contract/*`

If any older document conflicts, treat it as non-authoritative.

## Required Logging Discipline (Core Rule)
All active implementation work must update memory and learning logs.

- Index:
  - `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/MEMORY.md`
- Detailed progress:
  - `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/memory/deterministic-template-pipeline-progress.md`
- Repo mirror progress:
  - `/Users/davidmontgomery/local-explainer-video/.codex/progress/deterministic-template-pipeline.md`
- Failure/fix ledger:
  - `/Users/davidmontgomery/local-explainer-video/.codex/LEARNING.md`

Minimum log fields per milestone:
- commands executed
- output artifact paths
- pass/fail outcome
- next action

## Recovery Objective
- Re-anchor implementation to deterministic template pipeline behavior.
- Preserve useful scaffolding, remove production-risk shortcuts.
- Complete phased TDD execution through phases 1-5 with drift prevention.
- Single human approval stop only after Phase 2 Qwen template package.

## Architecture Contract

Default runtime path:
- Input text
- Director output (`scene_type + structured_data` first-class)
- Template selector
- Deterministic compositor
- Video assembly

Hard rules:
- Data-bearing slides in production use deterministic template rendering.
- `generic_data_panel_v1` is development-only and never valid production fallback.
- `origin=scaffold_only` templates stay `dev_only` until promoted as curated/approved assets.
- Production mode hard-fails on uncovered archetypes.
- Runtime Qwen generation is emergency-only fallback, auditable, and QC-gated.
- Downstream scene-typer fallback is migration-only and explicitly controllable.

## `plan.json` Contract
Each scene should carry:
- `scene_type`
- `structured_data` (schema-valid)
- render metadata after rendering:
  - `template_id`
  - `render_mode`
  - `render_backend`

Meta flags remain authoritative:
- `meta.use_template_pipeline`
- `meta.render_pipeline`

## Image Operations (Scoped)
- Template authoring (Phase 2): use generation to create text-free Qwen template assets.
- Emergency fallback remediation: use image edit on existing PNGs.
- Prompt refinement is not the production data-slide rendering mechanism and should be avoided for QC automation.

## Commands

```bash
# Run app
./start.sh
/opt/homebrew/bin/python3.10 -m streamlit run app.py

# Install deps
/opt/homebrew/bin/python3.10 -m pip install -r requirements.txt

# TDD loop
python3 scripts/tdd_loop.py status
python3 scripts/tdd_loop.py cycle

# Gates
python3 -m pytest tests/unit tests/contract -q --maxfail=1
python3 -m pytest tests/execution -q
python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --provider openai --template-mode development --name integration_gate
python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --provider openai --template-mode production --name production_gate

# QC + publish
python3.10 qc_publish.py --project 09-23-1982-0
python3.10 qc_publish.py --project 09-23-1982-0 --auto-fix-images
python3.10 qc_publish_batch.py
```

## Environment Variables (key)
- `USE_TEMPLATE_PIPELINE=true`
- `TEMPLATE_PIPELINE_MODE=development|production`
- `ALLOW_TEMPLATE_FALLBACK_TO_QWEN=false` (default; emergency-only)
- `ALLOW_DOWNSTREAM_SCENE_TYPER_FALLBACK=true|false`
- `TEMPLATE_COMPOSITOR_BACKEND=pillow|cairo`
- `SCENE_TYPER_REQUEST_TIMEOUT_SECONDS`
- `SCENE_TYPER_MAX_RETRIES`
- `SCENE_TYPER_RETRY_BACKOFF_SECONDS`

QC integration:
- `QEEG_ANALYSIS_DIR`
- `QEEG_BACKEND_URL`
- `CLIPROXY_BASE_URL`
- `CLIPROXY_API_KEY`
- `ANTHROPIC_API_KEY`
- `REPLICATE_API_TOKEN`

## Quality Gates
- Do not advance phase if blocker learnings are open.
- Every gate failure must produce a learning entry.
- Every resolved issue must cite regression test coverage.
- Production parity gate target confidence: `>= 84.7` using project-defined weighted metrics.
