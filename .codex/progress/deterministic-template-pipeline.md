# Deterministic Template Pipeline Progress (Repo Mirror)

Canonical memory log: [MEMORY.md](/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/MEMORY.md)

This file mirrors implementation milestones for repository-local visibility.

## 2026-02-22
- Kickoff complete.
- Context loaded from handoff + research + runtime code paths.
- Preparing deterministic pipeline scaffolding and feature-flag integration.
- Core deterministic modules landed (`core/template_pipeline/*`) with schema validation, downstream typing, template manifest selection, and compositor backends.
- Starter template assets generated under `templates/` with JSON anchors + manifest.
- Streamlit and scene rendering flow wired to deterministic mode toggle (`USE_TEMPLATE_PIPELINE`).
- QC publish path now supports deterministic skip mode (render+publish without visual/narrative QC checks).
- BAR phase-1 proof-of-concept generated for `09-05-1954-0` scene 004 with Pillow and Cairo outputs plus comparison strip/report.
  - Pillow MAD: `33.05`
  - Cairo MAD: `32.90`
- Validation run:
  - `python3 -m compileall app.py core qc_publish.py qc_publish_batch.py scripts`
  - deterministic render smoke via `core.image_gen.generate_scene_image()`
  - manifest + selector smoke for `roadmap_agenda`
  - full scene-type coverage smoke (`25` rendered types) at `projects/09-05-1954-0/poc_template_pipeline/smoke/`

## 2026-02-22 (Visual loop continuation)
- Re-rendered real-patient deterministic project:
  - `projects/09-05-1954-0__visual_loop_20260222_052430`
  - `16` scenes rendered via `core.template_pipeline.renderer.render_scene_to_image()`
  - updated contact sheet: `projects/09-05-1954-0__visual_loop_20260222_052430/contact_sheet_template_pipeline.png`
- Visual inspection executed on full-resolution outputs:
  - checked scenes `002, 003, 004, 006, 008, 010, 012, 013, 014` and contact sheet.
- Found and fixed summary-text quality regressions in generic panel templates:
  - removed schema-ish key noise (`label`, `id`, `kind`)
  - improved bullet phrasing for node/edge/session/value records
  - better formatting for `from -> to`, `label: value unit`, and `kind (severity)` shapes
  - file: `core/template_pipeline/compositor.py`
- Re-render after patch confirms improved readability with deterministic correctness maintained.

## 2026-02-22 (Handoff-for-next-agent)
- Added explicit transfer document after user flagged drift concern:
  - `/Users/davidmontgomery/local-explainer-video/HANDOFF_NEXT_AGENT_DETERMINISTIC_TEMPLATE_PIPELINE_2026-02-22.md`
- Document captures:
  - original requirements vs implemented status,
  - unresolved gaps and risks,
  - strict user constraints (including Qwen-curated template requirement and rejection of generic fallback visuals),
  - mandatory memory/progress update behavior linked to project-local `MEMORY.md`.

## 2026-02-22 (Recovery re-anchor: drift guard checklist)
- Added explicit repo contract:
  - `/Users/davidmontgomery/local-explainer-video/.codex/REPO_CONTRACT.md`
- Contract now locks canonical precedence:
  - primary: `HANDOFF-deterministic-template-pipeline.md`
  - secondary clarifications: `HANDOFF_NEXT_AGENT_DETERMINISTIC_TEMPLATE_PIPELINE_2026-02-22.md`
  - research docs marked advisory-only.
- Added drift guard checklist enforcement in runtime/test plan:
  - no production use of `generic_data_panel_v1`,
  - production mode hard-fails on uncovered archetypes,
  - fallback/audit artifacts required.

## 2026-02-22 (Recovery implementation: phase-0/1/3/4 scaffolding)
- Added continuous learning ledger + loop hooks:
  - `/Users/davidmontgomery/local-explainer-video/.codex/LEARNING.md`
  - `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/learning_log.py`
  - `/Users/davidmontgomery/local-explainer-video/scripts/tdd_loop.py`
- Manifest contract upgraded (`v1.1.0`) with required metadata:
  - `origin`, `operational_status`, `production_ready`, `selector_metadata`, curation fields.
  - `generic_data_panel_v1` explicitly set `dev_only` + not production-ready.
- Runtime hardening:
  - scene typer retry/backoff/error typing in `core/template_pipeline/scene_typer.py`
  - render failure classification + audit trail in `core/template_pipeline/renderer.py`
  - template fallback audit trail in `core/image_gen.py`
  - E2E terminal status artifact (`completed` / `failed`) in `scripts/run_template_pipeline_e2e.py`
- Director alignment (structured-first primary path) in `core/director.py` and `prompts/director_system.txt` with guarded downstream typer fallback.
- QC policy hardening:
  - if any scene uses emergency Qwen fallback, full QC is forced in `core/qc_publish.py`.
- Added recovery test suite (`55` tests passing across `tests/unit`, `tests/contract`, `tests/execution`).

## 2026-02-22 (Core rule enforcement: memory + learning updates)
- User raised process failure: memory updates were not consistently enforced in core repo rules.
- Added mandatory logging discipline sections to:
  - `/Users/davidmontgomery/local-explainer-video/AGENTS.md`
  - `/Users/davidmontgomery/local-explainer-video/CLAUDE.md`
- Embedded required log targets:
  - project memory index (`MEMORY.md`)
  - detailed progress memory note
  - repo progress mirror
  - learning ledger (`.codex/LEARNING.md`)
- Next action: continue phase-2 tooling implementation for Qwen template generation + approval packet workflow.

## 2026-02-22 (Rule/doc consistency normalization to latest handoff)
- User-reported contradiction in active instructions was resolved with repo-wide normalization.
- Active instruction files now aligned to strict handoff precedence and deterministic runtime policy:
  - `/Users/davidmontgomery/local-explainer-video/AGENTS.md`
  - `/Users/davidmontgomery/local-explainer-video/CLAUDE.md`
  - `/Users/davidmontgomery/local-explainer-video/README.md`
  - `/Users/davidmontgomery/local-explainer-video/prompts/director_system.txt`
- Legacy sources converted to archival/advisory only:
  - `/Users/davidmontgomery/local-explainer-video/PROJECT.md`
  - `/Users/davidmontgomery/local-explainer-video/AGENTS_override.md`
- Contract update:
  - `/Users/davidmontgomery/local-explainer-video/.codex/REPO_CONTRACT.md` includes `PROJECT.md` in advisory-only list.
- Validation:
  - targeted contradiction scan completed
  - `python3 -m pytest tests/contract/test_repo_contract_precedence.py tests/contract/test_manifest_contract.py -q` -> `3 passed`.
- Next: resume Phase 2 tooling implementation.

## 2026-02-22 (Executable loop + one-stop approval enforcement)
- Implemented executable multi-phase loop in `/Users/davidmontgomery/local-explainer-video/scripts/tdd_loop.py`.
- Added CLI controls:
  - `init-plan` (writes default phase plan)
  - `show-plan`
  - `run-plan` (runs phases and enforces single approval stop)
  - `set-approval` (writes approval artifact)
- One human stop behavior is now code-enforced at phase transition `2 -> 3` only.
- Added phase-2 tooling scripts:
  - `/Users/davidmontgomery/local-explainer-video/scripts/generate_qwen_templates.py`
  - `/Users/davidmontgomery/local-explainer-video/scripts/build_template_anchors.py`
  - `/Users/davidmontgomery/local-explainer-video/scripts/build_template_approval_packet.py`
- Added parity scorer scaffold:
  - `/Users/davidmontgomery/local-explainer-video/scripts/eval_parity.py`
- Added tests:
  - `/Users/davidmontgomery/local-explainer-video/tests/execution/test_tdd_loop_phase_runner.py`
  - `/Users/davidmontgomery/local-explainer-video/tests/execution/test_phase2_template_scripts.py`
- Validation:
  - full suite green: `61 passed`.

## 2026-02-22 (Systemic drift cleanup + Phase 1 stabilized)
- Normalized all scaffold-era contradictions to handoff contract:
  - `templates/manifest.json`: every `origin=scaffold_only` entry is now `dev_only` + not production-ready.
  - `scripts/bootstrap_template_assets.py`: scaffold defaults now `dev_only`.
  - Added guard test: `tests/contract/test_scaffold_templates_dev_only.py`.
- Removed gate fragility from plugin-dependent timeout flags:
  - `scripts/tdd_loop.py` now enforces native per-step command timeouts.
  - docs/commands updated to remove `pytest --timeout` dependency.
- Stabilized E2E provider determinism for loop gates:
  - `scripts/run_template_pipeline_e2e.py` default provider pinned to `openai`.
  - phase plan E2E gates in `scripts/tdd_loop.py` now pass `--provider openai`.
- Added schema compatibility fixes discovered by integration failures:
  - `target_band.low/high` accepted (normalized to `min/max`).
  - table row cells accept range lists (`[min,max]`).
  - coherence edges accept `pair` and map to `from/to`.
  - quality flags accept string list and coerce to structured flags.
  - dotplot sessions accept dict form and coerce to list form.
- Validation:
  - `python3 -m pytest tests -q` -> `70 passed`
  - dev E2E pass: `projects/09-05-1954-0__integration_gate_20260222_073928`
  - loop phase-1 pass: `python3 scripts/tdd_loop.py run-plan --target-phase 1` -> completed.
- Loop state now has no open blocker learnings for Phase 1.
- Phase 2 execution precondition still unmet locally: `REPLICATE_API_TOKEN` is not set.

## 2026-02-22 (`.env` load fix + Phase 2 pause reached)
- Fixed phase-2 token loading bug:
  - `/Users/davidmontgomery/local-explainer-video/scripts/generate_qwen_templates.py` now loads repo `.env` before live Replicate calls.
  - generation report includes `env_loaded` and `replicate_token_present` flags.
- Added regression test:
  - `/Users/davidmontgomery/local-explainer-video/tests/execution/test_phase2_template_scripts.py::test_generate_templates_loads_repo_dotenv_for_replicate_token`.
- Validation:
  - `python3 -m pytest tests -q` -> `71 passed`.
- Long loop resumed and reached exactly the required human stop:
  - status: `WAITING_TEMPLATE_APPROVAL` at phase 2.
  - approval artifact: `/Users/davidmontgomery/local-explainer-video/.codex/template_approval/approval.json` (`approved=false`).
  - packet: `/Users/davidmontgomery/local-explainer-video/artifacts/template_approval/20260222_075005/`.
