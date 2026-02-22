# Repository Agents (local-explainer-video)

Active operational rules for AI agents in this repo.

## Canonical Precedence (strict)
1. `/Users/davidmontgomery/local-explainer-video/HANDOFF-deterministic-template-pipeline.md`
2. `/Users/davidmontgomery/local-explainer-video/HANDOFF_NEXT_AGENT_DETERMINISTIC_TEMPLATE_PIPELINE_2026-02-22.md`
3. `/Users/davidmontgomery/local-explainer-video/.codex/REPO_CONTRACT.md`
4. This file and `/Users/davidmontgomery/local-explainer-video/CLAUDE.md`

Advisory only (never overrides the above):
- `/Users/davidmontgomery/local-explainer-video/remotion-migration-report-v2.md`
- `/Users/davidmontgomery/local-explainer-video/AGENTS_override.md`
- `/Users/davidmontgomery/local-explainer-video/PROJECT.md`

## Current Objective (Recovery)
- Recover to deterministic template pipeline behavior from the handoff docs.
- Data-bearing slides in production must use template selection + deterministic compositor.
- `generic_data_panel_v1` is development-only scaffolding.
- Any `origin=scaffold_only` template is development-only until replaced/approved as `qwen_curated`.
- Production mode must hard-fail on missing archetype coverage.
- Runtime Qwen generation is emergency-only fallback, auditable, and QC-gated.

## Runtime Contract
- `projects/<project>/plan.json` is source of truth for narration, `scene_type`, `structured_data`, and render metadata.
- `visual_prompt` is optional and non-authoritative for deterministic data-slide rendering.
- Title/roadmap and all data-bearing slides are expected to render via templates + anchors.

## Image Operations (Scoped)
- Template authoring (Phase 2 only):
  - Generate/Regenerate image = create text-free Qwen template assets for the template library.
- Emergency fallback QC remediation only:
  - Edit image = surgical correction on existing PNG when fallback path is used.
- Prompt refinement:
  - Allowed for template authoring iteration.
  - Not a production data-slide rendering mechanism.
  - Avoid in QC automation.

## Long-Running TDD Loop
- Execute phased recovery with `scripts/tdd_loop.py` and `/Users/davidmontgomery/local-explainer-video/.codex/TODO.md`.
- Single human stop only after Phase 2 Qwen template package is produced for approval.
- Do not introduce extra user-input pauses in the loop.

## Memory + Learning Discipline (mandatory)
- Update index on active work:
  - `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/MEMORY.md`
- Update detailed progress at each milestone/blocker/handoff:
  - `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/memory/deterministic-template-pipeline-progress.md`
- Keep repo progress mirror in sync:
  - `/Users/davidmontgomery/local-explainer-video/.codex/progress/deterministic-template-pipeline.md`
- Record every failure/fix in learning ledger:
  - `/Users/davidmontgomery/local-explainer-video/.codex/LEARNING.md`
- Do not claim completion unless logs contain:
  - commands executed
  - artifact paths
  - pass/fail outcomes
  - exact next action

## QC + Publish (qEEG Council integration)
If `../qEEG-analysis` exists:
- Narrative ground truth: Stage 4 consolidation
- Numeric ground truth: Stage 1 `_data_pack.json`
- Narrative judge: Claude Opus 4.6
- Visual judge: Gemini vision via CLIProxyAPI
- For text issues in rendered slide PNGs, use image edit (no regenerate).
- Deterministic template scenes can skip routine visual QC unless emergency fallback occurred.

## Quick Commands
- Run app: `./start.sh`
- Manual: `/opt/homebrew/bin/python3.10 -m streamlit run app.py`
- Loop status: `python3 scripts/tdd_loop.py status`
- Fast gate: `python3 -m pytest tests/unit tests/contract -q --maxfail=1`
- Execution gate: `python3 -m pytest tests/execution -q`
- E2E (dev): `python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --provider openai --template-mode development --name integration_gate`
- E2E (prod): `python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --provider openai --template-mode production --name production_gate`
