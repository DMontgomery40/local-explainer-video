# Deterministic Template Recovery TODO

Source of truth for execution order: `/Users/davidmontgomery/local-explainer-video/.codex/REPO_CONTRACT.md`

## Loop Rules
- Use `scripts/tdd_loop.py` as the execution coordinator.
- Log every failure/fix in `/Users/davidmontgomery/local-explainer-video/.codex/LEARNING.md`.
- Do not advance phase with unresolved blocker learnings.
- Single human stop only after Phase 2 template generation package is ready.
- Human stop mechanics:
  - Phase 2 writes pending approval artifact to `/Users/davidmontgomery/local-explainer-video/.codex/template_approval/approval.json`.
  - Loop blocks at phase advance `2 -> 3` until `approved=true`.
  - Resume with `python3 scripts/tdd_loop.py run-plan`.

## Gate Commands
- Fast gate:
  - `python3 -m pytest tests/unit tests/contract -q --maxfail=1`
- Medium gate:
  - `python3 -m pytest tests/execution -q`
- Integration gate:
  - `python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --provider openai --template-mode development --name integration_gate`
- Production integration gate:
  - `python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --provider openai --template-mode production --name production_gate`
- Initialize executable loop plan:
  - `python3 scripts/tdd_loop.py init-plan --force`
- Execute loop until block/completion:
  - `python3 scripts/tdd_loop.py run-plan`
- Approve Phase 2 packet and resume:
  - `python3 scripts/tdd_loop.py set-approval --approved true --approved-by human`
  - `python3 scripts/tdd_loop.py run-plan`

## Phase Checklist

### Phase 1 — Functionality hardening
- [x] Scene typer retry/backoff/error typing.
- [x] Render failure classification + audit artifact.
- [x] Template fallback audit artifact.
- [x] E2E status artifact (`running/completed/failed`).
- [x] Fast/medium/execution tests green.

### Phase 2 — Qwen template generation + approval packet (single human stop)
- [x] Add `scripts/generate_qwen_templates.py` with dry-run and live modes.
- [x] Add `scripts/build_template_anchors.py` (post-template anchor rebuild workflow).
- [x] Add `scripts/build_template_approval_packet.py` (contact sheets + manifest summary + gaps).
- [x] Add tests for phase-2 scripts.
- [ ] Run template generation for prioritized archetypes.
- [ ] Produce approval packet under `artifacts/template_approval/<timestamp>/`.
- [ ] Pause for human approval only here.

### Phase 3 — Remove production dependency on generic fallback
- [ ] Expand manifest to production-ready coverage for all 22 data-bearing archetypes.
- [ ] Ensure `generic_data_panel_v1` remains `dev_only` forever.
- [ ] Production-mode E2E runs with zero generic fallback.

### Phase 4 — Director structured-first path
- [ ] Validate director native structured output path as default in template mode.
- [ ] Keep downstream scene typer as guarded migration fallback only.
- [ ] Add/green director contract + integration tests.

### Phase 5 — QC/publish and parity scoring gate
- [x] Add parity evaluator script (`scripts/eval_parity.py`).
- [ ] Implement weighted confidence scoring gate >= `84.7`.
- [ ] Add runtime 5–7 minute guard test.
- [ ] Final production E2E + QC policy validation complete.

## Current Focus
- Phase 2 live template generation, approval packet review, and loop resume into Phase 3+.
