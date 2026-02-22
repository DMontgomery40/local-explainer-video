# Continuous Learning Ledger (`.codex/LEARNING.md`)

Purpose: Persistent, append-first learning ledger for deterministic TDD execution.

Hard rules:
- Every failure event must be logged before attempting a fix.
- Every resolved entry must include a regression test reference in absolute-path `::test_id` format.

## Entry Schema (Required Fields)
- ID (`L-YYYYMMDD-HHMMSS-<shorthash>`)
- Timestamp (UTC)
- Phase (`1..5`)
- Severity (`blocker|high|medium|low`)
- Type (`test_failure|runtime_error|policy_drift|quality_regression|timeout|external`)
- Signal
- Observed Error
- Root Cause
- Fix Applied
- Prevention Rule
- Regression Test Added (absolute path + test id)
- Artifacts
- Status (`OPEN|RESOLVED|REGRESSION`)
- Linked Commit/Change

## Active Guardrails
<!-- AUTO-GENERATED: updated from resolved high-severity learnings -->
- Keep command under automated loop coverage to prevent recurrence (from L-20260222-071706-b2aed8)
- Keep command under automated loop coverage to prevent recurrence (from L-20260222-071429-c9a69b)
- Every roadmap phase must map to runnable loop commands plus execution tests before marking phase tooling complete. (from L-20260222-071043-c2126a)
- When handoff precedence changes, update AGENTS.md, CLAUDE.md, README.md, and director_system.txt in one atomic consistency pass and re-run contradiction scan. (from L-20260222-065824-84dc05)

## Open Learnings
<!-- Entries with Status: OPEN -->

## Resolved Learnings
<!-- Entries with Status: RESOLVED -->

### L-20260222-071706-b2aed8
- Timestamp (UTC): 2026-02-22T07:17:06Z
- Phase: 1
- Severity: blocker
- Type: runtime_error
- Signal: python3 scripts/run_template_pipeline_e2e.py --source-project 09-05-1954-0 --template-mode development --name integration_gate
- Observed Error: Traceback (most recent call last): File "/Users/davidmontgomery/local-explainer-video/scripts/run_template_pipeline_e2e.py", line 242, in <module> raise SystemExit(main()) ^^^^^^ File "/Users/davidmontgomery/local-explainer-video/scripts/run_template_pipeline…
- Root Cause: phase-gate-integration-dev failure condition no longer present after code changes
- Fix Applied: phase-gate-integration-dev command now passes in TDD green step
- Prevention Rule: Keep command under automated loop coverage to prevent recurrence
- Regression Test Added: /Users/davidmontgomery/local-explainer-video/tests/execution/test_tdd_loop_learning_hooks.py::test_phase_gate_summary_written_on_gate_execution
- Artifacts: (none)
- Status: RESOLVED
- Linked Commit/Change: none
### L-20260222-071429-c9a69b
- Timestamp (UTC): 2026-02-22T07:14:29Z
- Phase: 1
- Severity: blocker
- Type: test_failure
- Signal: python3 -m pytest tests/unit tests/contract -q --maxfail=1 --timeout=120
- Observed Error: ERROR: usage: __main__.py [options] [file_or_dir] [file_or_dir] [...] __main__.py: error: unrecognized arguments: --timeout=120 inifile: None rootdir: /Users/davidmontgomery/local-explainer-video
- Root Cause: phase-gate-fast failure condition no longer present after code changes
- Fix Applied: phase-gate-fast command now passes in TDD green step
- Prevention Rule: Keep command under automated loop coverage to prevent recurrence
- Regression Test Added: /Users/davidmontgomery/local-explainer-video/tests/execution/test_tdd_loop_learning_hooks.py::test_phase_gate_summary_written_on_gate_execution
- Artifacts: (none)
- Status: RESOLVED
- Linked Commit/Change: none
### L-20260222-071043-c2126a
- Timestamp (UTC): 2026-02-22T07:10:43Z
- Phase: 2
- Severity: high
- Type: policy_drift
- Signal: tdd-loop-phase-runner-audit
- Observed Error: Loop had no executable phase runner enforcing single approval stop; phase-2 tooling was checklist-only.
- Root Cause: Phase intent was documented but not encoded as runnable orchestration and tested gate behavior.
- Fix Applied: Implemented run-plan/init-plan/set-approval commands, default 1..5 phase plan, phase-2-only approval gate, and phase-2 tooling scripts with execution tests.
- Prevention Rule: Every roadmap phase must map to runnable loop commands plus execution tests before marking phase tooling complete.
- Regression Test Added: /Users/davidmontgomery/local-explainer-video/tests/execution/test_tdd_loop_phase_runner.py::test_run_plan_stops_only_at_phase2_waiting_for_approval
- Artifacts: /Users/davidmontgomery/local-explainer-video/scripts/tdd_loop.py | /Users/davidmontgomery/local-explainer-video/scripts/generate_qwen_templates.py | /Users/davidmontgomery/local-explainer-video/scripts/build_template_anchors.py | /Users/davidmontgomery/local-explainer-video/scripts/build_template_approval_packet.py | /Users/davidmontgomery/local-explainer-video/tests/execution/test_tdd_loop_phase_runner.py | /Users/davidmontgomery/local-explainer-video/tests/execution/test_phase2_template_scripts.py
- Status: RESOLVED
- Linked Commit/Change: working-tree-loop-phase-runner-2026-02-22
### L-20260222-065824-84dc05
- Timestamp (UTC): 2026-02-22T06:58:24Z
- Phase: 1
- Severity: high
- Type: policy_drift
- Signal: doc-consistency-audit
- Observed Error: Active instruction docs mixed deterministic runtime rules with legacy per-scene generation language, causing agent-direction ambiguity.
- Root Cause: Legacy instruction blocks were still present in active rule docs and prompt policy after migration scope changed.
- Fix Applied: Rewrote active rule/docs to strict handoff precedence, scoped image operations to phase-2 authoring or emergency fallback, and archived conflicting legacy docs.
- Prevention Rule: When handoff precedence changes, update AGENTS.md, CLAUDE.md, README.md, and director_system.txt in one atomic consistency pass and re-run contradiction scan.
- Regression Test Added: /Users/davidmontgomery/local-explainer-video/tests/contract/test_repo_contract_precedence.py::test_repo_contract_declares_precedence_and_advisory_docs
- Artifacts: /Users/davidmontgomery/local-explainer-video/AGENTS.md | /Users/davidmontgomery/local-explainer-video/CLAUDE.md | /Users/davidmontgomery/local-explainer-video/README.md | /Users/davidmontgomery/local-explainer-video/prompts/director_system.txt | /Users/davidmontgomery/local-explainer-video/.codex/REPO_CONTRACT.md
- Status: RESOLVED
- Linked Commit/Change: working-tree-doc-normalization-2026-02-22
## Regression Learnings
<!-- Entries with Status: REGRESSION -->
