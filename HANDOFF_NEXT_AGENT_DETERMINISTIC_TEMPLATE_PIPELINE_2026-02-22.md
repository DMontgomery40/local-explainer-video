# Handoff Prompt For Next Agent: Deterministic Template Pipeline (No Drift Version)

## 0) Read This First
You are taking over an in-progress re-architecture in `/Users/davidmontgomery/local-explainer-video`.

This handoff exists because the implementation has partially progressed but has also drifted from explicit product requirements.

Your job is to **finish the architecture exactly as specified by David’s original handoff**, while honoring all additional clarifications the user provided during implementation.

Do not treat this as greenfield. You are continuing a partially integrated codebase with active edits.

---

## 1) Authoritative Source Documents (Must Read In This Exact Order)
1. Original implementation brief:
- `/Users/davidmontgomery/local-explainer-video/HANDOFF-deterministic-template-pipeline.md`

2. Deep research report:
- `/Users/davidmontgomery/local-explainer-video/.codex/research/remotion_hybrid_deep_research_2026-02-22.md`

3. Scene archetype corpus map (all scenes):
- `/Users/davidmontgomery/local-explainer-video/.codex/research/data/local_explainer_scene_archetypes_manual.csv`

4. Project architecture/runtime conventions:
- `/Users/davidmontgomery/local-explainer-video/CLAUDE.md`
- `/Users/davidmontgomery/local-explainer-video/AGENTS.md`

5. Current director prompt and implementation:
- `/Users/davidmontgomery/local-explainer-video/prompts/director_system.txt`
- `/Users/davidmontgomery/local-explainer-video/core/director.py`

6. Rendering/generation entrypoints:
- `/Users/davidmontgomery/local-explainer-video/core/image_gen.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/renderer.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/compositor.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/scene_schemas.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/selector.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/manifest.py`
- `/Users/davidmontgomery/local-explainer-video/templates/manifest.json`

7. UI and QC integration points:
- `/Users/davidmontgomery/local-explainer-video/app.py`
- `/Users/davidmontgomery/local-explainer-video/core/qc_publish.py`
- `/Users/davidmontgomery/local-explainer-video/qc_publish.py`
- `/Users/davidmontgomery/local-explainer-video/qc_publish_batch.py`

8. Helper scripts added during this effort:
- `/Users/davidmontgomery/local-explainer-video/scripts/bootstrap_template_assets.py`
- `/Users/davidmontgomery/local-explainer-video/scripts/migrate_plan_to_template_schema.py`
- `/Users/davidmontgomery/local-explainer-video/scripts/template_pipeline_poc.py`
- `/Users/davidmontgomery/local-explainer-video/scripts/run_template_pipeline_e2e.py`

9. Memory/progress logs:
- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/MEMORY.md`
- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/memory/deterministic-template-pipeline-progress.md`
- `/Users/davidmontgomery/local-explainer-video/.codex/progress/deterministic-template-pipeline.md`

---

## 2) Hard Non-Negotiables (User Clarifications You Must Follow)
These are explicit user decisions and override prior ambiguity.

1. **Do not ship generic fallback visuals.**
- User explicitly rejected generic panel fallback for production.
- If a scene is not covered by a proper archetype-specific template, that is unfinished work, not acceptable output.

2. **Visual quality must match original Qwen look.**
- Requirement is “must look like the originals,” with hallucinations removed.

3. **Template assets must be Qwen-generated curated images.**
- User explicitly corrected drift: agents were never instructed to create homegrown synthetic templates.
- Placeholder templates are scaffolding only and must not be treated as final production assets.

4. **Downstream typing architecture is acceptable; caching-only is not.**
- New patients must be processable from fresh input.
- Do not design around cached plans as a primary strategy.

5. **Functionality hardening is priority before aesthetic expansion.**
- Make runtime robust and deterministic first.

6. **`summary` mode was not part of original plan.**
- Treat summary fallback rendering as temporary/internal, not final architecture.

7. **Qwen fallback is emergency-only.**
- If used, QC/edit loop must handle spelling/text/data-placement errors.
- End goal remains eliminating runtime Qwen dependency for normal path.

8. **Patient ID schema is already robust and should not be reworked.**

9. **Skills and subagents should be used where appropriate; do not avoid them out of process overhead.**

---

## 3) What Was Originally Asked vs What Is Actually Implemented

### A) Requested (Original Handoff Scope)
1. Build deterministic 3-layer architecture with curated template library, deterministic compositor, selector/orchestration.
2. Cover **22 data-bearing archetypes** (+ atmospheric handling strategy), with variant strategy for cardinality and magnitude.
3. Generate **~60-120 text-free Qwen templates**, curate exemplars visually across corpus, create anchor JSONs and manifest.
4. Modify director path to emit `scene_type + structured_data` with schema validation.
5. Build deterministic selector (heuristic first), optionally subagent fallback.
6. Demonstrate compositor text-quality parity (evaluate ≥2 rendering options).
7. Integrate Streamlit storyboard so what user sees is final deterministic composite.
8. Simplify QC for deterministic slides (text correctness guaranteed by construction).
9. Run real patient end-to-end and visually inspect output (not metadata).

### B) Implemented So Far (Current State)
1. Deterministic framework exists in code:
- `core/template_pipeline/*` added (schemas, scene typer, manifest loader, selector, compositor, renderer).

2. Director path modified via downstream typing pass:
- `core/director.py` calls scene typer when template mode enabled.
- `prompts/scene_typer_system.txt` introduced.

3. Image generation path integrated:
- `core/image_gen.py` routes typed scenes to deterministic renderer first.
- optional fallback to legacy Qwen generation if deterministic render fails (env-controlled).

4. Streamlit integration added:
- template mode toggle
- scene type editor
- structured data editor
- auto-type button
- deterministic scene messaging

5. QC integration changed:
- `qc_publish` can skip narrative+visual checks when template mode is active and just render/publish.

6. POC and E2E scripts added:
- BAR POC script and visual comparison helper
- migration script for existing plans
- E2E deterministic runner script

7. Real patient visual loop was run and visually inspected:
- Primary verified project:
  - `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0__visual_loop_20260222_052430`
- Contact sheet:
  - `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0__visual_loop_20260222_052430/contact_sheet_template_pipeline.png`
- Several full-resolution scenes visually inspected repeatedly.

8. Compositor summary formatter improved during loop:
- Better bullet phrasing, wrapping, key cleanup, from->to formatting.
- Still not final architecture intent.

### C) Critical Gaps / Drift (Must Be Corrected)
1. **Template origin drift:**
- Current template backgrounds were generated by local drawing script:
  - `/Users/davidmontgomery/local-explainer-video/scripts/bootstrap_template_assets.py`
- This violates explicit requirement for Qwen-generated curated templates.

2. **Template coverage gap:**
- Manifest includes only a few specific templates + broad generic fallback:
  - `/Users/davidmontgomery/local-explainer-video/templates/manifest.json`
- Most archetypes collapse to `generic_data_panel_v1`, which is unacceptable for production and rejected by user.

3. **Aesthetic parity gap:**
- Specialized templates (`bar`, `split`) look materially stronger.
- Generic fallback slides do not match cinematic quality of original Qwen output.

4. **Architecture drift via summary fallback:**
- `summary` binding mode in compositor became heavy fallback behavior.
- User explicitly said summary mode was never part of intended plan.

5. **Functional robustness still incomplete:**
- Fresh E2E attempt (`visual_loop2`) hung and produced empty folder:
  - `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0__visual_loop2_20260222_053107`
- Need reliable long-run deterministic E2E behavior under real inputs.

6. **Director prompt mismatch remains:**
- Original director prompt (`prompts/director_system.txt`) still prose-first for visual prompts.
- Current typed path is downstream post-process, not native structured-first generation.

---

## 4) Exact Current File State You Must Assume

### New/modified deterministic pipeline files
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/__init__.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/scene_schemas.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/scene_typer.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/manifest.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/selector.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/compositor.py`
- `/Users/davidmontgomery/local-explainer-video/core/template_pipeline/renderer.py`

### Integration files
- `/Users/davidmontgomery/local-explainer-video/core/director.py`
- `/Users/davidmontgomery/local-explainer-video/core/image_gen.py`
- `/Users/davidmontgomery/local-explainer-video/app.py`
- `/Users/davidmontgomery/local-explainer-video/core/qc_publish.py`
- `/Users/davidmontgomery/local-explainer-video/qc_publish.py`
- `/Users/davidmontgomery/local-explainer-video/qc_publish_batch.py`

### Templates and manifest currently in repo (scaffolding)
- `/Users/davidmontgomery/local-explainer-video/templates/backgrounds/*.png`
- `/Users/davidmontgomery/local-explainer-video/templates/anchors/*.json`
- `/Users/davidmontgomery/local-explainer-video/templates/manifest.json`

### Progress/memory references
- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/MEMORY.md`
- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/memory/deterministic-template-pipeline-progress.md`
- `/Users/davidmontgomery/local-explainer-video/.codex/progress/deterministic-template-pipeline.md`

---

## 5) Required Next Work (Priority Ordered)

### Priority 0: Re-anchor to spec (no new architecture drift)
1. Preserve deterministic pipeline scaffolding and integration work already done.
2. Remove assumption that generated scaffolding templates are acceptable production assets.
3. Treat `generic_data_panel_v1` and heavy summary fallback as temporary stopgaps only.

### Priority 1: Functionality hardening (user requested first)
1. Stabilize deterministic E2E run path for real patient input.
2. Ensure scene typing + validation is resilient on real long reports.
3. Add robust retry/timeout/error surface for downstream typing step.
4. Ensure fallback behavior is explicit and auditable (emergency-only policy).
5. Verify end-to-end rendering/audio/video assembly for at least one real patient run from raw input text.

### Priority 2: Replace synthetic templates with Qwen-curated template library
1. Curate visual exemplars from actual patient outputs (as required in original plan).
2. Generate text-free templates via Qwen (no text/numbers/labels) for each archetype/variant.
3. Replace scaffold backgrounds with curated Qwen assets.
4. Rebuild anchors against those actual templates (anchor after template generation, not before).

### Priority 3: Remove production dependency on generic fallback
1. Expand manifest coverage with archetype-specific template variants.
2. Tighten selector rules beyond simple count matching (zones, density, trends, etc.).
3. Gate production mode so uncovered archetypes fail loudly instead of silently mapping to generic fallback.

### Priority 4: Policy-conformant QC behavior
1. Deterministic mode should not require routine QC for data slides.
2. If emergency fallback to Qwen is used, QC+image-edit loop must catch/fix text/data placement errors.
3. Do not regress legacy QC behavior for non-template path.

---

## 6) Explicit Operational Rules For This Handoff

1. **Do not claim completion until a real patient run is executed and visually inspected scene-by-scene.**
2. Visual inspection must use actual images, not metadata summaries.
3. If outputs look “schema dump”-like or generic, treat as failure even when text is technically correct.
4. Maintain feature flags and avoid breaking legacy flow during migration.
5. Keep Python 3.10 compatibility for Kokoro.

---

## 7) Memory/Progress Discipline (Mandatory)
At start of work and at every major milestone, update project-local memory logs.

### Required files
- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/MEMORY.md`
- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-local-explainer-video/memory/deterministic-template-pipeline-progress.md`
- `/Users/davidmontgomery/local-explainer-video/.codex/progress/deterministic-template-pipeline.md`

### Required behavior
1. Every update entry must include backlink to `MEMORY.md`.
2. On compaction/resume, re-open `MEMORY.md` and all linked docs before continuing.
3. Log: command run, artifact paths produced, visual QA findings, and exact next action.
4. Never leave a run in ambiguous state without recording what is done vs pending.

---

## 8) User Q&A Record (Authoritative Product Decisions)

### Earlier decisions from user (numbered answers in-thread)
1. Template names should be friendly and descriptive so an LLM can pick among them.
2. For undecided design choices, agent recommendation is acceptable.
3. Scene-typing decisioning should be downstream-capable; director can launch subagents.
4. The strategic goal is eventually no runtime Qwen usage if deterministic system works.
5. QC exists primarily to catch Qwen mistakes.
6. Patient ID schema is robust as-is.
7. Skills/subagents are expected and should be used.

### Later corrective answers (after drift concern)
1. Generic fallback look is unacceptable in client setting.
2. Output must look like original Qwen visuals.
3. Qwen fallback may exist only for emergencies; QC must catch/fix text and placement errors there.
4. New patients must be processable from fresh data (not cache-dependent architecture).
5. Harden functionality first.
6. Summary mode was not part of intended production design.
7. Templates were intended to be Qwen-created curated assets, not self-generated by agent code.

---

## 9) Known Artifacts From This Iteration

### Real-patient deterministic visual loop project
- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0__visual_loop_20260222_052430`
- contains `plan.json`, rendered `images/scene_*.png`, and contact sheet.

### Empty/hung run artifact
- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0__visual_loop2_20260222_053107`
- created during E2E attempt that did not complete.

---

## 10) First Action Checklist For You (Next Agent)
1. Read all docs in Section 1.
2. Confirm understanding of non-negotiables in Section 2 (especially Qwen-template requirement).
3. Audit current manifest coverage and enumerate archetypes still routed to generic fallback.
4. Stabilize real-patient E2E run from raw input text (functionality hardening first).
5. Begin replacing scaffold templates with Qwen-curated assets for highest-frequency archetypes.
6. Re-run one real patient end-to-end and visually inspect output scene-by-scene.
7. Update all memory/progress files with paths and findings.

