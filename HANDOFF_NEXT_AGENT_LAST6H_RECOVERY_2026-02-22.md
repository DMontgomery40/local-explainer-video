# HANDOFF NEXT AGENT (LAST 6 HOURS RECOVERY)

## 0) Read this first
This handoff is for continuing work from a recovery state after branch/worktree confusion and partial doc/memory corruption.

Primary goal now:
- Preserve and continue the **actual last-6-hours Blender/qEEG implementation**.
- Recreate/replace missing non-brain template-pack implementation (source missing in this branch).
- Keep clinical determinism constraints enforced.

## 1) Current branch and commit (authoritative)
- Branch: `recovery/last-6-hours-20260223-2238`
- HEAD commit: `a0932ff`
- Base at branch creation: `9f092ce`
- Working tree status at handoff creation:
  - clean except untracked `.claude/` mirror (do not commit `.claude/`).

## 2) What is confirmed present right now
### 2.1 Blender/qEEG core (present and tested)
- `core/visual_gen.py`
- `core/blender_gen.py`
- `core/qeeg_extract.py`
- `core/qeeg_data.py`
- `core/qc_video.py`
- `blender_pipeline/scripts/build_template.py`
- `blender_pipeline/scripts/render_batch.py`
- `blender_pipeline/assets/qeeg_template.blend`
- `blender_pipeline/assets/montage/standard_1020.json`
- `blender_pipeline/assets/fonts/NotoSans-Regular.ttf`

### 2.2 Director and skill assets
- Director side:
  - `core/director.py`
  - `prompts/director_blender_skill.txt`
- Skill folder (exists, untracked unless added on branch):
  - `.agents/skills/blender-mcp-qeeg-runtime/SKILL.md`
  - `.agents/skills/blender-mcp-qeeg-runtime/agents/openai.yaml`
  - `.agents/skills/blender-mcp-qeeg-runtime/references/qeeg-runtime-contract.md`
  - `.agents/skills/blender-mcp-qeeg-runtime/references/mcp-capabilities.md`
  - `.agents/skills/blender-mcp-qeeg-runtime/references/style-recipes.md`

### 2.3 Unit tests currently present
- `tests/unit/test_visual_gen_backend_selection.py`
- `tests/unit/test_qeeg_extract_heuristics.py`
- `tests/unit/test_qeeg_data_loader.py`
- `tests/unit/test_qc_video_sampling.py`
- `tests/unit/test_director_blender_fields.py`

### 2.4 Verified test status
Command run:
```bash
python3.10 -m pytest -q \
  tests/unit/test_visual_gen_backend_selection.py \
  tests/unit/test_qeeg_extract_heuristics.py \
  tests/unit/test_qeeg_data_loader.py \
  tests/unit/test_qc_video_sampling.py \
  tests/unit/test_director_blender_fields.py
```
Result: `19 passed`.

## 3) What is missing right now (critical)
The following source trees are missing `.py` and only have stale `__pycache__` bytecode in this branch:
- `core/template_pipeline/` (source missing; only pyc)
- `scripts/` template/TDD scripts (source missing; only pyc)
- `tests/contract/` (source missing; only pyc)
- `tests/execution/` (source missing; only pyc)
- `tests/unit/template_pipeline/` (source missing; only pyc)

Also missing as source assets/docs:
- `templates/manifest.json`
- `templates/anchors/*`
- `templates/backgrounds/*`
- `templates/README.md`
- historical handoff/docs from earlier deterministic template phase (not present in this branch)

Implication:
- non-brain deterministic template-pack pipeline is **not runnable from source** in this branch.

## 4) High-risk confusion to avoid
1. Do **not** switch back to `main` or other worktrees unless explicitly told.
2. Do **not** assume pyc-only modules are sufficient deliverables.
3. Do **not** commit `.claude/` mirror artifacts.
4. Do **not** rewrite policy docs aggressively again during implementation.

## 5) Product/technical intent to preserve
- Hard deterministic constraints:
  - exact spelling/labels
  - exact patient numeric values
  - exact EEG node placement on brain scenes
- Routing intent:
  - brain/electrode/coherence scenes -> Blender
  - non-brain chart/panel scenes -> curated template-pack renderer
- User accepts stylistic variation for brain scenes if above constraints hold.

## 6) Current architecture reality
- `core/visual_gen.py` still falls back to `core/image_gen.generate_scene_image(...)` for non-Blender scenes.
- `core/image_gen.py` still contains Replicate/DashScope/Qwen/Imagen generation/edit code.
- Therefore policy and runtime are currently mismatched for non-brain scenes until template path is restored/rebuilt.

## 7) Immediate next-agent mission (decision-complete)
### Step A: stabilize this branch state
- Ensure `.agents/skills/blender-mcp-qeeg-runtime/*` is tracked in branch commits.
- Ensure `.claude/` remains untracked.
- Keep current Blender/qEEG code exactly as baseline.

### Step B: recover non-brain renderer path (source)
Choose one explicit approach and stick to it:
1) **Recreate template pipeline source in this branch** using current requirements and pyc-informed behavior, OR
2) **Port only minimal deterministic template renderer** needed for non-brain scenes (title/roadmap/kpi/split/trend/wave panels) and delete pyc-only ghost dirs.

Given current state, option (2) is safer/faster and avoids pretending unrecoverable source exists.

### Step C: enforce routing in code (not docs)
- Update scene routing so non-brain scenes do **not** hit runtime Qwen generation in production mode.
- Keep an explicit fallback switch only for emergency/debug (off by default).

### Step D: align docs/rules to actual behavior
- Make AGENTS/CLAUDE/README match real runtime after Step B/C.
- Remove references to nonexistent `core/render.py`/`qeeg-blender` path if not actually used in this repo.

### Step E: add/restore tests for routing contract
Add tests for:
- Blender path trigger conditions
- Non-brain scenes choosing template path
- No accidental runtime AI image generation when production policy is active

### Step F: produce one proof run
- Run end-to-end on one known project with mixed scene types.
- Provide artifact list and explicit pass/fail summary.

## 8) Suggested acceptance criteria before merge
- Blender/qEEG tests pass (existing 19 + new routing tests).
- Non-brain scenes render without hitting Replicate/Qwen runtime generation in production mode.
- Docs match implementation (no stale architecture claims).
- Branch contains complete source for all active runtime paths (no pyc-only ghost modules).

## 9) Quick verification commands
```bash
# confirm branch and baseline
 git rev-parse --abbrev-ref HEAD
 git rev-parse --short HEAD
 git status --short

# confirm missing/source status
 find core/template_pipeline scripts tests/contract tests/execution tests/unit/template_pipeline templates -maxdepth 3 -type f

# confirm skill presence
 find .agents/skills/blender-mcp-qeeg-runtime -maxdepth 3 -type f

# baseline tests
 python3.10 -m pytest -q \
   tests/unit/test_visual_gen_backend_selection.py \
   tests/unit/test_qeeg_extract_heuristics.py \
   tests/unit/test_qeeg_data_loader.py \
   tests/unit/test_qc_video_sampling.py \
   tests/unit/test_director_blender_fields.py
```

## 10) Context note
This handoff was authored after emergency preservation of last-6-hours code into `a0932ff` and rollback of accidental older-tree backfill. Treat this branch as the canonical continuation point.
