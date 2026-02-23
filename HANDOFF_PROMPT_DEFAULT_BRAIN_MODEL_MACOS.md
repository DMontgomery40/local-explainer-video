# Handoff Prompt: Build Default qEEG Brain Base Model (macOS Only)

Use this prompt verbatim with the next coding/Blender agent.

---

You are the implementation agent for the qEEG explainer pipeline in `/Users/davidmontgomery/local-explainer-video`.

## Mission
Build and productionize a **default Blender qEEG brain base model** that becomes the canonical visual foundation for all brain/electrode/coherence scenes. This base model must be deterministic, reusable, and optimized for rapid per-scene updates driven by structured data.

The director will route brain scenes to Blender and use the existing Blender skill path to drive incremental updates for new images.

## Hard Scope Rules
1. **macOS only** in this environment. Do not include Windows/WSL logic, docs, or troubleshooting.
2. Stay deterministic for clinical correctness:
- exact 10-20 electrode labels/placement
- exact patient numeric binding
- exact session/band/metric extraction usage
3. Style variation is allowed only for camera/lighting/palette/atmosphere and must not reduce label readability.
4. Do not introduce diffusion-model generation paths for this task.

## Grounding Inputs You Must Use
1. Visually inspect many existing patient brain slides under `projects/*/images/` to extract style direction for the base model (clinical-cinematic dark background, readable bright labels, clean focal hierarchy).
2. Use current Blender MCP capability + reliability evidence from upstream/community sources:
- GitHub repo: `ahujasid/blender-mcp`
- Tool surface in server implementation (`server.py`)
- Discussions and issues listed below (focus on practical production constraints)

## Required Sources (read before coding)
- https://github.com/ahujasid/blender-mcp
- https://raw.githubusercontent.com/ahujasid/blender-mcp/main/src/blender_mcp/server.py
- https://github.com/ahujasid/blender-mcp/discussions/18
- https://github.com/ahujasid/blender-mcp/discussions/19
- https://github.com/ahujasid/blender-mcp/discussions/158
- https://github.com/ahujasid/blender-mcp/issues/148
- https://github.com/ahujasid/blender-mcp/issues/152
- https://github.com/ahujasid/blender-mcp/issues/177
- https://github.com/ahujasid/blender-mcp/issues/189
- https://github.com/ahujasid/blender-mcp/issues/190
- https://github.com/ahujasid/blender-mcp/pull/192

## Existing Repo Context
- Blender path used here: `/Volumes/Blender/Blender.app/Contents/MacOS/Blender`
- Existing deterministic blender pipeline files:
  - `core/blender_gen.py`
  - `core/qeeg_extract.py`
  - `blender_pipeline/scripts/build_template.py`
  - `blender_pipeline/scripts/render_batch.py`
  - `blender_pipeline/assets/qeeg_template.blend`
  - `.agents/skills/blender-mcp-qeeg-runtime/*`

## Implementation Objectives
### 1) Canonical Base Scene
Create/upgrade `blender_pipeline/assets/qeeg_template.blend` so it includes:
- high-quality anatomical brain mesh (readable gyri/sulci, not toy-styled)
- deterministic electrode anchor empties for canonical 10-20 set used by this project
- deterministic coherence line system with stable materials and z-order/readability
- text-safe composition regions for title/subtitle/footer overlays
- lighting rigs and camera presets for consistent clinical-cinematic look

### 2) Style Presets Without Data Drift
Add style presets (e.g., `clinical_glow`, `calm_precision`, `focus_contrast`) as pure visual transforms:
- camera framing / orbit amplitude
- light colors/intensity/fog/bloom
- palette accents
No preset may change electrode identity, positions, or numeric value mapping.

### 3) Deterministic Update Contract
Implement an agent-friendly update contract so new scene requests can mutate the base model safely:
- single entrypoint per scene update from structured payload
- idempotent material/object upsert strategy (avoid node duplication drift)
- explicit naming conventions for all key objects/materials/collections
- checkpoint render/screenshot hooks for verification between major edits

### 4) Render Harness + Visual Regression
Create deterministic fixture renders and a lightweight regression harness:
- fixed sample specs (at least 3) for electrode map/coherence variants
- render outputs to a dedicated `_brain_basemodel_validation` folder
- produce contact sheets (or side-by-side comps) for quick visual diffs
- fail when required objects/material channels are missing

### 5) Director + Skill Alignment
Ensure director/skill-facing contract remains clear for ongoing image updates:
- preserve `render_backend: blender` + `[[BLENDER_QEEG]]` routing compatibility
- ensure required extraction hints are documented (`session_index`, `band`, `metric`)
- update `.agents/skills/blender-mcp-qeeg-runtime/references/` if contract changes

## Production Constraints
- Keep non-brain rendering paths untouched in this task.
- Keep all brain-scene data binding deterministic from patient data.
- Prefer stable geometry/material workflows over one-off artistic hacks.
- Keep output readable at 16:9 delivery resolution.

## Acceptance Criteria
1. Base `.blend` is deterministic, reusable, and visually production-grade.
2. Three fixture scene specs render with clearly different data-driven results while preserving style consistency.
3. Electrode labels are exact and consistently legible.
4. Coherence lines remain visible and non-chaotic across camera presets.
5. No platform logic outside macOS is added.
6. Tests/checks pass and artifacts are written with explicit paths.

## Deliverables
1. Code + asset changes in repo.
2. `BRAIN_BASEMODEL_IMPLEMENTATION_NOTES.md` with:
- architecture decisions
- object/material naming contract
- style preset definitions
- how to extend safely
3. `BRAIN_BASEMODEL_VALIDATION_REPORT.md` with:
- fixture specs used
- output artifact paths
- pass/fail against acceptance criteria
4. Updated skill reference docs if contract/presets changed.

## Execution Style
- Be strict and surgical.
- Prefer incremental commits with test runs.
- Surface blockers early with concrete remediation.
- Do not add unrelated refactors.

