# Handoff Prompt For Next Model: Visual/Artistic Parity Redo (Blender qEEG Brain)

Use this prompt verbatim with the next model.

---

You are taking over the Blender visual design/parity effort in:
`/Users/davidmontgomery/local-explainer-video`

Your focus is **visual parity and artistic quality** for qEEG brain scenes, while preserving deterministic clinical data binding.

## Primary Goal
Achieve strong visual parity with the provided reference look (user-supplied target images) while keeping deterministic EEG data integrity.

The current implementation is technically wired and deterministic, but visual output is not acceptable. Your mission is to replace the visual quality layer and art direction outcomes without breaking deterministic correctness.

## Critical Context
- Platform scope: **macOS only** in this environment.
- Branch: `recovery/last-6-hours-20260223-2238`
- Recent Blender-focused commit: `a1963b1`
- Determinism constraints remain non-negotiable:
  - exact data values from patient data
  - exact electrode labels
  - reproducible mapping/render behavior

## Why This Is Being Reassigned
The current visuals are far from parity with the desired aesthetic. The user has explicitly rejected current quality as significantly below target across subjective and objective quality dimensions.

## Target References (what you should match)
Reference targets were provided by user in chat (4 images) and depict:
- polished, anatomically recognizable brain form
- strong cinematic-medical lighting
- coherent composition and typographic hierarchy
- convincing depth/material treatment
- EEG markers integrated in believable relation to the brain form

## Current Output Artifacts To Audit (failed parity baseline)
Start by reviewing these current outputs (do not assume they are acceptable):
- `blender_pipeline/_brain_basemodel_validation/renders/fixture_01.png`
- `blender_pipeline/_brain_basemodel_validation/renders/fixture_02.png`
- `blender_pipeline/_brain_basemodel_validation/renders/fixture_03.png`
- `blender_pipeline/_brain_basemodel_validation/renders/style_preview_clinical_glow.png`
- `blender_pipeline/_brain_basemodel_validation/renders/style_preview_calm_precision.png`
- `blender_pipeline/_brain_basemodel_validation/renders/style_preview_focus_contrast.png`
- `blender_pipeline/_brain_basemodel_validation/comps/fixtures_contact_sheet.png`
- `blender_pipeline/_brain_basemodel_validation/comps/style_presets_contact_sheet.png`

## High-Level Gap Summary (diagnostic, not solution guidance)
These are the observed issues in the current baseline:
1. Brain form reads as simplified/ball-like rather than convincingly anatomical.
2. Surface detail reads synthetic and low-believability versus target references.
3. Marker/label spatial relationship feels loosely overlaid rather than tightly integrated with anatomy.
4. Lighting/material interplay produces flatter or less premium results than target references.
5. Overall art direction is structurally functional but not production-grade parity.

## What Was Already Attempted
The previous implementation added substantial infra and contract scaffolding, including:
- canonical template build/update scripts
- style preset plumbing
- deterministic scene contract checks
- fixture/spec-based validation harness
- checkpoint/contact-sheet artifact generation

Key files changed in that effort:
- `blender_pipeline/scripts/build_template.py`
- `blender_pipeline/scripts/render_batch.py`
- `blender_pipeline/scripts/validate_template_contract.py`
- `core/blender_gen.py`
- `BRAIN_BASEMODEL_IMPLEMENTATION_NOTES.md`
- `BRAIN_BASEMODEL_VALIDATION_REPORT.md`

You should treat these as **starting infrastructure**, not proof of visual success.

## Non-Negotiables
1. Keep deterministic data binding behavior intact.
2. Do not regress render pipeline compatibility with existing spec flow.
3. Do not introduce Windows/WSL logic.
4. Do not break existing Blender/qEEG tests unless you replace them with stronger equivalent coverage.
5. Do not rely on diffusion generation for final brain scene runtime path.

## Document Priority (what to trust)
Use and align with the latest repo docs and skill materials:
1. `AGENTS.md`
2. `CLAUDE.md`
3. `HANDOFF_NEXT_AGENT_LAST6H_RECOVERY_2026-02-22.md`
4. `.agents/skills/blender-mcp-qeeg-runtime/SKILL.md`
5. `.agents/skills/blender-mcp-qeeg-runtime/references/qeeg-runtime-contract.md`
6. `.agents/skills/blender-mcp-qeeg-runtime/references/style-recipes.md`
7. `BRAIN_BASEMODEL_IMPLEMENTATION_NOTES.md`
8. `BRAIN_BASEMODEL_VALIDATION_REPORT.md`

## Task Requirements For You
1. Perform a full visual parity audit against user references and current renders.
2. Redesign/rebuild the brain visual quality layer until parity is credibly achieved.
3. Preserve deterministic EEG mapping + render contract.
4. Re-run fixture renders and checkpoint outputs with updated visuals.
5. Produce an updated validation packet that includes side-by-side parity evidence.

## Required Deliverables
Create/update all of the following:
1. `BRAIN_BASEMODEL_IMPLEMENTATION_NOTES.md`
- include what changed in visual system and why
- include object/material naming and contract impacts

2. `BRAIN_BASEMODEL_VALIDATION_REPORT.md`
- include explicit parity assessment vs references
- include artifact paths and pass/fail summary

3. Updated render artifacts under:
- `blender_pipeline/_brain_basemodel_validation/renders/`
- `blender_pipeline/_brain_basemodel_validation/checkpoints/`
- `blender_pipeline/_brain_basemodel_validation/comps/`

4. If contract/preset semantics changed, update:
- `.agents/skills/blender-mcp-qeeg-runtime/references/qeeg-runtime-contract.md`
- `.agents/skills/blender-mcp-qeeg-runtime/references/style-recipes.md`

## Verification Commands (minimum)
Run and report results for:
```bash
python3.10 -m pytest -q \
  tests/unit/test_blender_scene_spec_contract.py \
  tests/unit/test_director_blender_fields.py \
  tests/unit/test_visual_gen_backend_selection.py

python3.10 scripts/validate_brain_basemodel.py \
  --blender-bin /Volumes/Blender/Blender.app/Contents/MacOS/Blender \
  --samples 8
```

If additional tests/scripts are needed for parity proof, add them.

## Acceptance Standard
Do not claim success until parity is visibly credible against user references and documented with concrete artifacts.

If parity is not reached, explicitly report: "not at parity" and list remaining visual gaps.

## Important Constraint On This Handoff
This handoff intentionally does **not** prescribe specific artistic or marker-placement techniques. You must own those design decisions and produce the parity outcome.

