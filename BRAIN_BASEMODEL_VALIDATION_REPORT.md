# BRAIN_BASEMODEL_VALIDATION_REPORT

## Template Version
`brain_basemodel_v2` — MRI-derived pial cortical surface (brainder.org)

## Validation Run
Command:
```bash
python3.10 scripts/validate_brain_basemodel.py \
  --blender-bin /Volumes/Blender/Blender.app/Contents/MacOS/Blender \
  --samples 8
```

Result summary:
- `ok: true`
- `render_mode: gpu`
- `missing_checkpoints: []`

Primary report:
- `blender_pipeline/_brain_basemodel_validation/reports/validation_summary.json`

Template contract report:
- `blender_pipeline/_brain_basemodel_validation/reports/template_contract_report.json`

## Fixture Specs Used
- `blender_pipeline/_brain_basemodel_validation/fixtures/fixture_01.json`
- `blender_pipeline/_brain_basemodel_validation/fixtures/fixture_02.json`
- `blender_pipeline/_brain_basemodel_validation/fixtures/fixture_03.json`

## Render Artifacts
Fixture renders:
- `blender_pipeline/_brain_basemodel_validation/renders/fixture_01.png`
- `blender_pipeline/_brain_basemodel_validation/renders/fixture_02.png`
- `blender_pipeline/_brain_basemodel_validation/renders/fixture_03.png`

Style preview renders:
- `blender_pipeline/_brain_basemodel_validation/renders/style_preview_clinical_glow.png`
- `blender_pipeline/_brain_basemodel_validation/renders/style_preview_calm_precision.png`
- `blender_pipeline/_brain_basemodel_validation/renders/style_preview_focus_contrast.png`

Checkpoint renders:
- `blender_pipeline/_brain_basemodel_validation/checkpoints/scene_101_01_style_text.png`
- `blender_pipeline/_brain_basemodel_validation/checkpoints/scene_101_02_data_bound.png`
- `blender_pipeline/_brain_basemodel_validation/checkpoints/scene_102_01_style_text.png`
- `blender_pipeline/_brain_basemodel_validation/checkpoints/scene_102_02_data_bound.png`
- `blender_pipeline/_brain_basemodel_validation/checkpoints/scene_103_01_style_text.png`
- `blender_pipeline/_brain_basemodel_validation/checkpoints/scene_103_02_data_bound.png`

Contact sheets:
- `blender_pipeline/_brain_basemodel_validation/comps/fixtures_contact_sheet.png`
- `blender_pipeline/_brain_basemodel_validation/comps/style_presets_contact_sheet.png`

## Acceptance Criteria Check
1. Base `.blend` deterministic + reusable:
- Pass (template generated reproducibly from `build_template.py`, validated via contract script).

2. Brain anatomy anatomically recognizable:
- Pass (real MRI pial surface shows clear gyri/sulci, hemisphere separation, and natural cortical folds).

3. Electrodes placed on brain surface:
- Pass (BVH raycasting places electrodes directly on the cortical surface with normal-offset positioning).

4. Three fixture scene specs with data-driven differences:
- Pass (fixture 1/2/3 render distinct electrode/coherence patterns).

5. Electrode labels exact and legible:
- Pass (canonical labels are deterministic objects; active data-bound labels are rendered prominently).

6. Coherence lines visible and non-chaotic:
- Pass for fixture set (lines render with bounded width and deterministic arcs).

7. No platform logic outside macOS added:
- Pass (changes are macOS GPU defaults and generic Blender-safe runtime introspection only).

8. Tests/checks pass and artifacts written:
- Pass.
- Unit tests run:
  - `python3.10 -m pytest -q tests/unit/test_blender_scene_spec_contract.py tests/unit/test_director_blender_fields.py tests/unit/test_visual_gen_backend_selection.py`
  - Result: `13 passed`

## Visual Parity Assessment (v2 vs v1)

### What improved
- Brain form is now anatomically recognizable with real gyri/sulci (was: lumpy UV sphere with random displacement)
- Electrodes sit on the cortical surface (was: floating on a separate perfect sphere at r=2.22)
- Hemisphere separation is natural from the MRI mesh (was: boolean cut with a thin cube)
- Surface detail reads as organic cortical tissue (was: synthetic noise texture)

### Remaining gaps vs AI-generated reference targets
- Material color/glow could be further tuned for closer match to the warm gold + cool blue aesthetic in Qwen-generated references
- Electrode markers could use stronger color differentiation when data-bound (active vs inactive)
- The SSS translucency effect is subtle; reference images show more dramatic subsurface glow
- Camera framing and composition differences are style-preset level (adjustable without code changes)

### Verdict
The v2 brain is at credible anatomical parity. The form reads as a real brain. Further visual refinement is now in the domain of material/lighting tuning rather than fundamental geometry problems.

## Notes
- Validation harness uses lower samples for speed (`--samples 8`) while preserving deterministic scene contract checks.
- Production renders can increase quality by raising `BLENDER_SAMPLES`.
- Brain mesh post-processing: Decimate (0.35 ratio) + Smooth (8 iter, 0.5 factor) + SubSurf (render=1).
- Mesh source: brainder.org MNI152 pial surfaces, CC-BY-SA 3.0.
