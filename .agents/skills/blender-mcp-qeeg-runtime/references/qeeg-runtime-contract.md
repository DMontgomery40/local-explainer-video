# qEEG Runtime Contract

## Required Scene Fields
- `render_backend`: must be `"blender"`.
- `visual_prompt`: include `[[BLENDER_QEEG]]` marker for explicit routing safety.
- `blender.title`: short, exact display title.
- `blender.subtitle`: short, exact support line.

## Strongly Recommended Fields
- `blender.extract.session_index`: integer when session-specific claim is made.
- `blender.extract.band`: one of `delta|theta|alpha|beta|gamma` when band-specific.
- `blender.extract.metric`: one of `zscore|power|coherence`.

## Optional Override Fields
- `blender.electrode_values`: explicit `{label: value}` map (use when pre-extracted values are known).
- `blender.coherence_edges`: explicit edge list with entries `{a,b,value}`.
- `blender.value_map`: deterministic value-to-color mapping control.
- `blender.coherence_map`: deterministic edge styling control.
- `blender.animation`: deterministic animation controls.
- `blender.style`: stylistic presets (`lighting_preset`, `camera_preset`, `palette`).
  - `lighting_preset`: `clinical_glow|calm_precision|focus_contrast`
  - `camera_preset`: `three_quarter_left|three_quarter_right|frontal|top_center`
  - `palette`: `teal-amber|ice-white|cyan-orange`

## Runtime Update Contract
- Single entrypoint: one scene spec JSON per render job (contract version `brain_basemodel_v1`).
- Required naming conventions:
  - electrode anchors: `ANCH_E_<label>`
  - electrode meshes: `E_<label>`
  - electrode labels: `LBL_<label>`
  - coherence lines: `LINE_<a>_<b>`
  - generated materials: `MAT_JOB_E_<label>`, `MAT_JOB_LBL_<label>`, `MAT_JOB_LINE_<a>_<b>`
- Job updates must be idempotent:
  - upsert known object/material names
  - clear/rebuild line collection for each scene payload
  - avoid material-node duplication across repeated runs
- Optional checkpoint renders are supported between major phases (`style/text` and `data-bound`) for QC traceability.

## Determinism Boundaries
Deterministic by policy:
- Electrode names/placement
- Numeric values and labels
- Session/band/metric binding

Allowed to vary:
- Camera angle and subtle motion
- Light mood, volumetric strength, bloom
- Palette and atmospheric accents

## Failure Prevention
- Never emit electrode labels outside canonical 10-20 sets.
- Never emit text that restates a value not present in source data.
- Keep Blender text blocks concise to avoid clipping.
- Avoid visual designs where coherence lines overlap labels at center frame.
- For Blender API tuning changes, gate updates through Context7 + runtime `bl_rna` checks to avoid stale enum/property assumptions across Blender versions.
