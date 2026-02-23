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
