# Template Selector Heuristic

Use this skill when selecting a deterministic template for a scene in the local explainer pipeline.

## Purpose
- Map `scene_type + structured_data` to `template_id` with deterministic, zero-LLM rules.
- Keep runtime selection fast, reproducible, and auditable.

## Source of Truth
- Template manifest: `templates/manifest.json`
- Scene schema: `core/template_pipeline/scene_schemas.py`
- Selector implementation: `core/template_pipeline/selector.py`

## Rules
1. Normalize scene type first (`normalize_scene_type`).
2. Filter manifest entries by `scene_types`.
3. Build selector probe from structured payload counts:
   - `bar_count`, `edge_count`, `item_count`, `row_count`, `session_count`, `ring_count`, `trace_count`, `point_count`, `trend`
4. Evaluate selector keys:
   - `<field>_min`
   - `<field>_max`
   - `<field>_eq`
   - `<field>_in`
   - `trend`
5. Choose lowest `priority` among matches.
6. Development mode: if no match, fallback to lowest-priority template for that scene type.
7. Production mode: if no production-ready match exists, fail loudly (do not fallback to generic).

## Do Not
- Do not use stochastic selection.
- Do not browse templates manually at runtime.
- Do not mutate manifest data during selection.

## Validation Checklist
- `scene_type` exists and is known.
- `structured_data` is schema-valid.
- selected `template_id` exists in manifest.
- anchor file and template background exist on disk.
