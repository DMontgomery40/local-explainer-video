# Deterministic Template Pipeline Recovery Contract

This file locks implementation precedence and anti-drift rules for the active recovery effort.

## Document Precedence
1. `/Users/davidmontgomery/local-explainer-video/HANDOFF-deterministic-template-pipeline.md`
2. `/Users/davidmontgomery/local-explainer-video/HANDOFF_NEXT_AGENT_DETERMINISTIC_TEMPLATE_PIPELINE_2026-02-22.md`
3. Runtime contracts in code/tests (`core/template_pipeline/*`, `tests/contract/*`)

## Advisory-Only Inputs
- `/Users/davidmontgomery/local-explainer-video/.codex/research/remotion_hybrid_deep_research_2026-02-22.md`
- `/Users/davidmontgomery/local-explainer-video/remotion-migration-report-v2.md`
- `/Users/davidmontgomery/local-explainer-video/AGENTS_override.md`
- `/Users/davidmontgomery/local-explainer-video/PROJECT.md`

Advisory docs can inform implementation details but cannot override scope/constraints from the two handoff documents above.

## Drift Guardrails
- `generic_data_panel_v1` is development-only and never valid as production fallback.
- Production template mode must hard-fail when archetype coverage is missing.
- Every template fallback event must emit an audit artifact.
- Director structured output (`scene_type + structured_data`) is first-class in template mode.
- Downstream scene-typer fallback is migration-only and must be explicitly controllable.

## Template Status Encoding
- `origin`: `qwen_curated` or `scaffold_only`
- `operational_status`: `production` or `dev_only`
- `production_ready`: boolean release gate flag
- `origin=scaffold_only` must always remain `operational_status=dev_only` and `production_ready=false`

No template is considered production-eligible unless these fields are present in `templates/manifest.json`.
