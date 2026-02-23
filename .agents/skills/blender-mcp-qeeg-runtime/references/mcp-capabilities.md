# Blender MCP Capabilities And Constraints

## Current Tool Surface (source-backed)
From `ahujasid/blender-mcp` server implementation (`src/blender_mcp/server.py`), core tools include:
- Scene read tools: `get_scene_info`, `get_object_info`, `get_viewport_screenshot`
- Scene mutation tool: `execute_blender_code`
- Asset/search tools:
  - PolyHaven: `get_polyhaven_categories`, `search_polyhaven_assets`, `download_polyhaven_asset`, `set_texture`
  - Sketchfab: `get_sketchfab_status`, `search_sketchfab_models`, `get_sketchfab_model_preview`, `download_sketchfab_model`
  - Hyper3D: `get_hyper3d_status`, `generate_hyper3d_model_via_text`, `generate_hyper3d_model_via_images`, `poll_rodin_job_status`, `import_generated_asset`
  - Hunyuan3D: `get_hunyuan3d_status`, `generate_hunyuan3d_model`, `poll_hunyuan_job_status`, `import_generated_asset_hunyuan`

## Operational Notes For Runtime Reliability
- Prefer scene-read tools before mutation (`get_scene_info` -> `get_object_info` -> `execute_blender_code`) for predictable edits.
- Keep generated/imported assets optional for qEEG slides; core clinical pipeline should work from the existing template.
- Use `get_viewport_screenshot` checkpoints between major mutations to catch regressions early.

## Known Pitfalls From Upstream Issue Tracker
- Mixed Windows+WSL screenshot path coupling can fail (`get_viewport_screenshot` issue #189).
- Repeated material resets can create duplicated node graphs and unstable shading if not normalized (issue #190).
- Model-gen APIs can fail on payload encoding mismatch (Hyper3D image mode issue #177).

## Practical Guardrails For This Repo
- Keep the patient-data visualization template as primary path.
- Treat imported/generative assets as optional style accents only.
- Never let MCP stylistic operations override deterministic electrode/data bindings.

## Sources
- GitHub README: <https://github.com/ahujasid/blender-mcp>
- Server tool registry: <https://raw.githubusercontent.com/ahujasid/blender-mcp/main/src/blender_mcp/server.py>
- Issue #189: <https://github.com/ahujasid/blender-mcp/issues/189>
- Issue #190: <https://github.com/ahujasid/blender-mcp/issues/190>
- Issue #177: <https://github.com/ahujasid/blender-mcp/issues/177>
