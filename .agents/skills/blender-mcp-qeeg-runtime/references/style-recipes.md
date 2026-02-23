# Style Recipes (Clinical + Cinematic)

Use these recipes to vary style without affecting clinical correctness.

## Baseline: `clinical_glow`
- Camera: 3/4 head view, mild downward angle, brain occupies ~35-45% of frame.
- Key light: cool (teal-blue), soft shadows.
- Rim light: warm (amber) to separate silhouette.
- Background: dark navy gradient with low-density volumetric fog.
- Post: subtle bloom and vignette, no heavy chromatic aberration.

## Variation A: `calm_precision`
- Camera: slightly wider lens/FOV and more negative space.
- Palette: cool neutrals + white emissive labels.
- Effects: minimal fog, crisp coherence lines, low bloom.

## Variation B: `focus_contrast`
- Camera: tighter crop around active regions.
- Palette: high-contrast cyan/orange for signal emphasis.
- Effects: localized glow around top-quantile electrodes only.

## Animation Recipe (subtle)
- Duration: 4-6s, 24 fps.
- Camera: slow orbit (10-18 degrees total arc).
- Emission: soft sinusoidal pulse on active electrodes and strong coherence edges.
- Avoid fast cuts, whip pans, or aggressive shake.

## Do/Don't
- Do prioritize electrode and text readability over atmosphere.
- Do keep line thickness and emissive values stable enough for QC visibility.
- Don’t place bright fog layers between camera and labels.
- Don’t saturate so hard that red/green value distinctions collapse.

## Community References
- Blender MCP discussion board: <https://github.com/ahujasid/blender-mcp/discussions>
- Blender Artists 3D-agent thread (tooling and practical quality focus): <https://blenderartists.org/t/3d-agent-for-blender-mcp-llm-powered-3d-model-generation-with-clean-topology/1577821>
- Reddit community usage examples:
  - <https://www.reddit.com/r/mcp/comments/1lmy31f/turned_blender_into_a_graphics_editor_with/>
  - <https://www.reddit.com/r/vibecoding/comments/1lvcw17/i_tried_to_build_a_castle_in_blender_with_claude/>
