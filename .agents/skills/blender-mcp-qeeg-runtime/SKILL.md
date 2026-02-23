---
name: blender-mcp-qeeg-runtime
description: Route qEEG brain/electrode/coherence scenes to Blender MCP with deterministic clinical data binding and controlled stylistic variation. Use when scenes depict brain topology, electrode maps, or coherence lines. Do not use for non-brain chart/panel slides, which should stay in the template-pack path.
---

# Blender MCP qEEG Runtime Skill

## Overview
Use this skill to produce Blender-ready scene objects for brain visuals while keeping clinical data deterministic and exact.

## Prerequisites: Blender MCP Connection

Before issuing any Blender MCP tool calls, ensure the addon socket server is running inside Blender. Use the `macos-automator` MCP server (AppleScript/accessibility) to automate this -- do NOT use Playwright (Blender is a native app, not a browser).

### Auto-connect sequence (via macos-automator)
1. **Check if Blender is running.** Use AppleScript:
   ```applescript
   tell application "System Events" to (name of processes) contains "Blender"
   ```
2. **If not running, launch it** (non-headless -- must be visible on screen):
   ```applescript
   tell application "Blender" to activate
   ```
   Wait 5-8 seconds for full startup.
3. **Open the BlenderMCP sidebar panel and click "Connect to MCP server".**
   Use `accessibility_action` to find the BlenderMCP panel button. The addon registers a sidebar tab called "BlenderMCP" (accessed via N key in the 3D Viewport). The connect button's label is "Connect to MCP server" (`BLENDERMCP_OT_StartServer`).
   ```applescript
   tell application "System Events"
     tell process "Blender"
       -- Press N to toggle sidebar if not visible
       keystroke "n"
       delay 0.5
       -- Navigate to BlenderMCP tab and click Connect
       -- Use accessibility_action to find button by title "Connect to MCP server"
     end tell
   end tell
   ```
   Alternatively, use the `accessibility_action` tool directly to locate and click the button by its accessibility label.
4. **Verify connection.** After clicking connect, call `get_scene_info` via the blender MCP server. If it returns scene data, the socket bridge on port 9876 is live. If it errors, retry the connect step.

### Important notes
- The addon listens on `localhost:9876` by default.
- Blender MUST be visible on screen (not minimized) for accessibility actions to work.
- macOS Accessibility permissions must be granted to the terminal/app running the agent (System Settings > Privacy & Security > Accessibility).
- If Blender was already running with the addon connected, skip straight to the `get_scene_info` verification.

## Routing Rules
- Use for scenes that visualize a brain surface, EEG electrodes, band activity topography, or coherence connections.
- Do not use for non-brain chart/panel scenes (bars, timelines, split comparisons, KPI cards). Those stay in template-pack rendering.
- If unsure: if the scene must place labels like `C3`, `P4`, `Fp1`, or draw inter-electrode lines on a head model, route to Blender.

## Hard Constraints
- Keep electrode labels exact and canonical (10-20 naming only).
- Keep patient numeric values exact; never invent or “smooth” values.
- Keep on-slide text exact and short (title/subtitle/footer only in Blender scene text fields).
- Keep extraction hints explicit when available (`session_index`, `band`, `metric`) so data binding is stable.
- For Blender API params/enums/commands, verify against Context7 first, then confirm against runtime `bl_rna` enum/property availability in the active Blender version before coding.

## Output Contract
Emit these fields on each Blender scene:

```json
{
  "render_backend": "blender",
  "visual_prompt": "... include [[BLENDER_QEEG]] marker ...",
  "blender": {
    "title": "...",
    "subtitle": "...",
    "footer": "...",
    "extract": {
      "session_index": 1,
      "band": "alpha",
      "metric": "coherence"
    },
    "value_map": {"type": "zscore", "clip": 2.5},
    "coherence_map": {"type": "magnitude", "min": 0.0, "max": 1.0},
    "animation": {
      "enabled": true,
      "duration_sec": 5,
      "fps": 24,
      "camera_orbit_deg": 14
    },
    "style": {
      "lighting_preset": "clinical_glow",
      "camera_preset": "three_quarter_left",
      "palette": "teal-amber"
    }
  }
}
```

Allowed style values:
- `lighting_preset`: `clinical_glow|calm_precision|focus_contrast`
- `camera_preset`: `three_quarter_left|three_quarter_right|frontal|top_center`
- `palette`: `teal-amber|ice-white|cyan-orange`

## Style Freedom
Allow variation only in style dimensions that do not alter correctness:
- Camera framing/angle/orbit
- Lighting mood, bloom/fog intensity
- Color palette and atmosphere
- Subtle camera/emission animation

Never allow style choices to obscure electrode readability, line legibility, or text clarity.

## Reference Files
- Read `references/qeeg-runtime-contract.md` for field-level rules and edge cases.
- Read `references/mcp-capabilities.md` for current Blender MCP tool capabilities and operational pitfalls.
- Read `references/style-recipes.md` for production-grade clinical look recipes and "fancy touches".
