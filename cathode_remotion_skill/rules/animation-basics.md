# Animation Basics (Remotion)

The composition templates use these standard animation patterns. This is reference material — the director does not write animation code, but understanding the timing helps with narration pacing.

## Spring reveal

Every template uses a spring animation for the initial entrance:
- Config: damping=20, stiffness=100, mass=0.9
- Duration: approximately 0.5 seconds to settle
- Effect: elements fade in and slide up from a slight offset

## Staggered list items

Lists (bullet_stack, data_stage, brain_region regions) reveal items one at a time:
- Base delay: 12 frames (0.4s at 30fps)
- Per-item delay: 8 frames (0.27s)
- A 5-item list takes approximately 2 seconds to fully reveal

## Scene timing

Default scene duration is 5 seconds (150 frames at 30fps). In production, scene duration is set to match the narration audio length.

The director should write narration that gives the visuals time to land:
- Opening sentence while the headline animates in
- Key data points after the reveal has settled
- Don't front-load all information — let the staggered reveals create natural beats

## Transitions

Scenes are hard-cut by default. The assembly pipeline handles scene-to-scene flow.

## Design tokens

All templates share a consistent dark-premium aesthetic:
- Background: dark gradient (#0a0a0f to #1a1a2e) when no background image
- Text: warm off-white (#f7efe6)
- Accent: teal (#5eead4) — used for highlights, improved status, key values
- Headlines: Georgia serif, bold
- Body: Inter sans-serif
- Data values: monospace (SFMono/Menlo)
- Padding: 88px on all sides

These are fixed — the director does not need to specify colors or fonts.
