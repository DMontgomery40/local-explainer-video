---
name: qeeg-remotion-scenes
description: Remotion composition templates for qEEG explainer videos — brain region maps, metric cards, timelines, data stages, and analogy panels with deterministic text rendering.
---

## When to use

Use this skill when generating storyboard JSON for qEEG/neurofeedback patient explainer videos. The storyboard scenes map to a fixed set of Remotion composition templates rather than freeform image prompts.

## How it works

1. The director receives a patient's qEEG analysis report.
2. For each scene, the director picks a composition family and fills structured props.
3. Remotion renders each scene deterministically — all text is pixel-perfect, no OCR needed.

## Key rules

- Every scene must specify a `composition.family` and `composition.props`.
- Props contain the exact on-screen text (headlines, values, labels). Use digits in props for precision.
- Narration must spell out all numbers as words (TTS requirement).
- The `brain_region_focus` template has built-in 10-20 electrode positions — just use region names.
- Do not generate freeform image prompts, TSX code, or arbitrary React components.

## Available templates

Load [./rules/scene-templates.md](./rules/scene-templates.md) for the full catalog of composition families and their props.

## Brain mapping

Load [./rules/brain-mapping.md](./rules/brain-mapping.md) for the 10-20 electrode coordinate system used by `brain_region_focus`.

## Animation basics

Load [./rules/animation-basics.md](./rules/animation-basics.md) for Remotion animation patterns used in the compositions.
