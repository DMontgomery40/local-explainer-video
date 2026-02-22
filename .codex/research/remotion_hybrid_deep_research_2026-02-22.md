# Local Explainer Video: Deterministic Data Slide Migration Research

Date: 2026-02-22
Repo: /Users/davidmontgomery/local-explainer-video
Analyst: Codex (local, read-only analysis pass; no production file modifications)

## Executive summary

This report completes a full-corpus, scene-by-scene analysis of the patient output pipeline and validates a hybrid Remotion migration strategy against current (February 2026) Remotion guidance.

Main conclusions:

1. The corpus supports a stable canonical template set for data-bearing slides.
2. A deterministic Remotion layer can cover all observed data-bearing scenes without sacrificing the strongest atmospheric aesthetics.
3. Atmospheric and metaphor scenes should remain AI-generated and be composed with deterministic overlays when data labels are present.
4. The previous migration report is directionally correct, but several technical assumptions are now stale and should be amended.

Critical technical correction:

- `getAudioDurationInSeconds()` is deprecated in Remotion docs; duration extraction should be migrated to newer media parsing pathways (Mediabunny-first strategy).

---

## Corpus census and methodology

### Scope

Analyzed all patient-like project folders in:

- `/Users/davidmontgomery/local-explainer-video/projects`

Using strict patient-id pattern `MM-DD-YYYY-N`:

- Patient dirs found: 12
- With `plan.json`: 11
- Missing `plan.json`: `09-05-1954-1`

### Scene/image coverage

- Total scenes parsed from patient plans: 169
- Resolved rendered images: 169/169
- Decode failures: 0

Note on path reconciliation:

- One project (`10-07-1963-0`) contained stale absolute image paths pointing at `10-07-1963__02`.
- Resolver fallback to local `projects/<id>/images/scene_xxx.png` recovered all scenes.

### What was inspected per scene

For each scene:

- `title`, `narration`, `visual_prompt`, `image_path`, optional fields
- Pixel-level image decode and visual metrics
- OCR token extraction and numeric cross-check against scene text

### Quantitative quality proxy (OCR numeric consistency)

- Scenes with numeric expectations: 152
- Mean numeric match ratio: 0.532
- Median numeric match ratio: 0.551
- Perfect numeric match scenes: 37

This is a conservative proxy because stylized typography can reduce OCR capture even when numbers are visually present.

### Evidence artifacts generated

Data:

- `/Users/davidmontgomery/local-explainer-video/.codex/research/data/local_explainer_scene_audit.json`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/data/local_explainer_scene_audit.csv`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/data/local_explainer_scene_archetypes_manual.csv`

Visual contact sheets (all projects):

- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/01-01-1983-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/01-01-1989-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/01-01-1991-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/01-01-2013-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/01-19-1966-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/04-08-1997-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/07-14-2008-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/09-05-1954-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/09-23-1982-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/10-07-1963-0_contact.jpg`
- `/Users/davidmontgomery/local-explainer-video/.codex/research/contact-sheets/11-21-1996-0_contact.jpg`

---

## Canonical archetype taxonomy

This section lists the complete data-bearing template set derived from full corpus induction.

Counts below are from manual scene-level assignment across all 169 scenes.

### Data-bearing archetypes (canonical)

1. `split_opposing_trends`
- Count: 17
- Typical use: one metric improves while another worsens, paradox framing.
- Required props:
  - `left`: `{metric, from, to, unit, direction, pct_change?}`
  - `right`: same shape
  - `headline?`, `takeaway?`

2. `multi_session_trend`
- Count: 15
- Typical use: session 1 -> session 2 -> session 3 progression for one signal.
- Required props:
  - `metric`, `unit`
  - `points`: `[{session, date?, value}]`
  - `trend`: `{direction, pct_change?}`
  - `target_band?`

3. `verdict_summary`
- Count: 12
- Typical use: final conclusions with KPI bullets/checkmarks.
- Required props:
  - `verdict_title`
  - `bullets`: `[{label, status, from?, to?, delta?}]`
  - `recommendation`

4. `roadmap_agenda`
- Count: 11
- Typical use: slide 1 agenda and section framing.
- Required props:
  - `title`
  - `items`: `[{order, label}]`

5. `session_timeline`
- Count: 9
- Typical use: chronological dates/intervals between sessions.
- Required props:
  - `sessions`: `[{label, date, days_to_next?}]`
  - `total_span_days?`

6. `gauge_ratio_meter`
- Count: 8
- Typical use: theta-beta, alpha balance, zone normalization.
- Required props:
  - `metric`, `unit?`
  - `readings`: `[{label, value}]`
  - `target_band`: `{min, max}`
  - `zones?`

7. `coherence_network_map`
- Count: 8
- Typical use: one state/session brain topology with electrode edges.
- Required props:
  - `session_label`
  - `nodes`: `[{id, x, y}]` (10-20 map coords)
  - `edges`: `[{from, to, value, label?, style_key?}]`

8. `hemispheric_compare`
- Count: 8
- Typical use: left-vs-right dominance/rebalancing.
- Required props:
  - `left`: `{label, value}`
  - `right`: `{label, value}`
  - `session_pair?`, `delta?`

9. `state_flexibility_rest_task`
- Count: 8
- Typical use: rest decreases while task increases (state-dependent adaptation).
- Required props:
  - `rest`: `{summary, edges?, metrics?}`
  - `task`: `{summary, edges?, metrics?}`
  - `interpretation`

10. `baseline_target_split`
- Count: 6
- Typical use: baseline severity against normative range.
- Required props:
  - `baseline`: `{label, value, unit}`
  - `target`: `{min, max, unit}`
  - `status`

11. `measurement_primer`
- Count: 5
- Typical use: explanatory qEEG instrumentation/setup.
- Required props:
  - `modalities`: `[{label, description}]`
  - `electrode_sites?`: `string[]`
  - `definitions?`

12. `future_projection`
- Count: 5
- Typical use: forward monitoring plan and checkpoints.
- Required props:
  - `current_state`
  - `checkpoints`: `[{label, goal}]`
  - `monitoring_plan`: `string[]`

13. `line_trajectory`
- Count: 5
- Typical use: U-shape/jagged non-linear trend lines.
- Required props:
  - `x_labels`: `string[]`
  - `series`: `[{name, values}]`
  - `annotations?`: `[{x, text, severity?}]`

14. `bar_volume_chart`
- Count: 5
- Typical use: bar/volume meter quantitative comparison.
- Required props:
  - `metric`, `unit`
  - `bars`: `[{label, value, color_key?}]`
  - `target_band?`

15. `coherence_progression_sequence`
- Count: 5
- Typical use: same connection values across sessions.
- Required props:
  - `edges`: `[{from, to}]`
  - `sessions`: `[{label, values: number[]}]`

16. `dotplot_variability`
- Count: 4
- Typical use: reaction-time mean + spread comparisons.
- Required props:
  - `metric`, `unit`
  - `sessions`: `[{label, mean, spread}]`
  - `target_band?`

17. `pathway_hub_synthesis`
- Count: 4
- Typical use: causal pathways / hub-and-spoke synthesis.
- Required props:
  - `nodes`: `[{id, label, type}]`
  - `edges`: `[{from, to, label?}]`
  - `conclusion`

18. `regional_frequency_map`
- Count: 4
- Typical use: frontal/central-occipital directional shifts.
- Required props:
  - `regions`: `[{name, from, to, unit, target?}]`
  - `interpretation`

19. `table_dashboard`
- Count: 4
- Typical use: compact tabular metric panels.
- Required props:
  - `columns`: `string[]`
  - `rows`: `Array<Array<string|number>>`
  - `highlights?`

20. `quality_alert`
- Count: 3
- Typical use: low-yield flags, artifact caveats, reliability notes.
- Required props:
  - `flags`: `[{kind, severity, text}]`
  - `impacted_metrics`
  - `confidence_note`

### Atmospheric archetypes (retain AI generation)

- `atmospheric_title_card`
- `atmospheric_metaphor_scene`
- `mechanism_illustration`

These should remain generative or pre-baked-image-driven, with deterministic overlay text when data labels are shown.

---

## Exemplar gallery (2 best examples per archetype)

Selection criteria used:

- cinematic gravitas
- compositional elegance
- chromatic sophistication
- typographic harmony
- data-visual congruence

### 1) `roadmap_agenda`

- `/Users/davidmontgomery/local-explainer-video/projects/10-07-1963-0/images/scene_001.png`
- `/Users/davidmontgomery/local-explainer-video/projects/01-01-2013-0/images/scene_001.png`

Why: strongest hierarchy and pacing cues with minimal clutter.

### 2) `session_timeline`

- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0/images/scene_002.png`
- `/Users/davidmontgomery/local-explainer-video/projects/04-08-1997-0/images/scene_002.png`

Why: best temporal spacing and milestone readability.

### 3) `measurement_primer`

- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1983-0/images/scene_003.png`
- `/Users/davidmontgomery/local-explainer-video/projects/04-08-1997-0/images/scene_003.png`

Why: clear educational framing, premium medical visual tone.

### 4) `baseline_target_split`

- `/Users/davidmontgomery/local-explainer-video/projects/10-07-1963-0/images/scene_002.png`
- `/Users/davidmontgomery/local-explainer-video/projects/11-21-1996-0/images/scene_002.png`

Why: immediate severity-vs-target communication.

### 5) `multi_session_trend`

- `/Users/davidmontgomery/local-explainer-video/projects/07-14-2008-0/images/scene_003.png`
- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1983-0/images/scene_004.png`

Why: strong three-stage narrative progression.

### 6) `bar_volume_chart`

- `/Users/davidmontgomery/local-explainer-video/projects/07-14-2008-0/images/scene_004.png`
- `/Users/davidmontgomery/local-explainer-video/projects/04-08-1997-0/images/scene_007.png`

Why: highest clarity for scale and comparative magnitude.

### 7) `gauge_ratio_meter`

- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0/images/scene_012.png`
- `/Users/davidmontgomery/local-explainer-video/projects/09-23-1982-0/images/scene_011.png`

Why: best target-zone legibility and clinical tone.

### 8) `dotplot_variability`

- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0/images/scene_006.png`
- `/Users/davidmontgomery/local-explainer-video/projects/11-21-1996-0/images/scene_012.png`

Why: best mean/spread communication in a single frame.

### 9) `split_opposing_trends`

- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1989-0/images/scene_004.png`
- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1983-0/images/scene_009.png`

Why: strong semantic polarity and visual immediacy.

### 10) `coherence_network_map`

- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0/images/scene_007.png`
- `/Users/davidmontgomery/local-explainer-video/projects/11-21-1996-0/images/scene_003.png`

Why: most compelling topology rendering for electrode links.

### 11) `coherence_progression_sequence`

- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0/images/scene_009.png`
- `/Users/davidmontgomery/local-explainer-video/projects/10-07-1963-0/images/scene_010.png`

Why: best session-over-session network evolution framing.

### 12) `hemispheric_compare`

- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1989-0/images/scene_009.png`
- `/Users/davidmontgomery/local-explainer-video/projects/11-21-1996-0/images/scene_007.png`

Why: strongest lateralized contrast and rebalancing story.

### 13) `regional_frequency_map`

- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1989-0/images/scene_010.png`
- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0/images/scene_011.png`

Why: good region segmentation and directional tuning clarity.

### 14) `table_dashboard`

- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1991-0/images/scene_002.png`
- `/Users/davidmontgomery/local-explainer-video/projects/04-08-1997-0/images/scene_005.png`

Why: compact numeric density with acceptable readability.

### 15) `state_flexibility_rest_task`

- `/Users/davidmontgomery/local-explainer-video/projects/07-14-2008-0/images/scene_008.png`
- `/Users/davidmontgomery/local-explainer-video/projects/10-07-1963-0/images/scene_011.png`

Why: best rest-vs-task distinction with coherent narrative semantics.

### 16) `quality_alert`

- `/Users/davidmontgomery/local-explainer-video/projects/01-01-2013-0/images/scene_012.png`
- `/Users/davidmontgomery/local-explainer-video/projects/11-21-1996-0/images/scene_013.png`

Why: clear caution signals without panic or clutter.

### 17) `verdict_summary`

- `/Users/davidmontgomery/local-explainer-video/projects/09-05-1954-0/images/scene_015.png`
- `/Users/davidmontgomery/local-explainer-video/projects/09-23-1982-0/images/scene_014.png`

Why: strongest authority, readability, and closure quality.

### 18) `future_projection`

- `/Users/davidmontgomery/local-explainer-video/projects/10-07-1963-0/images/scene_015.png`
- `/Users/davidmontgomery/local-explainer-video/projects/09-23-1982-0/images/scene_013.png`

Why: clear actionability and forward-looking framing.

### 19) `pathway_hub_synthesis`

- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1991-0/images/scene_012.png`
- `/Users/davidmontgomery/local-explainer-video/projects/04-08-1997-0/images/scene_014.png`

Why: strongest causal storytelling topology.

### 20) `line_trajectory`

- `/Users/davidmontgomery/local-explainer-video/projects/04-08-1997-0/images/scene_008.png`
- `/Users/davidmontgomery/local-explainer-video/projects/01-01-1989-0/images/scene_005.png`

Why: best non-linear dip/rebound expression.

---

## Director output schema proposal

Add strict discriminated scene typing to `plan.json`.

Top-level:

```json
{
  "schema_version": "3.0.0",
  "meta": {
    "patient_id": "MM-DD-YYYY-N",
    "created_at": "ISO-8601"
  },
  "scenes": [
    {
      "id": 4,
      "uid": "abcd1234",
      "title": "Signal Strength: Nearly Doubled",
      "narration": "...",
      "scene_type": "bar_volume_chart",
      "structured_data": {},
      "render_hints": {
        "theme": "warm_clinical",
        "background": "programmatic_or_ai"
      },
      "audio_path": "projects/.../audio/scene_004.wav"
    }
  ]
}
```

Example A: `bar_volume_chart` (real scene: `09-05-1954-0/scene_004`)

```json
{
  "scene_type": "bar_volume_chart",
  "structured_data": {
    "metric": "P300 Signal Voltage",
    "unit": "uV",
    "bars": [
      {"label": "Session 1", "value": 13.1},
      {"label": "Session 2", "value": 22.9},
      {"label": "Session 3", "value": 24.0}
    ],
    "target_band": {"min": 9.0, "max": 22.0}
  }
}
```

Example B: `coherence_network_map` (real scene: `09-05-1954-0/scene_007`)

```json
{
  "scene_type": "coherence_network_map",
  "structured_data": {
    "session_label": "Session 1",
    "nodes": [
      {"id": "C3", "x": 0.35, "y": 0.45},
      {"id": "CZ", "x": 0.50, "y": 0.44},
      {"id": "C4", "x": 0.65, "y": 0.45},
      {"id": "T5", "x": 0.20, "y": 0.65},
      {"id": "T6", "x": 0.80, "y": 0.65}
    ],
    "edges": [
      {"from": "T5", "to": "C3", "value": 0.89, "label": "Dominant", "style_key": "over"},
      {"from": "T6", "to": "C4", "value": 0.65, "label": "Weak", "style_key": "under"}
    ]
  }
}
```

Example C: `split_opposing_trends` (real scene: `01-01-1989-0/scene_004`)

```json
{
  "scene_type": "split_opposing_trends",
  "structured_data": {
    "left": {
      "metric": "P300 Latency",
      "from": 344,
      "to": 264,
      "unit": "ms",
      "direction": "improved"
    },
    "right": {
      "metric": "Reaction Time",
      "from": 246,
      "to": 293,
      "unit": "ms",
      "direction": "worsened"
    },
    "takeaway": "Neural recognition speed improved while motor response lagged"
  }
}
```

Example D: `dotplot_variability` (real scene: `09-05-1954-0/scene_006`)

```json
{
  "scene_type": "dotplot_variability",
  "structured_data": {
    "metric": "Reaction Time",
    "unit": "ms",
    "sessions": [
      {"label": "Session 1", "mean": 313, "spread": 170},
      {"label": "Session 2", "mean": 287, "spread": 87},
      {"label": "Session 3", "mean": 256, "spread": 80}
    ],
    "target_band": {"min": 220, "max": 320}
  }
}
```

Schema principles:

- every data-bearing scene must have `scene_type` + validated `structured_data`
- every displayed numeric label must come from `structured_data`
- no freeform numeric text inside component logic
- render should fail fast on schema violations

---

## Component priority ranking (frequency x complexity)

Scoring used:

- `priority_score = frequency * implementation_complexity`
- complexity scale: 1 (trivial) to 5 (hard)

| Rank | scene_type | Frequency | Complexity | Score |
|---:|---|---:|---:|---:|
| 1 | split_opposing_trends | 17 | 3 | 51 |
| 2 | coherence_network_map | 8 | 5 | 40 |
| 3 | state_flexibility_rest_task | 8 | 4 | 32 |
| 4 | multi_session_trend | 15 | 2 | 30 |
| 5 | verdict_summary | 12 | 2 | 24 |
| 6 | hemispheric_compare | 8 | 3 | 24 |
| 7 | gauge_ratio_meter | 8 | 3 | 24 |
| 8 | coherence_progression_sequence | 5 | 4 | 20 |
| 9 | session_timeline | 9 | 2 | 18 |
| 10 | baseline_target_split | 6 | 2 | 12 |
| 11 | table_dashboard | 4 | 3 | 12 |
| 12 | regional_frequency_map | 4 | 3 | 12 |
| 13 | pathway_hub_synthesis | 4 | 3 | 12 |
| 14 | dotplot_variability | 4 | 3 | 12 |
| 15 | roadmap_agenda | 11 | 1 | 11 |
| 16 | measurement_primer | 5 | 2 | 10 |
| 17 | line_trajectory | 5 | 2 | 10 |
| 18 | future_projection | 5 | 2 | 10 |
| 19 | bar_volume_chart | 5 | 2 | 10 |
| 20 | quality_alert | 3 | 2 | 6 |

Recommended phase ordering:

- Phase A (highest impact): 1-9
- Phase B (mid): 10-15
- Phase C (long tail): 16-20

---

## Remotion ecosystem intelligence and migration-plan amendments

This section audits prior assumptions against current public Remotion sources (as of 2026-02-22).

### Confirmed

1. Hybrid architecture is still the right target.
- deterministic React rendering for data slides
- AI imagery for atmospheric backgrounds and metaphor scenes

2. `calculateMetadata()` remains first-class.
- still the correct pre-render hook for dynamic duration/dimensions/props.

3. v4 stable remains current.
- npm `remotion` latest is `4.0.427`
- no stable v5 at this date

### Corrections

1. Duration API assumption must change.
- `getAudioDurationInSeconds()` is deprecated in Remotion docs.
- prior plan using it directly should be updated.

2. Media stack transition nuance.
- Remotion docs now position Media Parser as deprecated and recommend migration to Mediabunny.
- migration plan should avoid locking into legacy helper functions.

3. Chrome runtime assumptions have shifted.
- Remotion now documents Chrome Headless Shell as default managed strategy.
- `chromeMode` is now a meaningful control surface (`chrome-headless-shell` vs `chrome-for-testing`).

4. Concurrency should be benchmarked, not hardcoded.
- official docs recommend `npx remotion benchmark` to discover optimal concurrency.
- issue evidence shows `100%` concurrency can degrade throughput on some workloads.

5. Hardware acceleration constraints.
- on macOS, VideoToolbox acceleration is supported.
- with hardware acceleration, CRF cannot be used; bitrate-based control is required.

### Extensions to add to migration plan

1. Render profile presets
- `cpu_default`
- `mac_videotoolbox`
- `low_memory`

2. Determinism guardrails
- strict schema validation before render
- all on-screen numbers sourced only from structured props
- preserve render metadata (remotion version, chrome mode, renderer options)

3. Timeout hardening
- label every `delayRender()` source
- set per-asset timeout override where needed

4. Memory-management fallback path
- expose `disallowParallelEncoding` option for memory-constrained runs
- expose media/offthread cache and thread options in advanced config

5. Skills/mcp alignment
- remotion skills are now first-party documented for agent workflows
- keep project-local skill rules for qEEG clinical numeric integrity (domain-specific constraints not covered by generic Remotion skills)

---

## Proposed target architecture

### Layer 1: deterministic data components (Remotion)

Implement all 20 data-bearing archetypes as typed components.

### Layer 2: atmospheric generation (AI)

Keep title/metaphor/mechanism art generated by existing image pipeline.

### Layer 3: hybrid composition

Where needed, composite deterministic text/SVG overlays on generated backgrounds.

Result:

- preserve visual richness
- remove stochastic data text failures
- improve reproducibility and QC throughput

---

## Known limitations of this analysis

1. OCR is used as one proxy for rendered text fidelity.
- stylized fonts and glow effects can undercount true correctness.

2. One patient-like directory lacked plan data.
- `09-05-1954-1` had no `plan.json`, so no scene-level inclusion.

3. Archetype assignment is semantically manual, then codified.
- assignment table is included for reproducibility in CSV.

---

## Appendix A: full scene-level mapping

Use this file for complete scene-to-archetype traceability:

- `/Users/davidmontgomery/local-explainer-video/.codex/research/data/local_explainer_scene_archetypes_manual.csv`

Columns:

- `project`
- `scene_id`
- `title`
- `archetype_code`
- `archetype_name`
- `image_path`

---

## Appendix B: full per-scene audit payload

- `/Users/davidmontgomery/local-explainer-video/.codex/research/data/local_explainer_scene_audit.json`

Contains:

- all parsed scene fields
- resolved image paths
- image metrics
- OCR text/confidence
- numeric overlap diagnostics

---

## Appendix C: source links used for Remotion intelligence

Official docs and release channels:

- https://www.remotion.dev/docs/calculate-metadata
- https://www.remotion.dev/docs/performance
- https://www.remotion.dev/docs/renderer/render-media
- https://www.remotion.dev/docs/cli/render
- https://www.remotion.dev/docs/hardware-acceleration
- https://www.remotion.dev/docs/miscellaneous/chrome-headless-shell
- https://www.remotion.dev/docs/timeout
- https://www.remotion.dev/docs/cli/benchmark
- https://www.remotion.dev/docs/get-audio-duration-in-seconds
- https://www.remotion.dev/docs/media-parser/parse-media
- https://www.remotion.dev/docs/mediabunny
- https://www.remotion.dev/docs/mediabunny/version
- https://www.remotion.dev/docs/ai/skills
- https://www.remotion.dev/docs/ai/mcp
- https://github.com/remotion-dev/remotion/releases
- https://www.npmjs.com/package/remotion
- https://github.com/remotion-dev/skills

Issue/PR references sampled for operational evidence:

- https://github.com/remotion-dev/remotion/issues/4300
- https://github.com/remotion-dev/remotion/issues/5792
- https://github.com/remotion-dev/remotion/issues/5843
- https://github.com/remotion-dev/remotion/issues/6446
- https://github.com/remotion-dev/remotion/pull/6329
- https://github.com/remotion-dev/remotion/pull/6410
- https://github.com/remotion-dev/remotion/pull/6487

