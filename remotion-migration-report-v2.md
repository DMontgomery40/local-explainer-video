# Remotion Migration: Grounded Analysis After Reviewing Actual Patient Output

## What I Actually Saw

After examining slides and plan.json files across three patients (71F MCI, 32M anxiety/depression, 17M ASD), I need to revise my original report significantly. The pipeline produces far more creative variation than a simple 5-component template system can handle.

### The Good: Qwen Gets a Lot Right

| Slide | Patient | Verdict |
|-------|---------|---------|
| Title card (scene_000, 71F) | Cinematic brain + EEG waveform, warm gold lighting | **Stunning.** Hard to replicate in React without WebGL. |
| P300 voltage meters (scene_004, 71F) | Three glowing volume bars, 13.1/22.9/24.0 µV | **Clean and accurate.** Numbers, units, labels all correct. |
| Frequency puzzle (scene_011, 71F) | Two-column arrows, Central-Parietal vs Occipital | **Layout matches prompt exactly.** Values correct. Checkmark/question mark present. |
| Verdict card (scene_015, 71F) | Checklist with green checkmarks | **Simple and correct.** |
| Title card (scene_000, 17M ASD) | Neural network particles, gold text overlay | **Beautiful atmospheric quality.** |
| Behavioral improvements (scene_004, 32M) | Three bar groups with percentages | **Values correct**, but bars grow taller for faster times (semantic inversion — misleading). |
| Coherence disconnect (scene_005, 32M) | Two brains connected by bridge | Values correct, but it shows two separate brains instead of left/right within one brain (conceptual miss). |

### The Bad: Where Qwen Consistently Fails

| Slide | Patient | Failure |
|-------|---------|---------|
| Coherence values (scene_009, 71F) | Three brains with C3-CZ/C4-CZ values | **All three brains show all three sessions' data** instead of each brain showing one session. "078" instead of "0.78". |
| Bad example (scene_009, 71F) | QC-rejected version | **"Mession 2"** instead of "Session 2". Session 3 shows Session 2's values. |
| Brain coherence labels (scene_007, 71F) | Brain with connection labels | **"0.65 — Dominant"** when prompt said "0.65 — Weak". |
| Balance scale (scene_010, 71F) | Three scales for alpha ratio | Session 1 scale tilts the wrong direction. Right Frontal label missing. |

### The Pattern

**Failure correlates directly with the number of distinct data values on a single slide.** 

- 0–3 values: Qwen nails it (title cards, single-metric comparisons)
- 4–6 values: Usually correct, occasional label swaps
- 7+ values: Frequent errors in positioning, labeling, or value accuracy
- Tabular/matrix data: Almost always garbled

This is the fundamental problem. The LLM director produces scenes with 10+ specific coherence values, multi-session electrode readings, and cross-referenced labels. Qwen treats these as aesthetic suggestions, not data contracts.

---

## Why My Original "5 Components" Architecture Was Wrong

Each patient's plan.json reveals **completely different visualization designs**:

**71-year-old female (MCI):**
- Orchestra metaphor illustration
- Volume meter bars (P300 voltage)
- Horizontal timeline arrow (P300 latency)
- Dot-plot columns with error bars (reaction time)
- Abstract brain diagram with labeled connections (coherence)
- Balance scales (alpha ratio)
- Arc meter/gauge (theta-beta)
- Two-column arrow comparison (peak frequency)

**32-year-old male (anxiety/depression):**
- Calendar + data table split layout
- Three ascending bar groups with percentages
- Side-by-side brain comparison with bridge
- Brain with radiating glow (compensation metaphor)
- Topographic site-by-site table (CP P300 values)

**17-year-old male (ASD):**
- Top-down head with 19 electrode positions
- P300 amplitude trajectory (4x increase)
- Trail Making split comparison (B improves, A fluctuates)
- Connectivity paradox (coherence down at rest, up during task)
- State-dependent network flexibility diagram

The `visual_prompt` field isn't describing a template — it's a **bespoke creative brief** that changes per patient, per finding. The director invents new visualization metaphors based on what the clinical data demands.

---

## The Revised Architecture: Hybrid + Structured Data

Instead of replacing all image generation with Remotion, the migration should be **surgical**: Remotion handles the slides where deterministic data rendering matters, while preserving the pipeline's ability to generate atmospheric/artistic content.

### Tier 1: Remotion (deterministic, data-critical)

These slide types contain specific numerical values that must be pixel-perfect:

| Component | What It Renders | Data Required |
|-----------|----------------|---------------|
| `MetricComparison` | Bar charts, volume meters, session-over-session values | `{bars: [{label, value, unit, color}], target_range, title}` |
| `TimelineArrow` | Horizontal progression with labeled points and target zones | `{points: [{label, value, color, unit}], target_band, direction}` |
| `CoherenceNetwork` | Top-down brain SVG with labeled electrode connections | `{connections: [{from, to, value, color, thickness}], title, session_label}` |
| `DotPlotRange` | Values with error bars/spread indicators | `{columns: [{label, mean, spread, unit, color}], target_range}` |
| `BalanceScale` | Left/right comparison with tilt direction | `{sessions: [{label, value, tilt_direction}], target_range}` |
| `GaugeMeter` | Arc meter with needle positions | `{needles: [{label, value, color}], target_zone, scale}` |
| `SplitComparison` | Two-column with arrows showing opposing trends | `{left: {label, from, to, status}, right: {label, from, to, status}}` |
| `DataTable` | Tabular values with color-coded cells | `{headers: [], rows: [[]], highlight_rules: {}}` |
| `ChecklistSummary` | Verdict card with checkmarks and findings | `{items: [{text, status}], recommendation, tagline}` |
| `SessionTimeline` | Calendar/timeline showing session dates and intervals | `{sessions: [{label, date, interval_to_next}]}` |
| `RoadmapAgenda` | Numbered exploration list | `{title, items: [{number, text}]}` |

### Tier 2: AI Image Generation (atmospheric, non-data)

These slides have zero data values — they're purely visual/metaphorical:

- Title cards (cinematic brain imagery, neural network particles)
- Metaphor illustrations (orchestra, brain-as-concept)
- Electrode placement reference (top-down head with glowing sensors)
- Artistic brain cross-sections

For Tier 2, keep Qwen or switch to a curated library of pre-rendered background images with Remotion text overlay.

### Tier 3: Hybrid (Remotion overlay on AI background)

Some slides benefit from both:

- Brain diagram with labeled connections → SVG overlay on atmospheric brain background
- Compensation visualization → Gradient/glow background with data text overlay

---

## The Director Prompt Is the Real Work

The biggest change isn't in the rendering layer — it's in `director.py`. Currently the director outputs:

```json
{
  "narration": "...",
  "visual_prompt": "Split three-panel comparison on dark background..."
}
```

For Remotion, it needs to output:

```json
{
  "narration": "...",
  "visual_prompt": "...",  // Keep for Tier 2 slides
  "scene_type": "metric_comparison",
  "structured_data": {
    "title": "P300 Signal Voltage",
    "subtitle": "Target: 6–14 µV",
    "bars": [
      {"label": "Session 1", "value": 13.1, "unit": "µV", "color_key": "dim"},
      {"label": "Session 2", "value": 22.9, "unit": "µV", "color_key": "bright"},
      {"label": "Session 3", "value": 24.0, "unit": "µV", "color_key": "gold"}
    ],
    "target_range": {"min": 6, "max": 14}
  }
}
```

This is a **significant prompt engineering effort**. The director needs to:

1. Decide which `scene_type` fits each finding
2. Extract the exact numerical values from the qEEG analysis
3. Structure them into the correct schema for that component
4. Choose appropriate color keys and layout options
5. Still generate `visual_prompt` for Tier 2 scenes

The LLM is already doing most of this work (it embeds values in the prompt text), but the output format shifts from natural language to structured JSON with a type discriminator.

### Zod Schema for plan.json (TypeScript side)

```typescript
const SceneBase = z.object({
  id: z.number(),
  uid: z.string(),
  title: z.string(),
  narration: z.string(),
  audio_path: z.string().optional(),
});

// Tier 1: Deterministic Remotion rendering
const MetricComparisonScene = SceneBase.extend({
  scene_type: z.literal("metric_comparison"),
  structured_data: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    bars: z.array(z.object({
      label: z.string(),
      value: z.number(),
      unit: z.string(),
      color_key: z.enum(["dim", "medium", "bright", "gold", "red", "green", "orange"]),
    })),
    target_range: z.object({ min: z.number(), max: z.number() }).optional(),
  }),
});

const CoherenceNetworkScene = SceneBase.extend({
  scene_type: z.literal("coherence_network"),
  structured_data: z.object({
    title: z.string(),
    session_label: z.string(),
    connections: z.array(z.object({
      from: z.string(),  // electrode label e.g. "C3"
      to: z.string(),
      value: z.number(),
      color_key: z.enum(["strong", "weak", "dominant", "reduced", "strengthened"]),
      label: z.string().optional(),  // e.g. "Dominant" or "Reduced"
    })),
  }),
});

// Tier 2: AI-generated image
const AtmosphericScene = SceneBase.extend({
  scene_type: z.literal("atmospheric"),
  visual_prompt: z.string(),
});

// Union of all scene types
const Scene = z.discriminatedUnion("scene_type", [
  MetricComparisonScene,
  CoherenceNetworkScene,
  TimelineArrowScene,
  DotPlotRangeScene,
  BalanceScaleScene,
  GaugeMeterScene,
  SplitComparisonScene,
  DataTableScene,
  ChecklistSummaryScene,
  SessionTimelineScene,
  RoadmapAgendaScene,
  AtmosphericScene,
]);
```

---

## Cost Comparison (Revised)

For a typical 15-scene video with Kokoro TTS, assuming ~5 atmospheric + ~10 data-driven slides:

| Component | Current Pipeline | Hybrid Remotion | Full Remotion |
|-----------|-----------------|-----------------|---------------|
| Director LLM | $0.05 | $0.07 (longer structured output) | $0.07 |
| Image gen (15 scenes) | $0.045 | $0.015 (5 atmospheric only) | $0.00 |
| QC image edits (4 fixes) | $0.18 | $0.00 (data slides are deterministic) | $0.00 |
| TTS (Kokoro) | $0.00 | $0.00 | $0.00 |
| Video assembly | $0.00 | $0.00 | $0.00 |
| **Total per video** | **~$0.28** | **~$0.09** | **~$0.07** |
| **Savings** | — | **68%** | **75%** |
| **QC eliminated** | — | **100% for data slides** | **100%** |

The real savings isn't the $0.19/video — it's the **QC labor time** and the **regeneration cycles**. Every image-edit fix is a round-trip to DashScope that might introduce new errors. Remotion data slides need zero QC for text accuracy.

If using ElevenLabs instead of Kokoro, TTS dominates cost at ~$1.13/video regardless of rendering approach.

### Remotion Licensing

- **Free** for individuals and companies ≤3 people (including commercial use, unlimited renders)
- **Company License**: $25/month per seat (creators) or $0.01/render + $100/mo minimum (automators)
- You qualify for the free tier

---

## Remotion Technical Architecture

### How It Works

Remotion renders React components frame-by-frame into video:

1. **Bundle** React code into a static site
2. **Start** local HTTP server serving the bundle
3. **Open** a pool of headless Chrome instances (10+ on M4 Pro with 48GB)
4. **Screenshot** every frame in parallel across Chrome instances
5. **Stitch** frames + audio into MP4 via FFmpeg (bundled, no separate install)

Key APIs:
- `useCurrentFrame()` — returns current frame number, drives all animation
- `useVideoConfig()` — returns fps, width, height, durationInFrames
- `interpolate(frame, inputRange, outputRange)` — maps frame to animated values
- `<Audio src={audioFile} />` — plays pre-generated WAV, syncs to timeline
- `<Series>` / `<TransitionSeries>` — sequences scenes along timeline
- `calculateMetadata()` — runs before render, reads WAV durations, sets total length
- `getAudioDurationInSeconds()` — reads WAV file length for dynamic timing

### M4 Pro Rendering Performance

- 10+ concurrent Chrome instances within 48GB RAM (each ~200-500MB)
- Hardware-accelerated H.264 via **VideoToolbox**: `--hardware-acceleration=if-possible`
- Estimated render time for 15-scene, 3-minute video: **2-5 minutes**
- `npx remotion still` renders individual frames for visual QA (~1s per frame)

### Audio Integration (Your TTS Pipeline Unchanged)

Remotion does **not** have built-in TTS. Your existing `voice_gen.py` (Kokoro/ElevenLabs/Qwen3-TTS) generates WAV files before Remotion renders. The WAVs get placed in `remotion/public/audio/` and referenced by `<Audio>` components. `calculateMetadata()` reads each WAV's duration via `getAudioDurationInSeconds()` and sums them for total video length.

### Data Flow: plan.json → Remotion

```
plan.json (Python generates)
    ↓
--props=./plan.json (CLI flag)
    ↓
calculateMetadata() reads WAV durations, returns durationInFrames
    ↓
<QeegExplainerVideo> maps scenes to <Series.Sequence> components
    ↓
Each scene dispatches to correct component based on scene_type
    ↓
<Audio> component per scene plays WAV in sync
    ↓
npx remotion render → output.mp4
```

### CLI Render Command (from Python subprocess)

```bash
npx remotion render src/index.ts QeegExplainerVideo output.mp4 \
  --props=./plan.json \
  --codec=h264 \
  --hardware-acceleration=if-possible \
  --concurrency=100%
```

### Individual Frame QA

```bash
# Render just frame 0 of scene 4 as PNG for visual inspection
npx remotion still src/index.ts QeegExplainerVideo scene_004.png \
  --props=./plan.json \
  --frame=120
```

---

## Project Structure

```
/Users/davidmontgomery/local-explainer-video/
├── app.py                              # Minor changes: route by scene_type
├── core/
│   ├── director.py                     # MAJOR CHANGE: output structured_data
│   ├── voice_gen.py                    # UNCHANGED
│   ├── video_assembly.py               # REWRITE: subprocess to Remotion
│   ├── image_gen.py                    # KEEP for Tier 2 atmospheric scenes
│   └── qc_publish.py                   # SIMPLIFY: skip text QC for Tier 1
├── remotion/                           # NEW
│   ├── package.json
│   ├── tsconfig.json
│   ├── remotion.config.ts
│   ├── public/
│   │   ├── audio/                      # WAVs copied here before render
│   │   └── backgrounds/                # Pre-rendered atmospheric images
│   └── src/
│       ├── index.ts                    # registerRoot()
│       ├── Root.tsx                     # Composition + calculateMetadata()
│       ├── theme.ts                    # Shared design tokens
│       ├── types.ts                    # Zod schemas for plan.json
│       ├── QeegExplainerVideo.tsx       # Main composition
│       ├── SceneRouter.tsx             # Dispatches scene_type → component
│       └── components/
│           ├── data/
│           │   ├── MetricComparison.tsx
│           │   ├── TimelineArrow.tsx
│           │   ├── CoherenceNetwork.tsx
│           │   ├── DotPlotRange.tsx
│           │   ├── BalanceScale.tsx
│           │   ├── GaugeMeter.tsx
│           │   ├── SplitComparison.tsx
│           │   ├── DataTable.tsx
│           │   ├── ChecklistSummary.tsx
│           │   ├── SessionTimeline.tsx
│           │   └── RoadmapAgenda.tsx
│           ├── atmospheric/
│           │   └── ImageSlide.tsx       # Renders pre-generated PNG
│           └── shared/
│               ├── SlideBackground.tsx  # Dark gradient, consistent across all
│               ├── SlideTitle.tsx
│               ├── TargetRangeBand.tsx
│               └── BrainOutlineSVG.tsx  # 10-20 electrode map (reusable)
```

---

## The Brain Diagram Component: Worth Getting Right

The `CoherenceNetwork` component is the highest-impact piece. It replaces the slide type that fails most often (scene_007, 009 from patient 71F).

It renders a **top-down SVG brain outline** with:
- 19 electrode positions from the 10-20 system as circles
- Connections between electrodes as lines with variable thickness/color
- Value labels positioned along each connection line
- Session label and title

The electrode coordinates are a fixed constant (they never change across patients):

```typescript
const ELECTRODES_10_20: Record<string, {x: number, y: number}> = {
  FP1: {x: 340, y: 80}, FP2: {x: 660, y: 80},
  F7: {x: 200, y: 220}, F3: {x: 370, y: 220}, FZ: {x: 500, y: 200},
  F4: {x: 630, y: 220}, F8: {x: 800, y: 220},
  T3: {x: 120, y: 400}, C3: {x: 350, y: 380}, CZ: {x: 500, y: 370},
  C4: {x: 650, y: 380}, T4: {x: 880, y: 400},
  T5: {x: 200, y: 580}, P3: {x: 370, y: 560}, PZ: {x: 500, y: 550},
  P4: {x: 630, y: 560}, T6: {x: 800, y: 580},
  O1: {x: 380, y: 700}, O2: {x: 620, y: 700},
};
```

Connection lines use Remotion's `interpolate()` to animate from zero opacity to full, with staggered delays per connection. Color maps to coherence value via `interpolateColors()` using a diverging blue-white-red scale.

This component alone eliminates the most expensive QC failures — no more "Mession 2", no more swapped values, no more mislabeled "Dominant" vs "Weak".

---

## Migration Plan (Revised, 4 Phases)

### Phase 1: Prove the Render Loop (Week 1)

1. Install Node.js 20 LTS via nvm
2. `npx create-video@latest` in `remotion/` subfolder
3. Install Remotion Agent Skills: `npx skills add remotion-dev/skills`
4. Build ONE component: `ChecklistSummary` (simplest, uses patient 71F scene_015 data)
5. Hardcode the props, render with `npx remotion render`
6. Prove subprocess call from Python works

**Exit criterion:** Python calls Remotion, gets back an MP4 with a verdict card that matches scene_015 pixel-for-pixel on text accuracy.

### Phase 2: Build Core Components + Wire plan.json (Weeks 2-3)

Build the 11 data components in order of patient frequency:
1. `MetricComparison` (bar charts — most common slide type)
2. `RoadmapAgenda` (trivial)
3. `SessionTimeline` (trivial)
4. `TimelineArrow` (P300 latency)
5. `DotPlotRange` (reaction time with spread)
6. `CoherenceNetwork` (brain diagram — highest impact)
7. `SplitComparison` (frequency puzzle)
8. `BalanceScale`
9. `GaugeMeter`
10. `DataTable`
11. `ChecklistSummary` (already done in Phase 1)

Wire `calculateMetadata()` + `getAudioDurationInSeconds()` for dynamic duration.
Test each component with real patient data from all 3 patients reviewed.

### Phase 3: Modify the Director (Week 4)

This is the hardest phase. Update `director.py` to output `scene_type` + `structured_data`:

1. Add a scene-type taxonomy to the system prompt
2. Add Zod-compatible JSON schemas for each scene type
3. Add examples of correct structured output for each type
4. Test against all 3 reviewed patients — compare director output quality
5. Validate structured_data against Zod schemas before rendering
6. Feature flag: `USE_STRUCTURED_SCENES=true`

### Phase 4: Integration + Cleanup (Weeks 5-6)

1. Rewrite `video_assembly.py` to call Remotion for Tier 1 scenes
2. Keep `image_gen.py` for Tier 2 atmospheric scenes (or switch to background library)
3. Simplify `qc_publish.py` — skip vision AI for Tier 1 scenes
4. Add `<TransitionSeries>` with 15-frame fade between scenes
5. Side-by-side comparison: old pipeline vs new for all 3 patients
6. Remove feature flag, make Remotion the default

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Director fails to produce valid structured_data | **High** | Zod validation with fallback to atmospheric/image-gen |
| Visual quality of Remotion data slides feels "flat" vs AI imagery | **Medium** | Invest in theme.ts: gradients, glow effects, subtle animations |
| CoherenceNetwork SVG doesn't match the cinematic brain aesthetic | **Medium** | Layer SVG electrodes over a pre-rendered brain background image |
| Remotion v5 migration | **Low** | Pin v4.0.425 with exact versions |
| Node.js toolchain friction alongside Python | **Low** | nvm + subprocess isolation, separate directory |
| Audio sync issues | **Low** | Normalize WAVs to 44.1kHz before render |

The biggest risk is the **director prompt change**. If the LLM can't reliably produce structured scene data with correct values, the whole approach falls apart. Mitigation: every scene validates against its Zod schema before render. If validation fails, fall back to `atmospheric` type and generate an image with the existing pipeline.

---

## What This Means for the "Cool Someday" EEG Animation

The `CoherenceNetwork` component is 80% of the way to animated voltage flow. Once the SVG brain outline with electrode positions exists, adding per-frame color animation is straightforward:

```typescript
// Future: animated voltage heatmap
const voltageAtFrame = voltageTimeSeries[currentFrame];
electrodes.map(e => ({
  ...e,
  fill: interpolateColors(
    voltageAtFrame[e.label],
    [-3, 0, 3],  // z-score range
    ['#2563EB', '#FFFFFF', '#DC2626']  // blue → white → red
  )
}));
```

This requires per-frame electrode data as an input array — the director would need to output a time series instead of a single snapshot. But the rendering infrastructure is the same component.

---

## Summary

The original report designed a clean theoretical architecture without looking at the actual patient output. Having now reviewed slides from 3 patients with different conditions, ages, and clinical findings:

1. **Qwen produces beautiful atmospheric/title slides** — don't throw those away
2. **Qwen fails predictably on data-dense slides** — "Mession 2", swapped values, mislabeled connections
3. **The slide vocabulary is much richer than 5 templates** — 11+ distinct visualization types across patients
4. **The director prompt change is the hardest part** — shifting from free-text visual_prompt to structured_data with correct scene typing
5. **A hybrid approach beats total replacement** — Remotion for data accuracy, AI gen for visual atmosphere
6. **Cost savings are real but modest** ($0.19/video with Kokoro) — the actual win is eliminating QC labor and regeneration cycles

Start with Phase 1: prove one Remotion component renders correctly via Python subprocess. Everything else builds from there.
