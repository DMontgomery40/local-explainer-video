# Deterministic Template Pipeline: Implementation Handoff

**To: Claude Code & Codex**
**From: David (via Claude Opus research sessions)**
**Date: 2026-02-22**
**Repo: `/Users/davidmontgomery/local-explainer-video`**

This is a meticulously prepared implementation brief synthesized from three prior research sessions: two deep-dive analyses by Claude Opus (examining actual rendered patient output across 3 patients with pixel-level visual inspection) and one exhaustive corpus census by Codex (covering all 11 patients with plan.json, 169 total scenes, full archetype taxonomy). The attached Codex research report (`.codex/research/remotion_hybrid_deep_research_2026-02-22.md`) contains the complete archetype taxonomy, scene-level CSV mapping, and Remotion ecosystem intelligence. Read it thoroughly before beginning — but understand that **some of its recommendations have been superseded** by the architectural decisions documented here.

---

## The Problem (Concise)

The current pipeline generates stunning, cinematically polished slide images via Qwen-image-2512. When the slide has ≤3 data values, Qwen nails it — text is accurate, layout is clean. When the slide has 4+ data values, Qwen fails predictably: misspellings ("Mession 2"), transposed values (Session 3 displaying Session 2's data), semantic label inversions ("Dominant" where it should say "Weak"), and garbled tabular layouts. Every failure requires expensive QC cycles (Gemini vision review → Qwen image-edit → re-review) that cost more than the original generation and sometimes introduce new errors.

The visual quality of Qwen's output is not the problem. The text accuracy is the problem.

---

## The Solution (Architectural)

**Generate gorgeous template images via Qwen ONCE. Store them as static assets. For every future patient video, composite deterministic text onto those templates with pixel-precise placement. Qwen never renders patient-specific text again.**

The templates retain all of Qwen's visual splendor — glowing meters, cinematic brain renders, neural particle backgrounds, warm color progressions, atmospheric depth — while a deterministic text compositor handles every label, value, unit, site name, session identifier, and annotation. Text rendering quality must achieve parity with Qwen's native typography: proper anti-aliasing, drop shadows, glow effects where appropriate, professional font choices.

### Three-Layer Architecture

**Layer 1: Static Template Library**
~60-120 pre-rendered Qwen images across 22 archetype categories. These are the visual "chrome" — backgrounds, glowing shapes, brain renders, gauge arcs, meter frames — with NO text, NO numbers, NO patient-specific data baked in. Generated once, curated by a human, stored permanently.

**Layer 2: Deterministic Text Compositor**
A compositing engine that stamps text onto templates at defined anchor positions. Every on-screen character — every value, label, unit, electrode name, session identifier — originates from a structured data payload, not from an image model. The compositor must produce text that is visually indistinguishable from Qwen's native text rendering: anti-aliased, properly shadowed, color-matched to the template's palette.

**Layer 3: Orchestration**
A modified director (or downstream subagent) that outputs structured scene data including `scene_type` and `structured_data`, paired with a template selector that maps each scene to the appropriate template image + anchor configuration.

### On Remotion

The original research explored a full Remotion migration (React components rendered frame-by-frame via headless Chrome). Remotion remains a **viable and potentially superior approach** for the compositor layer, particularly because:
- It would enable light animations (fade-ins, value counters, slide transitions) essentially for free
- React + CSS gives access to sophisticated text rendering (font-face, text-shadow, letter-spacing) without fighting Pillow's limitations
- The `<Img>` component can use the Qwen templates as backgrounds while React components overlay the data text
- Remotion Agent Skills are now first-party documented, making Claude Code + Remotion a well-trodden workflow

However, a pure Python approach (Pillow/Pycairo + MoviePy) keeps the stack unified and avoids Node.js toolchain complexity.

**You must decide** which compositor approach to use. Evaluate both. Present your recommendation with rationale. If you choose Remotion, follow the ecosystem intelligence in the Codex report — particularly the corrections about `getAudioDurationInSeconds()` deprecation, Mediabunny migration, Chrome Headless Shell defaults, and hardware acceleration constraints. If you choose PIL/Pycairo, demonstrate that text rendering quality achieves parity with Qwen's native output.

---

## The Template Library: What Must Be Built

### Archetype Taxonomy (22 Types)

Codex's corpus census identified 20 data-bearing archetypes plus 3 atmospheric types. However, the CSV reveals 2 additional types that were silently dropped from the taxonomy:

- **WAVEFORM** (ERP voltage panel — patient 11-21-1996-0, scene_010)
- **RADIAL** (KPI ring display — patient 01-01-1991-0, scene_013)

The complete data-bearing archetype set is therefore **22 types**, not 20. Refer to the Codex report for the full taxonomy with frequency counts and required props schemas. The archetype codes from the CSV are:

`SPLIT`, `MULTI_TREND`, `SUMMARY`, `ROADMAP`, `TIMELINE`, `GAUGE`, `COH_MAP`, `HEMI`, `STATE`, `BASELINE_SPLIT`, `MEASURE`, `FUTURE`, `LINE_TRAJECTORY`, `BAR`, `COH_SEQ`, `DOTPLOT`, `PATHWAY`, `REGIONAL`, `TABLE`, `QC`, `WAVEFORM`, `RADIAL`

Plus 3 atmospheric: `AT_TITLE`, `AT_METAPHOR`, `AT_MECHANISM`

### Template Variants and the Cardinality Problem

Templates must be **creatively neutral** — usable across different patients with different magnitudes. A gauge showing a low theta/beta ratio (1.2) and a gauge showing a high one (4.8) should use different template variants (one with the visual indicator in the "green" zone, one in the "red" zone), but neither should contain actual numbers.

Some archetypes require multiple variants not just for magnitude but for **element count**:
- `BAR` / `MULTI_TREND`: Most patients have 3 sessions, but some might have 2. Templates for 2-bar and 3-bar layouts.
- `COH_MAP`: Number of electrode connections varies wildly (2-8+). This may need a more dynamic approach — perhaps the brain background is a template but the connection lines and labels are drawn programmatically.
- `SPLIT`: Always exactly 2 panels (left vs right), so a single template works.
- `SUMMARY` / `ROADMAP`: Variable number of bullet items (3-8). The template is the background; text layout must be dynamic.

**The heuristic**: If the visual chrome is fixed-count (gauges, bars, split panels), make template variants per count. If the visual chrome is variable-count (bullet lists, connection maps, table rows), use a background template + programmatic overlay for the variable elements.

Generate as many template variants as needed. The constraint is not quantity but **indexability** — the system choosing templates needs to be able to select the right one without browsing 1000 images. Clear, systematic naming conventions and a JSON manifest are essential.

### Template Generation Strategy

Each template must be generated via Qwen with carefully crafted prompts that produce the visual chrome WITHOUT any text. This is non-trivial — Qwen's instinct is to add text to everything.

Prompting approach:
- Explicitly specify "NO TEXT. NO NUMBERS. NO LABELS. NO WORDS OF ANY KIND."
- Describe the visual elements in concrete, non-medical terms (same principle as the existing director prompt)
- Specify exact layout positions where visual elements should appear (e.g., "three vertical glowing cylindrical meters evenly spaced, left meter 25% filled, center meter 75% filled, right meter 90% filled")
- Use `negative_prompt` on DashScope: "text, numbers, labels, words, letters, writing, captions, titles"
- Generate 3-6 variants per prompt and curate the best
- Final images must be 1664×928 (matching existing pipeline)

### Exemplar Re-Curation

Codex's exemplar selections were based on metadata heuristics, not genuine aesthetic evaluation. **You must redo the exemplar selection.** For every archetype, visually inspect the actual rendered images across all patients and identify the 2-3 most visually stunning examples. These become the aesthetic targets that your templates must match or exceed.

Pay particular attention to:
- **09-05-1954-0** (71F MCI) — consistently the highest visual quality across the corpus. Her title card (scene_000), P300 voltage meters (scene_004), and coherence maps (scene_007, 008) are particularly strong.
- **07-14-2008-0** (17M ASD) — beautiful neural network particle aesthetic on title card, strong state flexibility visualizations.
- **11-21-1996-0** — good coherence progression and dot-plot examples.

Your exemplar picks guide template generation. If the best coherence map in the corpus has a specific brain angle, neural particle density, and color temperature, the text-free template should evoke that same aesthetic.

---

## The Anchor System: Text Placement

Every template image needs a companion JSON file defining where text gets stamped. The schema should look something like:

```json
{
  "template_id": "bar_3session_ascending",
  "archetype": "BAR",
  "variant": "3bar_ascending",
  "dimensions": [1664, 928],
  "anchors": {
    "title": {
      "position": [832, 60],
      "align": "center",
      "font": "Inter-Bold",
      "size": 64,
      "color": "#FFFFFF",
      "shadow": {"offset": [2, 2], "blur": 4, "color": "#00000080"}
    },
    "bar_1_label": {
      "position": [277, 180],
      "align": "center",
      "font": "Inter-Regular",
      "size": 28,
      "color": "#8CB4D0"
    },
    "bar_1_value": {
      "position": [277, 780],
      "align": "center",
      "font": "Inter-Bold",
      "size": 72,
      "color": "#8CB4D0"
    },
    "bar_2_label": { "..." : "..." },
    "bar_2_value": { "..." : "..." },
    "bar_3_label": { "..." : "..." },
    "bar_3_value": { "..." : "..." },
    "subtitle": {
      "position": [832, 890],
      "align": "center",
      "font": "Inter-Regular",
      "size": 24,
      "color": "#888888"
    }
  },
  "palette": {
    "accent_1": "#8CB4D0",
    "accent_2": "#C0E0FF",
    "accent_3": "#D4A84B"
  }
}
```

This anchor JSON is what makes text placement deterministic. The compositor reads it, reads the structured_data from plan.json, and stamps text at exactly the right coordinates with exactly the right styling. No LLM involved in the rendering step.

**Important**: anchor coordinates should be defined AFTER templates are generated, not before. Generate the template, then annotate where text belongs. Trying to specify coordinates before seeing the image will produce misaligned results.

---

## The Pipeline: End-to-End Flow

### Current Pipeline
```
qEEG report → Director (LLM) → plan.json (visual_prompt per scene)
→ Qwen image gen (per scene) → QC (Gemini vision) → image edit (fix errors)
→ Voice gen (Kokoro TTS) → MoviePy assembly → MP4
```

### New Pipeline
```
qEEG report → Director (LLM) → plan.json (scene_type + structured_data per scene)
→ Template Selector (subagent or heuristic) → picks template_id per scene
→ Text Compositor → stamps structured_data onto template at anchor positions
→ Voice gen (Kokoro TTS, unchanged)
→ Video assembly (MoviePy or Remotion) → MP4
```

### What Changes

| Component | Current | New |
|-----------|---------|-----|
| `director.py` | Outputs `visual_prompt` (prose) | Outputs `scene_type` + `structured_data` (typed JSON) |
| `image_gen.py` | Called per scene, every patient | Called ONCE during template generation; dormant in production |
| Template library | Does not exist | ~60-120 curated Qwen images + anchor JSONs |
| Template selector | Does not exist | New component: maps scene_type + data characteristics → template_id |
| Text compositor | Does not exist | New component: stamps text onto template images |
| `voice_gen.py` | Unchanged | Unchanged |
| `video_assembly.py` | MoviePy with hard cuts | MoviePy (or Remotion if you choose that path) |
| `qc_publish.py` | Gemini vision QC + image edit | **Dramatically simplified** — text accuracy is guaranteed by construction |

### The Director Modification

The director prompt (`prompts/director_system.txt`) currently instructs the LLM to write `visual_prompt` as a cinematographic prose description. This must change.

The director must now output:

```json
{
  "id": 4,
  "uid": "abcd1234",
  "title": "Signal Strength: Nearly Doubled",
  "narration": "...",
  "scene_type": "bar_volume_chart",
  "structured_data": {
    "title": "P300 Signal Voltage",
    "subtitle": "Target: 6–14 µV",
    "bars": [
      {"label": "Session 1", "value": 13.1, "unit": "µV"},
      {"label": "Session 2", "value": 22.9, "unit": "µV"},
      {"label": "Session 3", "value": 24.0, "unit": "µV"}
    ],
    "target_range": {"min": 6, "max": 14},
    "trend": "ascending"
  }
}
```

The `visual_prompt` field is **retained** only for atmospheric scenes (AT_TITLE, AT_METAPHOR, AT_MECHANISM) where Qwen may still generate per-patient imagery if desired — though even title cards could use a static template with stamped text.

The Codex report contains example structured_data schemas for `bar_volume_chart`, `coherence_network_map`, `split_opposing_trends`, and `dotplot_variability`. Extend these to cover all 22 archetypes. Use Zod (if Remotion) or Pydantic (if Python) for validation — every scene's structured_data must validate against its archetype's schema before rendering.

### The Template Selector

This is the new decision-making component. Given a scene's `scene_type` and `structured_data`, it must select the correct `template_id` from the library.

**Two viable approaches:**

**A) Programmatic heuristic (preferred for v1):**
- `scene_type` directly maps to an archetype folder
- Within that folder, variant selection uses simple rules from structured_data:
  - `BAR` with 3 bars + ascending values → `bar_3session_ascending`
  - `GAUGE` with value < target_min → `gauge_below_target`
  - `GAUGE` with value within target → `gauge_in_zone`
  - `COH_MAP` with ≤4 edges → `coh_map_sparse`
  - `COH_MAP` with >4 edges → `coh_map_dense`
- Fast, deterministic, zero LLM cost, no context window constraints
- Implemented as a pure Python function with a lookup table

**B) Subagent selector (for edge cases or later refinement):**
- A lightweight LLM subagent with tool access (`ls` the template directory, `view` candidate thumbnails)
- Given the structured_data, selects between 2-3 candidate templates
- More flexible, can handle novel data patterns
- More expensive, slower, less deterministic
- **Use subagents and skills** — David specifically noted these are underutilized in this pipeline

**Recommendation**: Start with (A) for all archetypes. Add (B) as a fallback only for archetypes where the heuristic produces ambiguous results. The heuristic should be implemented as a Claude Code skill (`.claude/skills/`) so it's discoverable and maintainable.

### The Text Compositor

This is the rendering engine. Given:
- A template image (PNG, 1664×928)
- An anchor JSON (text positions, fonts, colors, effects)
- A structured_data payload (the actual values to stamp)

It produces a final composited image with all text rendered at anchor positions.

**Requirements:**
- Anti-aliased text rendering (NOT Pillow's basic `ImageDraw.text()` which is primitive)
- Drop shadows and glow effects matching the template's aesthetic
- Support for the font families used in the existing Qwen output (Inter, Montserrat, or similar high-quality sans-serif)
- Dynamic text sizing: if a value is "13.1 µV" vs "1,342.7 ms", the font size may need to scale to fit the anchor region
- Color inheritance from the anchor definition, with optional per-anchor overrides from structured_data
- Support for Unicode characters: µV, →, %, ±, °

**Library options to evaluate:**
- `Pillow` with `ImageFont.truetype()` — baseline, may need augmentation for shadows/glow
- `Pycairo` — much richer text rendering, supports gradients, shadows, anti-aliasing natively
- `Skia-Python` (skia-python) — Google's rendering engine, overkill but extremely capable
- `wand` (ImageMagick bindings) — composite operations, rich text rendering
- Remotion `<AbsoluteFill>` + `<Img>` + styled `<div>` elements — if you go the Remotion route, text rendering is just CSS, which handles all of the above trivially

**You must evaluate** at least two of these options and demonstrate text rendering parity with Qwen's output quality. Render the P300 voltage meters slide (patient 09-05-1954-0, scene_004) as a proof-of-concept using both the template background and your compositor, and compare against the original.

---

## The Storyboard Output

After the director generates plan.json and the compositor produces all slide images, the Streamlit app should display:

**The full storyboard with final composited images** — template + deterministic text already placed. What you see in Streamlit is what goes into the video. No more "generate → QC → edit → re-QC" cycle. The images are correct by construction.

The existing Streamlit UI (Step 2 in `app.py`) already shows a scene-by-scene storyboard with images, narration, and controls. The compositor runs as part of image generation, so the flow is:

```
User pastes qEEG report → Director generates plan.json → Template selector picks templates
→ Compositor stamps text → Streamlit displays final composited slides → User approves
→ Voice gen → Video assembly → MP4
```

If a slide looks wrong, the user can:
1. Edit the structured_data (fix a value, change a label) and re-composite — instant, no API call
2. Switch to a different template variant — also instant
3. For atmospheric scenes only, regenerate via Qwen as before

---

## Scope and Phasing

### Phase 1: Proof of Concept (Target: 1 week)
1. Pick ONE archetype — `BAR` (bar/volume meter chart) — the P300 voltage meters slide
2. Generate 3-5 text-free template variants via Qwen prompting
3. Define anchor JSON for the best template
4. Build the text compositor (evaluate library options, render proof-of-concept)
5. Side-by-side comparison: original Qwen slide vs template+compositor slide
6. **Exit criterion**: composited slide is visually indistinguishable from original at normal viewing distance

### Phase 2: Template Library Generation (2-3 weeks)
1. Redo exemplar curation across all 22 archetypes (actually look at the images)
2. Design Qwen prompts for text-free template generation per archetype
3. Generate and curate templates — expect 3-10 per archetype, ~60-120 total
4. Define anchor JSONs for every template
5. Build template manifest (JSON index of all templates with metadata for selector)
6. Test across the 3 most diverse patients: 09-05-1954-0 (71F MCI), 01-01-1991-0 (32M anxiety), 07-14-2008-0 (17M ASD)

### Phase 3: Director Modification (1-2 weeks)
1. Modify `prompts/director_system.txt` to output `scene_type` + `structured_data`
2. Add Pydantic/Zod schemas for all 22 archetype data payloads
3. Validate director output against schemas before proceeding
4. Test against all 11 patients in corpus — every plan.json must validate
5. Feature flag: `USE_TEMPLATE_PIPELINE=true` in `.env`
6. Fallback: if director outputs an unrecognized scene_type or invalid structured_data, fall back to legacy Qwen generation for that scene

### Phase 4: Integration (1 week)
1. Wire template selector into pipeline
2. Wire compositor into Streamlit app (replace image_gen calls for data-bearing scenes)
3. Simplify QC pipeline — data-bearing slides skip Gemini vision QC entirely
4. Side-by-side comparison: old pipeline vs new for all 11 patients
5. Remove feature flag, make template pipeline default

### Phase 5 (Optional): Remotion Animations
If the compositor was built with Remotion, add:
- Value counter animations (numbers counting up to final value)
- Fade-in transitions between scenes (15-frame crossfade)
- Staggered element reveals (title appears, then bars animate up, then values fade in)
- The infrastructure is already there; this is just component enhancement

---

## Constraints

- **Do not break the existing pipeline.** Feature-flag the new path. Legacy Qwen generation must remain functional for atmospheric scenes and as a fallback.
- **Template library is a curated asset.** Templates are generated, reviewed by David, and committed to the repo. They are not generated at runtime.
- **Every displayed number must originate from structured_data.** Zero exceptions. If a number appears on screen, it came from the data payload, not from an image model.
- **Text rendering parity.** The composited text must look as good as Qwen's native text. If it looks obviously pasted-on, the approach has failed.
- **Subagents and skills are first-class tools.** The template selector, compositor, and any future automation should be implemented as skills (`.claude/skills/`) and leverage subagent patterns where appropriate. These are powerful architectural primitives that are vastly underutilized in this pipeline.
- **Python 3.10 required** for Kokoro TTS compatibility.
- **Target resolution: 1664×928** (16:9, matching existing pipeline).
- **No PHI** — only de-identified metrics.

---

## Files to Read Before Starting

**Essential (read in this order):**
1. This handoff document
2. `.codex/research/remotion_hybrid_deep_research_2026-02-22.md` — full Codex research report with archetype taxonomy, schemas, frequency counts, Remotion intelligence
3. `.codex/research/data/local_explainer_scene_archetypes_manual.csv` — complete scene-to-archetype mapping for all 169 scenes
4. `CLAUDE.md` — project conventions, pipeline architecture, environment setup
5. `prompts/director_system.txt` — current director prompt (must be modified in Phase 3)
6. `core/director.py` — current storyboard generation code
7. `core/image_gen.py` — current Qwen image generation (understand what it does before replacing it)

**Reference (consult as needed):**
- `core/video_assembly.py` — MoviePy assembly (may need modification if Remotion is chosen)
- `core/voice_gen.py` — TTS generation (unchanged)
- `core/qc_publish.py` — QC pipeline (will be simplified)
- `app.py` — Streamlit UI (will need UI integration for template selection)
- Any patient's `projects/<id>/plan.json` — see the current scene format
- Any patient's `projects/<id>/images/` — see the current rendered output
- `.codex/research/contact-sheets/` — quick visual overview of all patients' slides

**Examine visually (actually decode and look at these images):**
- `projects/09-05-1954-0/images/scene_000.png` — title card aesthetic target
- `projects/09-05-1954-0/images/scene_004.png` — P300 voltage meters (Phase 1 proof-of-concept target)
- `projects/09-05-1954-0/images/scene_007.png` — coherence map (gorgeous visual, wrong labels — exactly the problem we're solving)
- `projects/07-14-2008-0/images/scene_000.png` — alternate title card aesthetic
- `projects/07-14-2008-0/images/scene_008.png` — state flexibility visualization

---

## Open Questions for You to Resolve

1. **Remotion vs PIL/Pycairo for the compositor** — evaluate both, recommend one with rationale. The animations upside of Remotion is real but so is the stack simplicity of staying in Python. Make the case.

2. **Variable-cardinality archetypes** — for `COH_MAP`, `SUMMARY`, `ROADMAP`, `TABLE`, where element count varies per patient: do you generate template variants per count, or use a hybrid approach (static background + programmatic overlay for variable elements)? The latter is more flexible but harder to match Qwen's aesthetic. Propose a solution.

3. **Template naming and manifest structure** — design a naming convention and JSON manifest that makes the template library navigable for both humans (David reviewing) and machines (the heuristic selector). Consider that this library will grow over time.

4. **Font selection and licensing** — identify 2-3 high-quality sans-serif fonts for the compositor that (a) match the aesthetic of Qwen's typical output, (b) are open-source or freely licensed, and (c) render well at the sizes needed (24px captions to 72px values).

5. **The atmospheric scene question** — title cards currently use Qwen to generate per-patient imagery with the patient-specific title baked in. David wants ONE beautiful title card background reused for every patient, with the title stamped deterministically. Same for roadmap slides. Should these remain as `AT_TITLE` and `ROADMAP` archetypes with templates, or should they be handled differently? (David's preference: template + stamp.)

6. **Who "decides" the visualization approach** — the current director writes prose `visual_prompt`. The new director must output `scene_type`. Is this a modification to the existing director prompt, or should a separate downstream agent receive the director's narrative storyboard and map it to scene_types? David suggests a subagent approach might be cleaner. Evaluate both options.

---

## Success Criteria

The implementation is complete when:

1. A qEEG report can be pasted into Streamlit and produce a full storyboard where every data-bearing slide uses a template + deterministic text — no Qwen generation, no QC needed for text accuracy.
2. The composited slides are visually indistinguishable from the best current Qwen output at normal viewing distance.
3. Every number, label, electrode name, and session identifier on screen is provably correct — traced directly from structured_data in plan.json.
4. The template library covers all 22 data-bearing archetypes with sufficient variants for the full range of clinical presentations observed in the 11-patient corpus.
5. The pipeline produces a valid MP4 video end-to-end with the same voice generation and assembly quality as the current system.
6. Atmospheric scenes (title cards, metaphors, mechanism illustrations) still look beautiful — whether via static templates or retained Qwen generation.
7. The QC pipeline is simplified — data-bearing slides skip vision QC entirely since text accuracy is guaranteed by construction.
