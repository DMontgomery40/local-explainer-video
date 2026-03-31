# HANDOFF PROMPT — Director System Prompt Image Quality Evaluation

**Date**: 2026-03-04
**Status**: INCOMPLETE — previous agent wasted two full sessions evaluating the WRONG thing
**Priority**: Generate images, compare them visually, pick a winner

---

## THE SINGLE MOST IMPORTANT THING TO UNDERSTAND

**THE TASK IS ABOUT IMAGE QUALITY. NOT NARRATION. IMAGES.**

The narration from the original Anthropic runs was already phenomenal. It just needed to be shorter (950–1,100 words for ~6 min videos). That's a trivial word-count enforcement fix that is ALREADY DONE in the patched prompts.

The ENTIRE exercise — creating three prompt variants, running them, comparing output — exists to answer ONE question:

> **Which director-system prompt variant produces the best RENDERED IMAGES from Qwen (`qwen/qwen-image-2512`)?**

The previous agent spent TWO FULL EVALUATION ROUNDS scoring narration text, writing analogy quality comparisons, and never once generated or looked at a single image from the new prompts. The user gave the critical Qwen MoE context THREE TIMES and the agent ignored it every time.

---

## WHAT HAPPENED (BRUTAL HONESTY)

### What the user asked for
1. Create 3 new director-system prompt versions that maintain image quality while enforcing word count
2. Run each variant to generate storyboards (narrative-only, fast)
3. **Generate images from each variant's visual prompts**
4. **Compare those images against the phenomenal baseline images in `projects/01-01-2013-0/images/`**
5. Score image quality, recommend a winner

### What the previous agent actually did
1. Created 3 prompt variants (v002, v003, v004) — this was done correctly
2. Ran each variant to generate storyboards — done correctly
3. Evaluated NARRATION TEXT quality for all 3 variants — WRONG, nobody asked for this
4. Got yelled at for narration quality issues (no analogies for a 13-year-old ADHD patient)
5. Added a "YOUR AUDIENCE" section to all 3 prompts and re-ran — reasonable but still not the point
6. Evaluated NARRATION TEXT quality AGAIN for all 3 patched variants — STILL WRONG
7. Got yelled at AGAIN because the task was always about IMAGES
8. Finally started to look at baseline images from `projects/01-01-2013-0/images/`
9. Then wrote a CUSTOM GENERATION SCRIPT instead of using the versioned generation commands that were LITERALLY IN THE HANDOFF PROMPT the user provided
10. Got yelled at a THIRD TIME, which is where we are now

### What was NEVER done
- Never generated a single image from any of the new prompt variants
- Never compared any new image against the baseline
- Never evaluated whether v003's compositional complexity language actually maintains Qwen's quality
- Never answered the core question the user has been asking since the beginning

---

## THE QWEN MoE CONTEXT — THIS IS LOAD-BEARING, READ IT CAREFULLY

Qwen (`qwen/qwen-image-2512`) uses a Mixture of Experts (MoE) architecture. The MoE router allocates more compute/expert capacity to prompts that contain more complex, detailed, domain-specific language.

**What this means in practice:**
- When visual prompts are COMPLEX and DETAILED (lots of specific compositional language, technical terminology, precise layout descriptions), Qwen activates more experts and produces HIGH QUALITY images
- When visual prompts are SIMPLIFIED (generic descriptions, fewer details, plain language), Qwen activates fewer experts and produces GENERIC, LOW QUALITY images that look like stock photos or hospital brochure clip art

**The original v001 prompts** contained electrode codes (O1, O2, CZ, FZ), brain region names, specific µV values, coherence numbers, and very detailed layout descriptions. This complexity triggered Qwen's MoE router to allocate maximum compute, producing phenomenal images.

**The problem:** Electrode codes like "O1" and "CZ" are meaningless to patients watching the video. The text on the slides is gibberish to a non-neurologist. But removing that complexity collapses image quality.

**The hypothesis being tested (v003 specifically):** Can we REPLACE electrode-code complexity with CINEMATOGRAPHIC COMPOSITION complexity (volumetric rays, subsurface scattering, bokeh particle fields, chromatic aberration, etc.) and maintain the same MoE quality routing? The artistic composition language should be equally "complex" to the MoE router but produces slides that look premium without meaningless electrode labels.

**THIS HYPOTHESIS HAS NEVER BEEN TESTED WITH ACTUAL IMAGES.**

---

## CURRENT STATE OF FILES

### Three prompt variants (all patched with "YOUR AUDIENCE" section)

```
prompts/director_system_versions/v002_word_budget_strict/director_system.txt
prompts/director_system_versions/v003_region_cinematic/director_system.txt
prompts/director_system_versions/v004_narrative_depth/director_system.txt
```

- **v002**: Pure word count enforcement. Visual prompts are essentially the same style as v001 (electrode codes, brain regions, specific data labels). Tests whether word count alone was the issue.
- **v003**: THE KEY VARIANT. Word count + visual prompt complexity swap. Replaces electrode-code complexity with cinematographic composition language (volumetric rays, subsurface scattering, bokeh, color temperature shifts, etc.). This is the one that tests the MoE hypothesis.
- **v004**: Fewer scenes (12–13 instead of 15), more narrative depth per scene. Visual prompts similar to v001 style. Tests whether fewer, richer scenes produce better output.

### Six storyboard outputs (already generated, narration is fine, USE THESE)

**First round (pre-audience-patch):**
```
/tmp/storyboard_v002_word_budget_strict.json  — 15 scenes, 1036 words
/tmp/storyboard_v003_region_cinematic.json    — 16 scenes, 1082 words
/tmp/storyboard_v004_narrative_depth.json     — 12 scenes, 994 words
```

**Second round (post-audience-patch, THESE ARE THE ONES TO USE):**
```
/tmp/storyboard_v002_patched.json  — 15 scenes, 953 words
/tmp/storyboard_v003_patched.json  — 15 scenes, 1064 words
/tmp/storyboard_v004_patched.json  — 12 scenes, 1048 words
```

### Baseline images (the quality benchmark — THESE ARE PHENOMENAL)

```
projects/01-01-2013-0/images/scene_000.png through scene_014.png
```

These were generated by the v001 prompt (original Anthropic run). They include:
- **scene_004.png**: "The Alpha Awakening" — three side-by-side 3D brains with µV values (15, 63, 45), glowing posterior regions, +320% arrow. Premium quality.
- **scene_005.png**: "Left Hemisphere Catches Up" — two brains from above, P3/P4 µV labels, +228% arrow. The user specifically showed this image as the quality benchmark.
- **scene_006.png**: "Session Two's Wild Ride" — network disruption diagram with F/C/P/O nodes and thick/thin connection lines
- **scene_009.png**: "N100 Speed vs Power" — stopwatch + battery metaphors with session values
- **scene_012.png**: "Data Quality Considerations" — clean info card with warning icons

The plan.json with the original visual prompts that generated these is at:
```
projects/01-01-2013-0/plan.json
```

### What does BAD look like

The user also showed an image from a previous pipeline iteration — a "Medical Detective Style" notebook doodle with garbled text, generic office imagery, and completely unusable rendering. That's what happens when Qwen gets weak prompts. The entire point is to never regress to that.

---

## WHAT YOU NEED TO DO

### Step 1: Generate images from each patched variant

Use the EXISTING generation command from the handoff prompt. Do NOT write custom scripts.

For each variant, pick 3–4 KEY SCENES that test different visual types:
- A **brain visualization** scene (3D brains with µV values — tests rendering quality)
- A **network/connectivity** scene (diagrams with connection lines — tests compositional complexity)
- A **data comparison** scene (before/after, split panels — tests layout precision)
- Optionally a **title card** (tests editorial design quality)

The command to generate images from a storyboard JSON:

```bash
cd /Users/davidmontgomery/local-explainer-video
/opt/homebrew/bin/python3.10 - <<'PY'
import json
from pathlib import Path
from dotenv import load_dotenv
from core.image_gen import generate_image

load_dotenv('.env', override=True)

# Load patched storyboard
plan = json.loads(Path('/tmp/storyboard_v003_patched.json').read_text())

# Pick specific scene IDs to generate (adjust per variant)
scene_ids_to_test = [0, 4, 7, 9]  # title, brain power, network, left hemi

output_dir = Path('/tmp/image_comparison/v003')
output_dir.mkdir(parents=True, exist_ok=True)

for scene in plan['scenes']:
    if scene['id'] in scene_ids_to_test:
        prompt = scene['visual_prompt']
        out_path = output_dir / f"scene_{scene['id']:03d}.png"
        print(f"[GEN] scene {scene['id']}: {scene['title']}")
        generate_image(prompt, out_path)
        print(f"[OK]  {out_path} ({out_path.stat().st_size} bytes)")

print('Done.')
PY
```

Adjust the storyboard path and output dir for each variant:
- v002: `/tmp/storyboard_v002_patched.json` → `/tmp/image_comparison/v002/`
- v003: `/tmp/storyboard_v003_patched.json` → `/tmp/image_comparison/v003/`
- v004: `/tmp/storyboard_v004_patched.json` → `/tmp/image_comparison/v004/`

**IMPORTANT**: Pick COMPARABLE scenes across variants. The scene IDs differ because the storyboards have different structures:

| Concept | v002 scene | v003 scene | v004 scene | Baseline scene |
|---------|-----------|-----------|-----------|---------------|
| Title card | 0 | 0 | 0 | 0 |
| Alpha power / brain viz | 4 | 4 | 4 | 4 |
| Network / coherence | 5 | 7 | 5 | 6 |
| Focus dial / balance | 7 | 5 or 6 | 7 | N/A (split across 2 scenes) |
| Left hemisphere | N/A | 9 | N/A | 5 |

### Step 2: LOOK at the images

Actually read/view the generated PNG files. Compare them visually against the baseline images in `projects/01-01-2013-0/images/`.

Evaluate each image on:
1. **Rendering quality**: Does it look premium and cinematic, or generic and stock-photo-like?
2. **Text accuracy**: Are labels, numbers, and titles rendered correctly? (Qwen is strong at text but not perfect)
3. **Compositional clarity**: Can you understand the data story from the image alone?
4. **Brain rendering**: Do the brain visualizations look like the phenomenal baseline (3D, glowing, detailed) or like flat clip art?
5. **Overall feel**: Would a patient watching this feel like they're getting a premium, trustworthy medical explanation?

### Step 3: Score and compare

For each variant, score:
- Image rendering quality (1-10)
- Text/label accuracy (1-10)
- Compositional clarity (1-10)
- Consistency with baseline quality (1-10)

The KEY comparison is **v003 vs v002/v004**. If v003's cinematographic complexity language produces images as good as (or better than) the electrode-code-rich prompts, that's the winner — because it achieves the same image quality WITHOUT putting meaningless electrode codes on patient-facing slides.

### Step 4: Recommend a winner

Based on ACTUAL IMAGE OUTPUT, not narration text.

---

## SCENE-BY-SCENE VISUAL PROMPT COMPARISON

To save you time, here are the visual prompts for the alpha power scene (scene 4) across all variants, so you can see the style difference:

### Baseline (v001) — scene 4 "The Alpha Awakening"
```
Three brain diagrams side by side, top-down view, showing posterior regions glowing. Left brain labeled "Session 1" with dim blue glow and text "15 μV" below. Center brain labeled "Session 2" with intense bright glow and text "63 μV" below, with upward arrow showing "+320%" in gold. Right brain labeled "Session 3" with strong glow and text "45 μV" below. Top banner text reading "The Alpha Awakening". Bottom text "Posterior Alpha Magnitude". Dramatic lighting with glowing neural aesthetic.
```

### v002 — scene 4 "The Volume Gets Turned Up"
```
Abstract audio volume metaphor visualization. Left side: small, dim glowing brain shape (occipital region highlighted) with a tiny sound wave icon and large text "15 µV" below, labeled "Session 1" and "Occipital Region". Right side: same brain shape but brightly glowing with large pulsing sound waves radiating outward, large text "52–63 µV" below, labeled "Session 2–3". Connecting arrow in center with text "+300%" in gold. Top title text reading "Back-of-Brain Alpha Power". Deep navy background with warm luminous elements.
```

### v003 — scene 4 "Power Surge in the Back" (THE KEY VARIANT)
```
Side-by-side 3D brain visualization viewed from behind. Left brain (labeled "Session 1" at bottom, date "Oct" in corner): occipital and parietal regions dim and gray-blue, muted glow, small text "15 µV" floating near rear. Right brain (labeled "Session 2" at bottom, date "Nov" in corner): same regions blazing in warm amber-gold light, strong volumetric radiance from rear surface, large text "52 µV" floating near rear. Top banner reading "Posterior Alpha Power" in white bold. Center connecting arrow labeled "+247%" in bright gold. Premium neural illustration style, dark background.
```

### v004 — scene 4 "The Volume Gets Turned Up"
```
Abstract audio volume metaphor visualization. Left side: small, dim glowing brain shape (occipital region highlighted) with a tiny sound wave icon and large text "15 µV" below, labeled "Session 1" and "Occipital Region". Right side: same brain shape but brightly glowing with large pulsing sound waves radiating outward, large text "52–63 µV" below, labeled "Session 2–3". Connecting arrow in center with text "+300%" in gold. Top title text reading "Back-of-Brain Alpha Power". Deep navy background with warm luminous elements.
```

Notice: v002 and v004 visual prompts are essentially identical (both derived from v001 style). v003 is different — it uses "volumetric radiance", "premium neural illustration style", "3D brain visualization viewed from behind" — compositional/cinematographic language rather than electrode codes.

---

## CODE ARCHITECTURE (what you need to know)

- `core/image_gen.py` → `generate_image(prompt, output_path)` is the function that calls Replicate's Qwen API
- It auto-appends a STYLE_SUFFIX: `", patient-friendly medical education video, warm and reassuring aesthetic, premium healthcare feel, soft lighting, modern and approachable, never clinical or scary, 16:9 aspect ratio"`
- Model: `qwen/qwen-image-2512` (default)
- Cost: ~$0.02/image, ~7 seconds per image
- The `generate_scene_image(scene, project_dir)` wrapper saves to `project_dir/images/scene_XXX.png`

Director model was switched from `claude-sonnet-4-5` to `claude-sonnet-4-6` in `core/director.py` with thinking budget 10240 and max_tokens 12000.

---

## ENVIRONMENT

- Python: `/opt/homebrew/bin/python3.10` (MUST use 3.10)
- Working dir: `/Users/davidmontgomery/local-explainer-video`
- `.env` file has all API keys (REPLICATE_API_TOKEN, ANTHROPIC_API_KEY, etc.)
- Platform: macOS, M4 Pro

---

## WHAT NOT TO DO

1. **Do NOT evaluate narration text.** It's already fine. Word counts are in range. Analogies are present in the patched versions. Move on.
2. **Do NOT write custom generation scripts.** Use the existing `generate_image()` function from `core/image_gen.py` or the commands in this prompt.
3. **Do NOT simplify visual prompts.** The complexity is load-bearing for Qwen's MoE quality routing.
4. **Do NOT delegate to lightweight/haiku agents for reading plan.json files.** Read them yourself.
5. **Do NOT spend time on anything other than generating and comparing images.** The user has been waiting for this since the beginning of the conversation.
6. **Do NOT score things without looking at actual rendered output.** No theoretical evaluation. Generate, render, look, compare.

---

## SUMMARY OF DELIVERABLES

1. Generate 3–4 images per variant (9–12 total images) from the patched storyboard visual prompts
2. View every generated image
3. Compare each against the corresponding baseline image in `projects/01-01-2013-0/images/`
4. Score image quality per variant
5. Recommend a winner based on IMAGE OUTPUT QUALITY
6. Specifically answer: Does v003's compositional complexity swap maintain Qwen's MoE rendering quality compared to v002/v004's electrode-code-rich prompts?
