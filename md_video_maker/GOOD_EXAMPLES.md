# GOOD_EXAMPLES.md

Concrete bad/good pairs from this project's actual video history. Every bad example here is a real line that was written, reviewed, and rejected. Every good example is what replaced it after the rule in `VIDEO_BRAIN.md` was applied.

Read `VIDEO_BRAIN.md` first. This file is the evidence. The rules there earn their place by these failures.

---

## 1. Cold open — no references to a prior video

**Bad** (signal-console first draft):

> "Here's the same question we started with last time. When something strange happens in a live basketball game, who notices first — the official scorekeepers, or the people betting on that game in real time? We've looked at this before."

A first-time viewer has no "last time." The opener immediately puts them on the back foot, wondering whether they missed a prequel. They keep wondering all the way to scene seventeen, where the sponsoring organization is finally named for the first time.

**Good** (suspend-signal v2 scene 0):

> "When something happens in a live basketball game, who notices first? The official scorekeepers, or the crowd of people betting on the game in real time? We looked into that, and the answer says something useful about how fast information really moves."

Same topic, no prior-video reference. Sponsor named in scene one or two, not scene seventeen.

---

## 2. Define every term inline at first use

**Bad** (tribe_affect first draft, scene 9):

> "Decoders trained on real recorded fMRI from actual humans, evaluated against shuffle-permutation controls. The repo enforces this in code..."

The term "shuffle-permutation controls" arrives cold. The viewer who doesn't already know it is expected to file it away. The explanation does not come until scene fourteen, five scenes later.

**Good** (tribe_affect v1, scene 9):

> "Decoders trained on real recorded fMRI from actual humans, then evaluated with a shuffle test — where you randomly scramble the training labels, refit the model many times, and see what correlation you'd get by pure chance. If the real model beats the 95th percentile of that random baseline, the signal isn't noise."

The term is named and explained in the same sentence. The viewer never has to hold an undefined concept across scenes.

---

## 3. No "Not X, but Y" — including soft variants

These three lines were all written across different videos and all rejected. They share one shape: a rhetorical contrast deployed as a punchline.

**Bad** (suspend-signal v1):

> "The signal is the lurch, not the whisper."

**Bad** (tribe_affect codex draft):

> "It's not just a score, it's a trace."

**Bad** (signal-console second draft, written by Claude):

> "For people new to the math, that small affordance is what makes the console approachable instead of opaque."

All three are the same construction with different surface words. The fact that the construction kept reappearing in different drafts by different authors is exactly why the rule has to be enforced by quoting the shape, not just by naming the principle.

**Good** (tribe_affect v1, scene 16):

> "Plotted over time, you get a curve showing where the predicted emotional response rises and dips — which is a more useful thing than a single average, because it reflects the actual temporal structure of the stimulus."

Full sentence, no punchline contrast. The reader still gets the point — temporal resolution matters — but without the trailer-fragment rhythm.

---

## 4. No three-word drama fragments

**Bad** (basketball v1 codex draft):

> "Small r, real signal."

The fragment exists for rhythmic emphasis. It does not add information. The sentence immediately before it already said the same thing in full English: "fMRI-to-emotion decoding is one of the hardest problems in computational neuroscience."

**Good**: cut the fragment entirely. The preceding sentence carries the meaning.

---

## 5. No self-validating tells

**Bad** (signal-console first draft, scene 0):

> "There's a new tool for it now, called the Signal Console, and the math underneath it is genuinely worth opening up."

"Genuinely" is the narrator pre-grading their own content. If the math is worth opening up, the next few scenes will demonstrate it. The word does not help.

**Good** (rewrite):

> "This video is about a new tool called the Signal Console, built by the bet365 research team, and the statistical model running underneath it."

The same scope of content, no self-validating language. Other tells to scan for and delete: "actually worth," "worth pausing on," "the interesting part," "the deeper idea," "and that's a pretty interesting thing to have found."

---

## 6. No triumph triples at the close

**Bad** (signal-console first draft, scene 17):

> "But the signal is real, the math is checkable, the queue is the right size, and for a problem where a few extra seconds can genuinely matter, that's a meaningful place to stand."

A parallel three-part list, capped by a closing flourish, is a trailer beat by structure alone. The viewer can hear it coming as soon as the first comma lands. None of the three claims is actually demonstrated in the body of the video.

**Good** (rewrite):

> "What it cannot prove on its own is how that alert timing compares to bet365's own internal desk-clock — that comparison requires running the console live alongside the desk and measuring both timestamps. That side-by-side trial hasn't happened yet, and it's the next thing to do."

States the actual remaining open question and stops. No three-part summary, no closing flourish, no self-congratulation.

---

## 7. Voice instructions all-positive

**Bad** (basketball v1):

> "Adult male voice, natural American English. A senior sports-trading analyst briefing the desk: warm, direct, confident, dry, and human. Plainspoken and a little opinionated, like someone who actually built this. NOT a hype reel, NOT a corporate explainer. Measured but not slow."

The negation prompts ("NOT a hype reel") teach the model the cadence of the thing being negated. The same TTS voice with the same script reads as pretentious here and as warm under the all-positive instructions below.

**Good** (basketball v2):

> "Warm, friendly, conversational adult male voice. Sounds like a curious person explaining something genuinely interesting to a smart friend over coffee. Speaks in full, relaxed sentences with natural pauses. Easygoing and a little delighted by the subject. Clear and unhurried, letting the numbers land on their own."

Only positive attributes. No "not." Same voice model, completely different cadence in the rendered audio.

---

## 8. Image prompts — strong positive anchor, not a ritual negative list

**Bad** (basketball v1, scene 1):

> "A single disputed play at center with arrows spreading to labeled market tiles..."

Without a sport anchor, "disputed play" defaulted to American football. The slide rendered a football image with "PASS INTERFERENCE ON #25, PIT AT BAL" in a video that was about NBA basketball.

**Good** (basketball v2, scene 2):

> "Clean diagram on a dark hardwood-textured basketball background. Two simple basketball player silhouettes in jerseys side by side. Left silhouette labeled \"HAYES - actually grabbed it\" with a green check mark. Right silhouette labeled \"REAVES - what the feed wrote down\" with a red X. A glowing orange basketball arcing between them."

The positive anchor — basketball, hardwood, jerseys, orange ball — does all the work. No negative list of forbidden sports is required.

**Bad — ritualized negation** (every prompt in the first signal-console pass):

> "...premium editorial illustration. Strictly NBA basketball only. No American football, no football helmets, no field, no soccer, no stock tickers, no candlestick charts, no cryptocurrency, no coins, no commodities, no ticker symbols."

The trailing negation list got copy-pasted onto thirteen prompts including ones where the topic wasn't basketball, wasn't sports, and had nothing to do with finance. It is bloat, it pollutes the prompt, and it can summon the things it forbids.

**Good** — keep the positive anchor, drop the ritual. A targeted negative is acceptable only when an earlier render of the same prompt actually produced the wrong thing, and only for that specific failure.

---

## 9. Image prompts — richness

A flat prompt produces a flat slide. A rich prompt produces a slide worth looking at. The bar:

**Bad**:

> "Dark slide with a diagram showing the math."

**Good** (signal-console scene 9):

> "Cinematic still-life of an open leather-bound scientific notebook on a dark wooden desk, lit dramatically from one side. On the left page, the Kalman update equation is hand-written in elegant cream-colored ink in clean monospaced notation: \"x̂_t = x̂_t⁻ + K_t · y_t\". Below it, a smaller annotation: \"K_t ∈ [0, 1]  ←  Kalman gain\". On the right page, a small two-row table: \"high K_t → trust the observation\" and \"low K_t → trust the prediction\". At the bottom of both pages, two small notes: \"Q = process noise (how much drift?)\" and \"R = measurement noise (how noisy is p_t?)\". Large header above the notebook reading \"STEP 4: UPDATE THE ESTIMATE\". Cinematic museum-quality scientific-notebook illustration, painterly, warm tungsten key light, deep navy shadows."

Composition: open notebook, two pages, lit from one side. Palette: cream ink, dark wood, deep navy shadows. Material: leather binding, paper, warm tungsten light. On-screen text: every word quoted. Editorial finish: museum-quality, painterly, cinematic. The model has enough to render something specific.

---

## 10. Fix by cutting or inline rewriting — never by adding a defensive scene

**Bad** (suspend-signal v1 correction loop):

After a reviewer flagged that scene 12 was confusing, the previous correction added two new "context" scenes before it to explain the background. The result was a longer, slower, more apologetic video that still had a confusing scene 12.

**Good** (suspend-signal v2 correction loop):

After the cold-read reviewer flagged eight specific issues, every issue was fixed inline — by cutting words from the offending scene, by glossing the unexplained term in place, or by rewriting the smug sentence. Scene count went from 55 down to 18 in the same correction pass. The final video was shorter, clearer, and the reviewer signed off on it.

---

## 11. Cold-read review is the final gate

Every video script in this project that shipped successfully passed a cold-read review before render. The reviewer agent received only the narration text, in scene order, with no context about the project, audience, or purpose. The reviewer's flags became the inline fix list. The render was only kicked off after the fixes were saved to `plan.json`.

Every video script in this project that was rejected after rendering had skipped this step or had treated the reviewer's flags as suggestions rather than required fixes.

The cold-read is the only honest test of whether the script stands on its own.

---

## 12. A full plan.json that shipped — annotated

The following plan.json is the actual `rondo-annotations-explainer` plan that passed cold-read review and shipped. The complete JSON is included verbatim. After it, every scene is annotated with a one-paragraph note on what the narration does well and what it specifically avoids.

**FULL LENGTH GOOD EXAMPLE plan.json:**

```
{
  "meta": {
    "project_name": "rondo-annotations-explainer",
    "llm_provider": "claude_direct_authored",
    "image_model": "gpt-image-2",
    "tts_provider": "openai",
    "tts_model": "gpt-4o-mini-tts",
    "voice": "ash",
    "audio_speed": 1.0,
    "title": "Annotations in Rondo \u2014 A Deep Dive for the Video-to-Data Team",
    "summary": "A 10-12 minute walkthrough of Rondo's new Annotation workspace for the bet365 Video-to-Data team. Covers what annotation is, all 7 source connectors including the SCFC Stoke City lane, all task templates including custom schema building, the Create wizard, the task lifecycle, AI assistance and how to control it, export formats, the direct path to training, and the benchmark-failure correction loop. Assumes no prior annotation experience.",
    "voice_instructions": "Warm, clear, welcoming adult male voice. Explaining something new and genuinely useful to smart colleagues who haven't done this before. Friendly and practical \u2014 not a lecture, more like a thorough walkthrough from someone who built it. Clear with every term. No jargon without explanation. Natural pace with emphasis on key words like tool names and action steps.",
    "video_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/rondo-annotations-explainer.mp4",
    "rendered_utc": "2026-05-22T07:49:39.861979+00:00",
    "actual_duration_seconds": 742.44
  },
  "scenes": [
    {
      "id": 0,
      "title": "What This Video Is",
      "narration": "Rondo is bet365's internal platform for football video analysis \u2014 detection, tracking, calibration, tactical analysis, all of it. Most of you know the analysis side. This video is about something new that just landed: a full annotation workspace. If you've never annotated data before, that's completely fine \u2014 this video starts from scratch. If you've used other annotation tools, there's a lot here that'll look different from what you're used to.",
      "visual_prompt": "Dark charcoal near-black title card. Top left: a small hexagonal logo icon. Center: large bold clean sans-serif white text reading \"Annotations in Rondo\". Below: subtitle text \"A deep dive for the Video-to-Data team\". Background: a very faint football pitch line diagram in dark teal, barely visible. Bottom: smaller text \"bet365 \u00b7 internal tooling \u00b7 alpha\". Professional internal software aesthetic, teal and white accents on near-black. No consumer app clich\u00e9s, no neon, no gimmicks.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_000.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_000.wav"
    },
    {
      "id": 1,
      "title": "What Is Annotation",
      "narration": "Annotation just means labeling. You take a piece of footage, and someone draws boxes around players, marks specific positions, or tags events and time ranges. That labeled data is what machine learning models train on. The model sees thousands of labeled examples and learns what a player looks like, where the ball is, when a tackle happens. Without those labels, the model has nothing to learn from.",
      "visual_prompt": "Dark charcoal diagram on near-black background. Left side: a stylized still frame of a football match scene, dark stadium atmosphere, with three clean teal bounding boxes drawn around player silhouettes and one smaller box around a ball, each labeled with small white text: \"player\", \"goalkeeper\", \"ball\". A bold arrow pointing right. Right side: a simple database/stack icon labeled \"training data\". Header at top: \"ANNOTATION: TEACHING THE MODEL WHAT TO LOOK FOR\". Teal accent boxes, clean editorial style.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_001.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_001.wav"
    },
    {
      "id": 2,
      "title": "Why It Matters for This Team",
      "narration": "For this team specifically: every model running through Rondo \u2014 detection, tracking, pose estimation, event spotting \u2014 was trained on labeled data, and can be updated on new labeled data. That updating process is called fine-tuning. If a model gets something wrong on your footage, the fastest path to fixing it is to label a set of correct examples and fine-tune on them. Annotations are also how you add entirely new detection targets \u2014 things Rondo doesn't currently recognize at all.",
      "visual_prompt": "Dark charcoal three-step flow diagram on near-black background. Step 1 box labeled \"Model gets it wrong\" with a red-tinted detection box drawn incorrectly around a player. Bold arrow right. Step 2 box labeled \"Annotate the corrections\" with a teal correct box drawn precisely. Bold arrow right. Step 3 box labeled \"Retrain on new labels\" with a clean detection result. A circular return arrow from step 3 back to step 1 labeled \"continuous improvement\". Header: \"ANNOTATIONS CLOSE THE LOOP\". Teal arrows, clean editorial.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_002.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_002.wav"
    },
    {
      "id": 3,
      "title": "Quick Rondo Overview",
      "narration": "Quickly, for anyone new: Rondo has seven workspaces in the left rail. Sources is where you bring in footage. Analyze runs detection and tracking on it. Train updates models on new data. Evaluate runs benchmark tests \u2014 structured tests where you run a model against clips with known correct answers and measure how well it does. Review lets you replay saved runs. System shows the harness internals. And now Annotation \u2014 the new one \u2014 which is what this whole video covers.",
      "visual_prompt": "Dark charcoal diagram on near-black background mimicking Rondo's sidebar. A clean vertical navigation column with seven workspace labels stacked in order: \"Sources\", \"Annotation\" (highlighted with a teal background pill and a small \"NEW\" badge), \"Analyze\", \"Train\", \"Evaluate\", \"Review\", \"System\". Each label has a small geometric icon beside it. Header to the right: \"7 WORKSPACES IN RONDO\". Subtitle: \"The left rail \u2014 Annotation is the new one\". Professional dark UI aesthetic, teal highlight on Annotation only.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_003.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_003.wav"
    },
    {
      "id": 4,
      "title": "The Annotation Workspace",
      "narration": "The Annotation workspace has eight tabs in the sub-navigation. Inbox is your task queue \u2014 the work waiting for you. Create starts a new annotation project. Imports shows what's been staged. Projects shows your active annotation projects. Review is for approving completed work. Schemas is where you define the labeling format. Automation controls the AI assistant. And Exports is where finished annotation bundles live, ready to push downstream.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-inbox.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-inbox.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_004.wav"
    },
    {
      "id": 5,
      "title": "Sources \u2014 Where Footage Comes From",
      "narration": "When you start a new annotation project, the first choice is where the footage comes from. There are seven source connectors: your local files and folders, SCFC footage from Stoke City's match library, SoccerNet games, saved Rondo review clips, existing Rondo analysis runs, Roboflow datasets, and benchmark failures \u2014 clips that a model got wrong in evaluation. The SCFC connector adds specific provenance fields for season, team, competition, and source tags, and those travel with every label all the way to export.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-source-connectors.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-source-connectors.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_005.wav"
    },
    {
      "id": 6,
      "title": "Task Templates \u2014 Standard Geometry",
      "narration": "Once you've picked a source, you choose a task template \u2014 this tells Rondo what kind of labeling you want to do. Detection boxes: draw rectangles around players, the ball, and referees. Object tracking: link those boxes across frames so the same player has a consistent track ID throughout a clip. Segmentation and polygons: trace exact outlines. Keypoints and pose: mark specific body joints. Roles and jersey attributes: tag players by role or team. That covers the main geometry-based labeling tasks.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-task-templates.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-task-templates.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_006.wav"
    },
    {
      "id": 7,
      "title": "Time-Based Templates and Custom Schemas",
      "narration": "There are also time-based templates. Event spans let you mark when something starts and ends in a clip \u2014 a pressing phase, a corner, a goalkeeping action, whatever the task needs. Ball path and speed review is for trajectory analysis. And then there's the custom schema builder. If you need to annotate something the standard templates don't cover \u2014 gaze direction, head orientation, spatial relations between players \u2014 you define your own entity labels, relation types, and which field types to include. Save it, and it's a reusable schema for any future project.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-custom-schema.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-custom-schema.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_007.wav"
    },
    {
      "id": 8,
      "title": "The Create Wizard",
      "narration": "Walking through the Create wizard: it's a six-step flow. Source, task template, labeling engine, AI assistance settings, review policy, and export targets. Each step is visible as a numbered breadcrumb at the top. The engine choice matters \u2014 CVAT is an open-source annotation tool that opens in a separate browser tab and is the right choice for geometry work like boxes, tracking, and poses. The built-in editor handles event spans and custom tasks directly inside Rondo without opening anything else.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-soccernet-wizard.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-soccernet-wizard.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_008.wav"
    },
    {
      "id": 9,
      "title": "Video Auto-Slicing",
      "narration": "One thing that happens automatically when you import footage: if a video is longer than a few minutes, Rondo slices it into manageable windows before creating tasks. For detection and tracking work, that's forty-five-second windows with a three-second overlap between them. For event annotation, it's ninety-second windows. Each slice becomes its own task in the Inbox. You don't need to pre-cut footage before importing \u2014 just bring in the raw clip and Rondo handles the planning.",
      "visual_prompt": "Dark charcoal timeline diagram on near-black background. At the top: a long horizontal video bar labeled \"full match clip\" spanning the full width, with a small duration label. Below it: the same timeline is shown sliced into multiple equal-length teal segments, each labeled in sequence: \"Task 1: 0:00\u20130:45\", \"Task 2: 0:43\u20131:28\", \"Task 3: 1:26\u20132:11\". Small overlap zones between segments highlighted in a slightly lighter teal. A small caption reading \"45s windows \u00b7 3s overlap\". Header: \"LONG VIDEOS AUTO-SLICE INTO TASKS\". Clean dark editorial infographic.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_009.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_009.wav"
    },
    {
      "id": 10,
      "title": "The Inbox and Task Lifecycle",
      "narration": "The Inbox is the central task queue. Each entry shows the clip, the task type, the current status, and what actions are available right now. Tasks move through a clear progression: draft, queued, ready to label, in labeling, needs review, changes requested, accepted, and exported. The action buttons change based on what's valid at each stage \u2014 you only see options that make sense for where the task currently is. You can filter by status and by project.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-inbox.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-inbox.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_010.wav"
    },
    {
      "id": 11,
      "title": "How AI Assistance Works",
      "narration": "The AI assistance feature is designed to speed up labeling by generating a first draft for you to edit, rather than starting from blank. When you request a suggestion on a task, Rondo builds a preview: for image tasks it's a single high-resolution still; for video tasks it's a grid of four frames sampled across the clip window. That preview goes to the configured AI provider along with a task-aware prompt explaining what to look for. The AI returns a structured suggestion with proposed labels, positions, and a confidence rating \u2014 basically how sure it is about each detection.",
      "visual_prompt": "Dark charcoal flow diagram on near-black background. Four connected boxes with bold teal arrows flowing left to right. Box 1: \"Task clip\" with a small video frame icon. Box 2: \"Preview built\" showing a 2x2 frame grid for video, single frame for image, labeled \"still or frame grid\". Box 3: \"AI provider called\" with small icons for OpenAI, Claude, Gemini stacked. Box 4: \"Structured suggestion\" showing sample output text: \"player: 3 detected, ball: 1 detected, confidence: high\". Header: \"HOW RONDO BUILDS AN AI SUGGESTION\". Teal arrows, clean dark editorial.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_011.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_011.wav"
    },
    {
      "id": 12,
      "title": "The Suggestion Is Always a Draft",
      "narration": "The suggestion is always a draft \u2014 it's never automatically accepted or exported. For tasks using the built-in editor, the draft gets pre-filled into the labeling interface so you can immediately see the proposal and edit it. For tasks going to CVAT, the suggestion sits beside the task for reference while you label. Either way, a human reviews it and makes the final call. The system records exactly which AI provider generated which suggestion.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-ai-suggestion.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-ai-suggestion.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_012.wav"
    },
    {
      "id": 13,
      "title": "Automation Rules \u2014 Controlling the AI",
      "narration": "The Automation tab is where you control all of this. Each AI provider \u2014 OpenAI, Anthropic Claude, Google Gemini, OpenRouter, and local models \u2014 has its own rule. You can enable or disable it, restrict which task types it's allowed to work on, set the prompt pack version, and control whether it can attach editable drafts automatically. The recorded automation history at the bottom shows every job that ran, which tasks it touched, and the outcome. Nothing runs silently.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-automation.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-automation.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_013.wav"
    },
    {
      "id": 14,
      "title": "Review and Export Policy",
      "narration": "After a task is labeled and submitted, it goes into review. The reviewer sees the annotations alongside the source clip, and can accept it, request changes with a note, or pass it along. Export policy controls what makes it into the final export bundle. The default is human accepted only \u2014 a task has to be manually approved before it can export. You can also configure consensus accepted, which requires multiple reviewers to agree, machine draft allowed for lower-stakes tasks, or a mixed policy that applies different rules to different task types.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-export-policy.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-export-policy.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_014.wav"
    },
    {
      "id": 15,
      "title": "Exports and Formats",
      "narration": "The Exports tab shows completed bundles by project. Each bundle shows the format, the policy applied, and how many tasks qualified. You don't need to know all the formats right now, but here they are: YOLO and COCO are widely used training and benchmark formats for object detection; COCO Keypoints is the same idea but for pose estimation; MOT \u2014 Multiple Object Tracking \u2014 is the standard format for tracking evaluation; SoccerNet Events JSON is for event spotting models; Roboflow upload syncs to Roboflow's platform; and Rondo native JSON is the full-fidelity format that keeps everything. Every bundle includes a manifest of what's in it.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-exports.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-exports.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_015.wav"
    },
    {
      "id": 16,
      "title": "Open in Train \u2014 The Direct Pipeline",
      "narration": "From the Exports tab, every completed bundle has an Open in Train button. Click it, and the export bundle opens as a dataset source directly in Rondo's Training Studio \u2014 immediately available for a fine-tuning run. You don't reformat anything or manually wire up file paths. If you just finished annotating detection boxes on Stoke City footage, that export goes straight into a new training run. The Training Studio runs the training and shows you charts tracking how the model is improving as it goes.",
      "visual_prompt": null,
      "image_source_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-training-metrics.png",
      "image_path": "/Users/davidmontgomery/nba-predict/.playwright-mcp/rondo-sc-training-metrics.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_016.wav"
    },
    {
      "id": 17,
      "title": "The Other Direction \u2014 Benchmark Failures",
      "narration": "The pipeline also runs the other direction. In the Evaluate workspace, when a benchmark run surfaces clips where the model got something wrong, those failures can be sent straight to the Annotation workspace as a new project. The clip that caused the failure becomes the source, the task template gets pre-selected based on what the benchmark was testing, and the project lands in your Create queue ready to label. Label it, export it, retrain on it.",
      "visual_prompt": "Dark charcoal circular loop diagram on near-black background. Four boxes connected by bold teal arrows flowing clockwise. Top: \"Benchmark run in Evaluate\" with a small metrics icon. Right: \"Failure clips flagged\". Bottom: \"Annotation project auto-created from failure\". Left: \"Labeled correction exported and used in training\". A return arrow completing the loop back to \"Benchmark run\" labeled \"model improves\". Header: \"THE CORRECTION LOOP\". Clean dark editorial, teal arrows.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_017.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_017.wav"
    },
    {
      "id": 18,
      "title": "FiftyOne Curation",
      "narration": "For larger annotation projects there's an optional curation layer that connects to a separate tool called FiftyOne \u2014 an open-source dataset curation platform. You launch it from inside Rondo for any project, and it opens in a new browser tab with the project's labeled dataset loaded and saved queue views set up: tasks by priority, by status, by which ones have revisions. It's most useful when you have many tasks and want to check quality across the whole set before exporting \u2014 finding the hard cases, the inconsistencies, the outliers.",
      "visual_prompt": "Dark charcoal split diagram on near-black background. Left panel: Rondo's annotation workspace showing a project list. A bold arrow labeled \"Launch curation session\" pointing right. Right panel: a FiftyOne-style dataset grid view showing thumbnail images with teal annotation overlays, and a filter sidebar with queue labels: \"priority queue\", \"ready to label\", \"accepted\", \"with revisions\". Header: \"FIFTYONE: INSPECT QUALITY ACROSS THE WHOLE DATASET\". Dark editorial style, teal highlights.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_018.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_018.wav"
    },
    {
      "id": 19,
      "title": "What the System Handles for You",
      "narration": "Annotation is always going to take time and attention \u2014 that part doesn't go away. But this setup removes a lot of the mechanical friction. Long videos are sliced automatically. AI suggestions pre-fill the interface so you're editing rather than drawing from blank. Match provenance \u2014 which game, which team, which season \u2014 is tracked without any extra steps. Exports go to training in one click. And the review queue keeps work organized without a separate spreadsheet.",
      "visual_prompt": "Dark charcoal summary slide on near-black background. Five bullet points with small teal icon before each: a scissors icon \"Long videos: auto-sliced into tasks\"; a robot+pencil icon \"AI suggestion: pre-fills the interface, you edit\"; a tag icon \"Match provenance: tracked automatically\"; an arrow icon \"Export to training: one click\"; a checkmark icon \"Review queue: built in\". Header: \"WHAT THE SYSTEM HANDLES FOR YOU\". Footer note in smaller text: \"The labeling judgment is still yours\". Clean dark list-card aesthetic.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_019.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_019.wav"
    },
    {
      "id": 20,
      "title": "Try It \u2014 Where to Start",
      "narration": "The Annotation workspace is in alpha \u2014 real, usable, and still being actively developed. The best way to start is to pick a short local clip and choose the Detection boxes template. That will route you through CVAT, which will open in a new browser tab \u2014 draw your boxes there, save, and the task syncs back to Rondo automatically. Run it through to export and see the full workflow. If anything is confusing or broken, that feedback is exactly what shapes what gets built next.",
      "visual_prompt": "Dark charcoal closing card on near-black background. Center: small Rondo hexagonal logo. Below it: large clean white text \"Annotation workspace\". Subtitle: \"Alpha \u00b7 live \u00b7 actively developed\". Then a clean three-step get-started block: \"1. Pick a short local clip\", \"2. Choose Detection boxes\", \"3. Run the full workflow\". Footer: \"Your feedback shapes what gets built next\". Bottom right: \"bet365 \u00b7 Video-to-Data team\". Warm, clean, professional. Teal accent line at top.",
      "image_source_path": null,
      "image_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/images/scene_020.png",
      "audio_path": "/Users/davidmontgomery/markdown-video-experiment/projects/rondo-annotations-explainer/audio/scene_020.wav"
    }
  ]
}

```

### Annotated walkthrough — why each scene in the good plan works

**Meta.** Names the audience ("bet365 Video-to-Data team") and the runtime target ("10-12 minute walkthrough"). Voice instructions are all-positive ("Warm, clear, welcoming"). No negation, no "NOT a corporate explainer." Sponsor and scope land before the first scene.

**Scene 0 — "What This Video Is."** Opens with a plain statement of what Rondo is, what the audience already knows, and what the video is about. Names the sponsor and the topic in the first sentence. Explicitly accommodates two audience subsets ("never annotated data before" / "used other annotation tools") in two short clauses. No "as we saw last time," no "the earlier work."

**Scene 1 — "What Is Annotation."** Defines the central term in the first sentence: "Annotation just means labeling." Then describes the activity concretely (draws boxes, marks positions, tags events). Closes with a one-line causal explanation ("Without those labels, the model has nothing to learn from"). No self-validating tells.

**Scene 2 — "Why It Matters for This Team."** Glosses "fine-tuning" the moment it's introduced ("That updating process is called fine-tuning"). Connects annotation directly to the team's own pipeline ("every model running through Rondo"). Avoids "matters" as a self-validating tell by tying it to concrete actions ("the fastest path to fixing it is to label a set of correct examples and fine-tune on them").

**Scene 3 — "Quick Rondo Overview."** A seven-item list, but each item is one verb phrase ("Sources is where you bring in footage. Analyze runs detection and tracking on it."). The one item that needs a gloss — "Evaluate runs benchmark tests" — gets the gloss inline ("structured tests where you run a model against clips with known correct answers and measure how well it does"). This is the rule-2 inline-definition pattern at full strength.

**Scene 4 — "The Annotation Workspace."** Eight tabs named in order with a single descriptive clause each. No clever framing, no "Inbox isn't just a queue — it's your command center." Just labels and what they do.

**Scene 5 — "Sources — Where Footage Comes From."** Seven source connectors listed. Then one sentence on the one connector that has unique behavior (SCFC) explaining what's different about it. The list-plus-one-detail pattern keeps the scene useful without being a dump.

**Scene 6 — "Task Templates — Standard Geometry."** Five templates, each defined by what the labeler does ("Detection boxes: draw rectangles around players, the ball, and referees"). Concrete, not abstract. The scene closes by naming what it covered ("the main geometry-based labeling tasks") and forecasts the next scene ("There are also time-based templates") instead of using a stinger.

**Scene 7 — "Time-Based Templates and Custom Schemas."** Continues the previous scene's structure, then introduces the more flexible piece (custom schemas) with a concrete use case ("gaze direction, head orientation, spatial relations between players"). The reader sees both standard and custom in the same beat.

**Scene 8 — "The Create Wizard."** The hardest piece of jargon in the workflow (CVAT) is glossed at first use: "CVAT is an open-source annotation tool that opens in a separate browser tab." The viewer leaves this scene knowing what CVAT is and when they'll use it.

**Scene 9 — "Video Auto-Slicing."** A behavior the user might be surprised by, described as a sequence of concrete numbers (forty-five-second windows, three-second overlap, ninety-second windows for events). No "magical," no "automatic" used as a marketing flourish — just "happens automatically."

**Scene 10 — "The Inbox and Task Lifecycle."** Eight states listed in order. No editorializing on the state machine. The one design observation made — "The action buttons change based on what's valid at each stage" — is the kind of thing a first-time user actually needs to know.

**Scene 11 — "How AI Assistance Works."** The math is replaced with a flow ("you request a suggestion... Rondo builds a preview... that preview goes to the configured AI provider... the AI returns a structured suggestion"). The word "confidence" gets a tiny inline gloss ("basically how sure it is about each detection") so the technical viewer doesn't trip on it.

**Scene 12 — "The Suggestion Is Always a Draft."** The single design constraint that matters most for trust ("a human reviews it and makes the final call") is stated plainly without a flourish. No "the human is the boss" or "humans are still in the loop" cliche.

**Scene 13 — "Automation Rules — Controlling the AI."** Five providers listed, then the four rule axes, then one sentence on what gets logged. Closes on "Nothing runs silently" — short, but it's a description of a real property, not a slogan.

**Scene 14 — "Review and Export Policy."** Three review outcomes, then four export policies. Each policy gets one-clause behavior. The default is named ("The default is human accepted only") so the reader knows where they're starting from.

**Scene 15 — "Exports and Formats."** A list of seven formats — the kind of thing that could easily become a jargon dump. The scene opens with explicit permission to skip ("You don't need to know all the formats right now, but here they are") so the technical reader doesn't feel they're being quizzed. Then each format gets one phrase. MOT, the only acronym not commonly known, is spelled out inline.

**Scene 16 — "Open in Train — The Direct Pipeline."** The single most operationally important workflow connection in the system is described in concrete actions: "Click it, and the export bundle opens as a dataset source directly in Rondo's Training Studio." No metaphor, no "the seamless pipeline" or "frictionless workflow."

**Scene 17 — "The Other Direction — Benchmark Failures."** A bidirectional pipeline relationship described as a simple chain: "the failure becomes the source, the task template gets pre-selected, the project lands in your Create queue ready to label. Label it, export it, retrain on it." The "label it, export it, retrain on it" is three imperatives but they're literal next steps, not a triumph triple.

**Scene 18 — "FiftyOne Curation."** FiftyOne is glossed at first mention ("a separate tool called FiftyOne — an open-source dataset curation platform"). When it's most useful is described concretely ("when you have many tasks and want to check quality across the whole set before exporting").

**Scene 19 — "What the System Handles for You."** Summary scene done well: five distinct items, each a concrete behavior. Opens by acknowledging what doesn't change ("Annotation is always going to take time and attention — that part doesn't go away") so the summary doesn't come across as overselling. No triumph triple structure — just a flat list with consistent grammar.

**Scene 20 — "Try It — Where to Start."** The close. States alpha status honestly ("real, usable, and still being actively developed"). Gives a specific three-step starting recipe ("pick a short local clip... choose the Detection boxes template... will route you through CVAT"). Forecasts what the user will see next so they're not surprised by CVAT opening in a new tab. Ends on an honest "if anything is confusing or broken, that feedback is exactly what shapes what gets built next" — no closing flourish, no "and that's a pretty interesting thing to have found."

---

## 13. A bad plan.json — example of what NOT to write

The plan.json below is a real one from an earlier internal video. The narrations are exactly as written. The visual prompts in the original were actually decent; they have been **deliberately replaced with lazy, generic ones** so this example is uniformly bad and matches the failure mode of "didn't bother on the visuals either."

**Read the narrations first. Then read the annotations.** Every problem flagged here has been written about somewhere in this file or in `VIDEO_BRAIN.md`; this is the same set of rules being violated in one continuous piece of work.

**BAD plan.json (do not imitate):**

```
{
  "meta": {
    "project_name": "rondo-internal-demo",
    "llm_provider": "codex",
    "image_model": "gpt-image-2",
    "tts_provider": "openai",
    "tts_model": "gpt-4o-mini-tts",
    "voice": "onyx",
    "audio_speed": 1.0,
    "title": null,
    "summary": null,
    "voice_instructions": null
  },
  "scenes": [
    {
      "id": 0,
      "title": "Rondo, Not A Sales Demo",
      "narration": "Rondo is named after the football exercise: small space, fast decisions, constant pressure. This internal demo is the same idea for AI engineering work. It is a source-first workbench for testing football hypotheses, running analysis, training detectors, and deciding what is real with evidence. I am going to be candid about what works today, what is still rough, and where the next engineering leverage is. The goal is shared judgment, not theater.",
      "visual_prompt": "Dark slide with the words 'Rondo' on it.",
      "image_source_path": null
    },
    {
      "id": 1,
      "title": "The Demo Map",
      "narration": "The shape of the product is simple: start with Sources, move into Analyze, review persisted outputs, improve detectors in Train, run research loops, score candidates in Evaluate, and keep System as the readiness surface. That ordering matters. Rondo is not pose-first, and it is not a taxonomy browser. It starts from the source and keeps every downstream claim attached to artifacts, so later decisions can be replayed instead of remembered.",
      "visual_prompt": "A diagram showing the workflow.",
      "image_source_path": null
    },
    {
      "id": 2,
      "title": "Source-First Intake",
      "narration": "Here is the first proof point. The operator starts by bringing in a source: a local clip, an upload, a URL, a benchmark asset, a simulation capture, or eventually an audio source. The current shell classifies those sources deterministically, so the interface can make a sane first guess before a model is involved. That is a small product choice, but it keeps the whole workflow grounded.",
      "visual_prompt": "Dark slide with text about sources.",
      "image_source_path": null
    },
    {
      "id": 3,
      "title": "Analyze Setup",
      "narration": "Once the source is loaded, Analyze exposes the concrete knobs: the soccana detector, player tracker, ball tracking, thresholds, and run controls. This is intentionally not magic. It gives researchers enough control to test a weird signal quickly, while still logging what was actually used. For bet three six five in-play research, that means ideas can move from hallway speculation to a replayable run.",
      "visual_prompt": "Screenshot of an analysis screen.",
      "image_source_path": null
    },
    {
      "id": 4,
      "title": "What Runs Under Analyze",
      "narration": "Under that button is a real pipeline. Today it runs soccana for players, referees, and the ball; hybrid appearance-aware tracking with tracklet stitching; ByteTrack for the ball; jersey-colour team clustering; field registration from soccana keypoints; and a calibration refresh every ten frames. The important part is not that every component is perfect. It is that each component leaves evidence behind, so failures can become specific engineering work instead of vague model blame.",
      "visual_prompt": "Dark slide with a pipeline diagram.",
      "image_source_path": null
    },
    {
      "id": 5,
      "title": "Overlay As Primary Artifact",
      "narration": "The Review page is where the analysis becomes inspectable. The overlay video is the hero artifact because it lets us see detections, tracks, and failure modes on the original footage. Alongside that, the pitch projection gives a second view of ball and player paths. This is the tone Rondo should keep: show the evidence first, then discuss whether the signal deserves belief.",
      "visual_prompt": "A screen with video on it.",
      "image_source_path": null
    },
    {
      "id": 6,
      "title": "The Artifact Ledger",
      "narration": "A run is not just a screen state. The backend persists summary dot JSON, detections CSV, track summaries, projections, calibration debug when present, entropy time series, goal events when labels exist, diagnostics, and a bundled zip. That matters for engineering culture. If someone claims a new detector, recipe, or volatility metric is better, Rondo should make it possible to point at the exact run and artifacts.",
      "visual_prompt": "Dark slide with a list of file names.",
      "image_source_path": null
    },
    {
      "id": 7,
      "title": "Diagnostics That Name The Problem",
      "narration": "The diagnostics layer is deliberately implementation-aware. It builds context from summary metrics, heuristics, logs, and code slices, then uses the configured AI provider if available, with a heuristic fallback when it is not. The point is not to pretend everything is green. The useful diagnostic says, here is the likely function, the condition that failed, the fallback that fired, and the code change worth trying next.",
      "visual_prompt": "Dark slide with technical text.",
      "image_source_path": null
    },
    {
      "id": 8,
      "title": "Caveat: Hypothesis Harness, Not Edge Claim",
      "narration": "This is where the honesty matters. The original spatial entropy idea was useful because it became cheap to test: do clustered team shapes in certain zones tell us anything about volatility thirty seconds out? Maybe not. Rondo should not claim that signal is proven. The win is the harness: encode the hypothesis, run the analysis, persist the outputs, and let Evaluate decide whether it survives. That is how early experiments stay useful without turning into folklore.",
      "visual_prompt": "Dark slide with the word 'Caveat'.",
      "image_source_path": null
    },
    {
      "id": 9,
      "title": "Train: Detector Fine-Tuning V One",
      "narration": "Train is the first model-improvement surface. Today it is detector fine-tuning V one, not a universal training factory. It scans local YOLO datasets, shows the class map, writes a run-local dataset runtime YAML, and launches isolated worker subprocesses. That boundary is important: pretrained soccana stays distinct from activated custom detectors, and product-facing detector lists stay curated.",
      "visual_prompt": "Dark slide with text about training.",
      "image_source_path": null
    },
    {
      "id": 10,
      "title": "Provenance Before Activation",
      "narration": "The training job surface is where Rondo starts to feel like a harness instead of a notebook. It persists logs, progress, summary, provenance, metrics curves, checkpoints, and review notes. A completed checkpoint can enter the local registry, but the UI should still warn when validation signal is weak. That is the correct product instinct: activation is allowed, but provenance and skepticism come first.",
      "visual_prompt": "Dark slide with a list of items.",
      "image_source_path": null
    },
    {
      "id": 11,
      "title": "Autoresearch Controller",
      "narration": "The research-loop tab is the most important direction of travel. It borrows the useful part of Andrej Karpathy's autoresearch pattern: program dot MD defines the agent policy, runs are bounded, results are scored, and only better changes should survive. Rondo is not optimizing nanochat. The adaptation is for football models and recipes: run overnight on local, GPU, or DGX resources, record provenance, and feed better candidates back into Analyze and Evaluate. Rafal is already close to this lane and can help finish it, especially around the policy, scoring, and review boundaries.",
      "visual_prompt": "Dark slide with text about a research loop.",
      "image_source_path": null
    },
    {
      "id": 12,
      "title": "The Loop Contract",
      "narration": "The loop contract is intentionally boring and strict: edit a bounded experiment, run it inside the allowed budget, score it against a metric, keep the change only if it improves the score, and write down what happened. That is how Rondo can make overnight work useful instead of theatrical. The agent is allowed to explore, but the harness decides what counts.",
      "visual_prompt": "Dark slide with a list of steps.",
      "image_source_path": null
    },
    {
      "id": 13,
      "title": "Evaluate As Fitness Layer",
      "narration": "Evaluate is the fitness layer. The Benchmark Lab runs suite-by-recipe comparisons, materializes benchmark state, logs task progress, preserves blocked or unsupported cells with reasons, and links candidates back to normal analysis runs. This is the other half of autoresearch: if Train and the controller generate possibilities, Evaluate says which ones deserve another minute of attention.",
      "visual_prompt": "Dark slide with text about evaluation.",
      "image_source_path": null
    },
    {
      "id": 14,
      "title": "Compare Matrix",
      "narration": "The compare view is where engineering arguments get shorter. Recipes, statuses, primary metrics, detailed panels, and charts sit together, including the cases that are blocked rather than silently missing. That is valuable for bet three six five because in-play research needs disciplined rejection as much as exciting positives. A failed recipe with a reason is still useful information.",
      "visual_prompt": "Dark slide with a comparison table.",
      "image_source_path": null
    },
    {
      "id": 15,
      "title": "Metric Drilldown",
      "narration": "The chart drilldown is a short beat, but it is philosophically important. Rondo should make research cheap, but not cheap in the sense of casual. It should let us test variables, recipes, detectors, gaze features, or volatility proxies, then ask: did this improve a benchmark, did it generalize, and can we replay the evidence? That is the bridge from in-play idea to engineering claim.",
      "visual_prompt": "Dark slide with a chart.",
      "image_source_path": null
    },
    {
      "id": 16,
      "title": "System Is Readiness, Not The Front Door",
      "narration": "System is where the future lanes are visible without pretending they are finished. SoccerMaster is a serious vision-foundation-model direction, but local readiness still has to prove detections, calibration, overlays, and summaries end to end. Acoustic Momentum is a first-class sibling lane for crowd-derived pressure, and gait or player-motion work is relevant, but those are not mature in this footage. The value of this page is the boundary: source taxonomy, adapter blockers, rights constraints, and replayable trace skeletons. It tells us what can be tested now and what needs adapter work first.",
      "visual_prompt": "Dark slide with text about system status.",
      "image_source_path": null
    },
    {
      "id": 17,
      "title": "Where This Helps The Team",
      "narration": "For Platform Innovation, the near-term use is a shared evidence surface for in-play signal research: detectors, variables, benchmark recipes, and overnight loops all judged in one place. For Stoke City, the same harness can support scouting, coaching, training, and tactical analysis, as long as we keep evidence attached to footage and benchmarks. Matt Dalley's detection and player-gaze work has immediate value here, especially if gaze becomes another source feature rather than a standalone demo.",
      "visual_prompt": "Dark slide with two bullet points.",
      "image_source_path": null
    },
    {
      "id": 18,
      "title": "The Ask",
      "narration": "The ask for the AI engineering team is straightforward. Bring your own model weights, variables, gaze features, benchmark recipes, and overnight research policies. Use Rondo to make those ideas cheap to run and hard to overclaim. If Rafal helps close the research loop, and Matt's detection and gaze work feeds the source and training surfaces, this becomes a useful shared workbench: not a finished oracle, but a disciplined way to find out what is real.",
      "visual_prompt": "Dark slide with the words 'The Ask'.",
      "image_source_path": null
    },
    {
      "id": 19,
      "title": "How To Use This Tomorrow",
      "narration": "If the team uses this tomorrow, the path is practical. Start with a clip or source, run the default analysis, inspect the overlay, look at the diagnostics, then decide whether the problem is detector quality, tracking, calibration, team assignment, or the hypothesis itself. That keeps discussion attached to evidence instead of intuition.",
      "visual_prompt": "Dark slide with a numbered list.",
      "image_source_path": null
    },
    {
      "id": 20,
      "title": "What Is Still Early",
      "narration": "The honest gaps are also clear. SoccerMaster is not proven end to end in the local Analyze path yet. Acoustic and gait lanes are product-significant, but they are not mature in this footage. Remote URLs and simulation import are classified but not fully wired. The value is that those gaps are named in the same place as the working paths.",
      "visual_prompt": "Dark slide with text about gaps.",
      "image_source_path": null
    },
    {
      "id": 21,
      "title": "The Shared Standard",
      "narration": "The shared standard I would like us to use is simple: no idea gets promoted because it sounds clever, and no model gets promoted because one overlay looked good. Rondo should make it easy to run the idea, score it, replay it, and keep the evidence. That is the bar that makes this useful for AI engineering rather than just another demo app.",
      "visual_prompt": "Dark slide with the words 'Shared Standard'.",
      "image_source_path": null
    }
  ]
}
```

### Annotated walkthrough — why every part of this fails

**Meta.** `title` is `null`. `summary` is `null`. `voice_instructions` is `null`. The video has no declared subject, no declared length, and no declared voice direction. The reader cannot answer "what is this video about and who is it for" from the meta at all. Compare to the good plan, which names the team, the runtime, and the voice character in the first three fields.

**Scene 0 — "Rondo, Not A Sales Demo."** Title is the banned "X, Not Y" construction (rule 3). The opener — "Rondo is named after the football exercise: small space, fast decisions, constant pressure" — is a cute origin story, not an explanation. A first-time viewer does not yet know what Rondo *is* or *does*, but they're already being told it's "named after" something. Then comes "source-first workbench" — an invented term, never defined. "I am going to be candid" pre-grades the narrator's own honesty. The closing fragment "shared judgment, not theater" is another rule-3 violation (and "theater" has no antecedent — what theater?). The viewer ends scene 0 not knowing what Rondo is, who it's for, or what's about to be demonstrated.

**Scene 1 — "The Demo Map."** Lists seven workspace names ("Sources, Analyze, persisted outputs, Train, research loops, Evaluate, System") with no glosses. Then the sentence "Rondo is not pose-first, and it is not a taxonomy browser" — two banned "not X" constructions back-to-back, *and* the things being negated ("pose-first," "taxonomy browser") are never explained. The viewer is told what Rondo *isn't* twice, in invented vocabulary they don't have. Closes with "later decisions can be replayed instead of remembered" — soft "not X but Y" rhythm.

**Scene 2 — "Source-First Intake."** Title doubles down on the undefined "source-first" jargon. Opens with "Here is the first proof point" — sales-deck phrasing. Six source types listed without explaining what any of them are. "Shell," "classifies deterministically," "before a model is involved" — implementation jargon in a row. Closes with the narrator commenting on their own design choice as "small."

**Scene 3 — "Analyze Setup."** Drops `soccana` and `bet three six five in-play research` cold with no gloss. "This is intentionally not magic" — rule-3 soft contrast, and now the viewer wonders who claimed it was magic. "Hallway speculation to a replayable run" is the kind of writerly line that feels good to write and lands poorly on a first-time viewer who has neither hallway nor speculation context.

**Scene 4 — "What Runs Under Analyze."** Six technical components ("hybrid appearance-aware tracking with tracklet stitching," "ByteTrack," "jersey-colour team clustering," "field registration from soccana keypoints," "calibration refresh every ten frames") stacked in one sentence with no gloss on any of them. Then "The important part is not that every component is perfect. It is that each component leaves evidence behind" — textbook rule-3 violation. Closes with another soft contrast ("specific engineering work instead of vague model blame").

**Scene 5 — "Overlay As Primary Artifact."** "Hero artifact" is marketing-deck speak. "This is the tone Rondo should keep" — the narrator is now giving design direction to his own team out loud, which is fine in a planning doc and wrong in a video for an audience that doesn't work on the project. "Show the evidence first, then discuss whether the signal deserves belief" — preachy.

**Scene 6 — "The Artifact Ledger."** Invents the term "Artifact Ledger" and never explains it. Lists nine file types in a row ("summary dot JSON, detections CSV, track summaries, projections, calibration debug when present, entropy time series, goal events when labels exist, diagnostics, and a bundled zip") — none of which mean anything to a non-implementer. "That matters for engineering culture" is the narrator preaching.

**Scene 7 — "Diagnostics That Name The Problem."** "Implementation-aware" is a piece of jargon defined nowhere. "The point is not to pretend everything is green" — rule 3 again. The list "here is the likely function, the condition that failed, the fallback that fired, and the code change worth trying next" assumes the audience is doing code review.

**Scene 8 — "Caveat: Hypothesis Harness, Not Edge Claim."** Title is the third "X, Not Y" construction in the deck. Opens with "This is where the honesty matters" — self-validating. Names "spatial entropy" without explaining it. "Maybe not." is a banned drama fragment (rule 4). "Without turning into folklore" — writerly flourish, not information.

**Scene 9 — "Train: Detector Fine-Tuning V One."** "First model-improvement surface" uses "surface" as a noun in a way that means nothing to anyone outside the project. "Not a universal training factory" — fourth "not X" structure of the deck. "It scans local YOLO datasets, shows the class map, writes a run-local dataset runtime YAML, and launches isolated worker subprocesses" — a sentence written for a code reviewer, deployed for an audience that may not be one. "Pretrained soccana stays distinct from activated custom detectors" assumes a vocabulary the viewer has not been given.

**Scene 10 — "Provenance Before Activation."** Title is project-internal vocabulary. "Harness instead of a notebook" — both ends of the contrast are insider terms. "The correct product instinct" — narrator grading his own design.

**Scene 11 — "Autoresearch Controller."** Name-drops "Andrej Karpathy's autoresearch pattern" without explaining it. Reads a filename aloud ("program dot MD"). Name-drops "nanochat" — a different tool the viewer has no reason to know. Lists "local, GPU, or DGX resources" — assumes NVIDIA vocabulary. Names a specific employee ("Rafal") who the viewer has no context for. The whole scene reads as a Slack message to the project team mistakenly recorded as a video.

**Scene 12 — "The Loop Contract."** Title is invented vocabulary. "Intentionally boring and strict" — narrator describing his own choices. "Useful instead of theatrical" — soft "X not Y" and "theater" reappears with still no explanation. "The harness decides what counts" — preachy.

**Scene 13 — "Evaluate As Fitness Layer."** "Fitness layer" is invented vocabulary. "Suite-by-recipe comparisons," "materializes benchmark state," "preserves blocked or unsupported cells with reasons" — three separate pieces of project-internal jargon in one sentence.

**Scene 14 — "Compare Matrix."** "Engineering arguments get shorter" — narrator-as-aphorist. Names another piece of internal vocabulary ("blocked rather than silently missing") and assumes it carries weight without setup.

**Scene 15 — "Metric Drilldown."** "Philosophically important" — narrator overstating his own beat. "Variables, recipes, detectors, gaze features, or volatility proxies" — five pieces of vocabulary in a row, none defined. "Gaze features" is named here for the first time and never explained anywhere in the video.

**Scene 16 — "System Is Readiness, Not The Front Door."** Fourth "X, Not Y" title. "SoccerMaster," "Acoustic Momentum," "gait or player-motion work" — three internal project codenames in two sentences, none of which the viewer has heard before. "Sibling lane" is invented vocabulary. The whole scene is a status update for the project team, dressed as an external explanation.

**Scene 17 — "Where This Helps The Team."** Names "Platform Innovation," "Stoke City," and "Matt Dalley" in quick succession — three internal entities, two of them named individuals, all without setup. The viewer still doesn't know what bet365 is to this video, let alone what Platform Innovation is. "Source feature rather than a standalone demo" — soft contrast again.

**Scene 18 — "The Ask."** Sales-deck title. "Bring your own model weights, variables, gaze features, benchmark recipes, and overnight research policies" — imperative voice dumping five pieces of jargon at the audience. Names "Rafal" and "Matt" again as if the viewer has been introduced. Closes with "not a finished oracle, but a disciplined way to find out what is real" — banned "not X, but Y" construction at scene-close.

**Scene 19 — "How To Use This Tomorrow."** "If the team uses this tomorrow" — assumes "the team" as a known referent. The five-step instruction list ("Start with a clip or source, run the default analysis, inspect the overlay, look at the diagnostics, then decide...") works mechanically, but the diagnostic categories ("detector quality, tracking, calibration, team assignment, or the hypothesis itself") are all internal vocabulary.

**Scene 20 — "What Is Still Early."** Repeats "SoccerMaster" and "Acoustic and gait lanes" without explaining either. "Classified but not fully wired" — implementation jargon. Closes with the narrator congratulating himself for being honest ("The value is that those gaps are named in the same place as the working paths").

**Scene 21 — "The Shared Standard."** The close. Opens with "the shared standard I would like us to use" — narrator preaching to his own team, not addressing the viewer. "No idea gets promoted because it sounds clever, and no model gets promoted because one overlay looked good" — parallel rhetorical construction. "Rondo should make it easy to run the idea, score it, replay it, and keep the evidence" — banned triumph triple-plus-one at the close (rule 6). "That is the bar that makes this useful for AI engineering rather than just another demo app" — closing flourish with embedded "X rather than Y" construction and self-congratulation.

**Visual prompts.** Every prompt in this deck is one of: "Dark slide with X," "Dark slide with text," "A diagram," "A screen." None of them name composition, palette, lighting, or on-screen text. The image model has nothing to render except "make a dark slide," and that is exactly what comes back. Compare to the cathode-quality prompt shown in section 9 above — composition, palette, materials, lighting, every on-screen word explicitly quoted, editorial finish named. The lazy prompts here would have produced a uniformly forgettable deck even if the narration had been good.

**Pattern across the whole deck.** No clean cold open, no audience naming, no inline glosses for any term, four "X, Not Y" titles, nine "not X, but Y" narration constructions, one banned drama fragment, multiple invented terms used without definition, multiple internal employee names without setup, a closing triumph triple, and a narrator who keeps stepping out of "explain" mode into "tell my team what I think" mode. This is a planning memo set to TTS, not a video.
