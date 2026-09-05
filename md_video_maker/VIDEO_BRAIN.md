# VIDEO_BRAIN.md

The discipline a narrated still-video factory needs to not slide back into LLM voice. Each rule here comes from a real failure across the project's video history. Read this in full before authoring any new narration or any new visual prompt.

If you find yourself wanting to add a rule, first check whether an existing rule covers it. This document is short on purpose — sprawl makes it ignored.

---

## 1. The audience starts cold

Never open with "as we saw last time," "the earlier work," or any other reference to a prequel the viewer may not have seen. Every video must stand alone.

Name the sponsor or the system early — within the first thirty seconds, ideally in the first or second scene. A viewer who reaches scene seventeen before learning who made the tool or who the work is for has been carrying that uncertainty the whole way.

**Why:** Multiple videos have leaked first-time viewers in the opening seconds by referencing a "previous" video those viewers had not seen, or by introducing the sponsoring organization only at the close. Both are recoverable but both bleed trust.

**How to apply:** First scene establishes the topic and the asker. By the end of scene two the viewer knows what the video is about and who is doing the work. Continuity to a prior video is optional flavor, never a load-bearing setup.

---

## 2. Define every term inline, at first use

When a term is introduced — a tool name, an acronym, a statistical concept, a piece of internal jargon — gloss it in the same sentence or the next. Not five scenes later. Not "we'll come back to that."

The gloss does not have to be long. Half a clause is usually enough: "evaluated with a shuffle test — where you randomly scramble the labels and see what correlation you'd get by chance." That is sufficient. What is not sufficient is naming "shuffle-permutation controls" in scene nine and explaining it in scene fourteen.

**Why:** A viewer who hits an unexplained term has two options: keep listening while uncertain, or rewind. Both cost attention. Across the project's videos, the most consistent reviewer complaint has been "this term landed cold."

**How to apply:** Read each scene as if you've never heard any of the terms it names. If any of them require prior knowledge that the narration has not yet provided, add the gloss inline. Common offenders in technical scripts include: cortex, parcellation, BOLD, Pearson r, shuffle controls, CVAT, FiftyOne, MOT, COCO, YOLO, fine-tuning, Kalman gain, innovation, process noise.

---

## 3. No "Not X, but Y" — including the soft variants

The construction "not X, it's Y" and its near relatives are banned. This includes:

- "Not just a number, it's a story."
- "Not the whisper, the lurch."
- "Not just a score, it's a trace."
- "Approachable instead of opaque."
- "The pattern, not the line."
- "Less like X, more like Y."

All of these have the same shape: a rhetorical contrast deployed as a punchline. The construction always reads as performative regardless of how well it fits the content. If a scene's idea is genuinely about a contrast, state both sides plainly in full sentences and let the contrast emerge from the meaning.

**Why:** This is the single most persistent stylistic regression across the project. It has appeared in codex-authored drafts, in Claude-authored drafts, and in drafts that explicitly banned it in the brief. The only defense that works is calling it out by name and quoting it back when it appears.

**How to apply:** After drafting any scene, scan for the words "not" and "instead" and any short clause beginning with a noun-pause-noun rhythm. Rewrite anything matching the shape above as a normal explanatory sentence.

---

## 4. No three-word drama fragments

Sentences like "Small r, real signal." or "Real, and early." or "Not enough, not yet." are banned regardless of how cleanly they land. Write the actual sentence.

**Why:** Clipped fragments used for rhythmic emphasis are the most reliable tell of LLM-authored marketing prose. They sound clever in isolation and corny in sequence. They do not help a viewer follow the content; they only flatter the author.

**How to apply:** If a sentence is shorter than five or six words and is doing rhetorical work rather than carrying information, rewrite it as a complete sentence that includes the information the fragment was pretending to imply.

---

## 5. No self-validating tells

Words and phrases that exist to tell the viewer the content is interesting:

- "genuinely"
- "actually worth"
- "worth opening up"
- "worth pausing on"
- "the deeper idea"
- "the interesting part"
- "and that's a pretty interesting thing to have found"

These are the narrator pre-grading their own content. They never add information. They almost always indicate that the surrounding sentence is not earning its claim on its own.

**Why:** A confident piece of writing lets the content carry its own weight. If a scene's substance is genuinely interesting, the viewer will notice without being told to. If it isn't, the word "genuinely" will not rescue it.

**How to apply:** Search every draft narration for the words in the list above and delete them, then check whether the sentence still works. If it doesn't, the sentence has a real problem that needs fixing — not a missing qualifier.

---

## 6. No triumph triples at the close

The pattern "the signal is real, the math is checkable, the queue is the right size" — a parallel three-part list right before the final sentence — is a trailer beat regardless of whether the three claims are true.

**Why:** This shape reads as self-congratulation by structure alone. A viewer can hear it coming as soon as the first comma lands.

**How to apply:** Closes should state the genuine remaining open question or limitation, and then stop. If you have three things to summarize, summarize each in a separate sentence with separate evidence, or summarize none of them and let the body of the video speak.

---

## 7. Voice instructions must be all-positive

The `voice_instructions` field in plan.json describes the desired voice. Write only what the voice should be, never what it should not be. Negation prompts ("NOT a hype reel," "NOT a corporate explainer," "no smug rhetoric") teach the model the cadence of the thing being negated.

**Why:** This was confirmed by a real before/after. A v1 video using voice instructions that included "NOT a hype reel" and "opinionated, dry" produced narration that sounded pretentious. The v2, with the same TTS voice but with the negation removed and replaced with "warm, friendly, conversational, easygoing," sounded warm. Same voice, same script structure, different cadence — driven by the instruction wording.

**How to apply:** Voice instructions should be a list of positive attributes only: warm, friendly, conversational, easygoing, curious, delighted by the subject, full sentences, natural pauses. If you find yourself writing the word "not" inside voice_instructions, rewrite that clause as a positive.

---

## 8. Fix by cutting or inline rewriting — never by adding a defensive scene

When a reviewer says a scene is confusing, the fix is to cut the confusing part or rewrite it inline. The fix is not to add a new scene that explains the confusing part.

**Why:** Adding defensive context scenes is the most common failure response and it makes videos longer, slower, and more apologetic. The original suspend-signal v1 was rejected partly because a previous correction loop added two passive-aggressive "context" slides instead of fixing the confusing scene in place.

**How to apply:** After every reviewer pass, count the scenes. If the count went up, you almost certainly made the wrong fix. The correct response to "I didn't understand scene nine" is to rewrite scene nine, not to add a new scene eight-and-a-half.

---

## 9. Cold-read narration review before render is mandatory

Every video must pass a zero-context narration review before render. A separate agent, given only the narration text and no context about the project, audience, or purpose, reads it as a first-time viewer.

The review must check at minimum: where the viewer gets lost, unexplained jargon, sentences that sound smug or performative, the tone of the close, and whether the named audience matches the apparent audience.

When the review surfaces fixes, apply them inline. Do not re-review until the inline fixes are made and saved to plan.json.

**Why:** The author of a video script always has more context than the viewer. The cold-read is the only honest test of whether the script stands on its own.

**How to apply:** After the script is drafted and before any render is kicked off, spawn a fresh agent. Give it only the narration field of each scene, in order. Ask it to identify: confusion, unexplained references, smug or performative language, condescension, and audience fit. Then revise inline.

---

## 10. Scene-by-scene visual QC after image generation is mandatory

A narration review is not enough. After images have been generated (or any external image_source_path / video_source_path is attached), every single scene must be visually inspected and matched against its narration before the video is considered final. No exceptions.

This is a separate gate from rule 9. It cannot be skipped because images "look fine in the prompt" or because the narration was already reviewed. The image render does not respect the prompt's intent — it interprets the literal words. Sport leakage, financial-imagery leakage, wrong-content screenshots, stale leftover PNGs from prior plan versions, and image-source filenames that do not match their actual content have all caused defective videos in this project's history. Every one of them would have been caught by a scene-by-scene visual pass.

The required steps for each scene:

1. Open the rendered image (or play the video clip / open the screenshot file).
2. Read the scene's narration in parallel.
3. Confirm: the image's subject matches what the narration claims. If the narration says "two knobs," the image must show two knobs. If the narration says "NBA," the image must show NBA — not NFL, not soccer, not any other sport.
4. Confirm: any on-screen text in the image (titles, labels, step numbers) matches the narration's numbering and terminology. "Step 3" in narration with "STEP 4" on slide is a defect.
5. Confirm: the image is not a stale leftover from a prior version of the plan whose content no longer matches the current scene at that ID.

If any scene fails any check, fix the visual_prompt or swap the image_source_path, delete the stale PNG, and re-render that scene. Do not ship until every scene has passed visual QC.

**Why:** This project has shipped videos with NFL referees in a basketball explainer, "Step 3" labels under "step two" narration, a single-knob screenshot under a "two-knob" narration, a fire-detail slide left over from a deleted scene, and an empty wooden-box illustration just before the real-app screenshot it was supposed to introduce. Every one of those landed because the author trusted the prompt text instead of looking at the rendered output.

**How to apply:** After the render completes, before declaring the video done, list every scene id. For each: load the image file (or the video/screenshot source), read the narration, verify the match. Use a checklist; do not eyeball-skim. If any image was generated under a prior plan version (check file modification times against the most recent plan edit), treat it as suspect and re-render it. When a defect is found, fix and re-render. Only then is the video done.

---

## Image discipline (applies to every visual_prompt)

Image prompts have the same all-positive rule as voice. The strongest defense against the model rendering the wrong thing is a strong positive subject anchor, not a long negative list.

**Strong positive anchor.** If the slide is about basketball, say "NBA basketball arena, hardwood court, glowing orange basketball" — that is enough. The model will render basketball. It does not also need to be told "no American football, no soccer, no helmets." The negative list mostly pollutes the prompt and can actively summon the things it is trying to forbid.

**The "trading terminal" trap.** The phrase "trading terminal" in any image prompt pulls stock candlesticks and cryptocurrency tokens, regardless of any negative list trailing the prompt. The fix is to use a different positive anchor — instrument panel, scientific notebook, chalkboard, instrument console, dashboard — rather than to add more nots.

**When negatives are warranted.** A single targeted negative is acceptable when an earlier render of the same prompt actually produced the wrong thing. "No American football" was a legitimate fix in one specific case where "disputed play" had rendered as a football scene. That was a surgical correction for a specific observed failure, not a default. Do not turn surgical corrections into ritual incantations on every prompt.

**Richness.** A prompt that says "a dark slide with a diagram" produces a generic boring slide. A prompt that names composition, palette, lighting, surface material, the exact on-screen text quoted, and the editorial finish produces a slide worth looking at. The good prompts in `GOOD_EXAMPLES.md` are the bar.

---

## Final reminder

These rules exist because every one of them has been violated, and every violation has cost a redo. Read them again before authoring the next video.
