Prompt for Codex or Claude Code
Goal
Build a Streamlit app that converts qEEG analysis text into a patient-friendly explainer video.
	•	Output style: slideshow video (static AI-generated images + AI voiceover), not true text-to-video.
	•	Core requirement: editability. If one scene is wrong, regenerate only that scene’s assets, not the entire video.
User constraints
	•	Dyslexia + ADHD: UI must be visual, card-based, minimal walls of text, no complex timeline editor.
	•	Not visually creative: system must auto-generate image prompts from narration, and allow simple refinements like:
	◦	“make it bluer”
	◦	“remove scary imagery”
	◦	“make it more friendly”
	•	No PHI involved: only de-identified metrics.
	•	Non-commercial use: prefer open-source / permissive components.
Deliverables
Create a complete repo with this structure and full code for every file.
qeeg-explainer/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── director.py
│   ├── image_gen.py
│   ├── voice_gen.py
│   └── video_assembly.py
│
├── prompts/
│   ├── director_system.txt
│   └── refiner_system.txt
│
└── projects/   # gitignored
    └── {project_name}/
        ├── plan.json
        ├── scene_000.png
        ├── scene_000.wav
        ├── scene_001.png
        ├── scene_001.wav
        └── final.mp4
Output requirements
	•	The app runs via: streamlit run app.py
	•	The app can:
	1	Paste text and generate a storyboard plan JSON.
	2	Display scenes as editable cards.
	3	Generate an image for a single scene.
	4	Refine the image prompt with plain-language feedback and regenerate.
	5	Render a final MP4 from scene images + scene narration audio.
	6	Download the final MP4.
Hard constraints
	•	No placeholders in code output.
	•	Do not omit any files listed above.
	•	Do not use “TODO” for core functionality.
	•	Keep V1 minimal: no music, no transitions beyond simple hard cuts.
	•	Default aspect ratio: 16:9.

Architecture
The app is a deterministic, editable pipeline with a JSON intermediate.
Workflow:
	1	Input text
	2	Director Agent produces plan.json (scenes array)
	3	Human edits per-scene narration/prompt in the UI
	4	Generate assets per-scene (image, then audio)
	5	Assemble MP4
Key design: plan.json is the source of truth. Assets are generated per scene and cached on disk.

Tech stack
	•	Streamlit UI
	•	LLM for planning and prompt refinement (OpenAI or Anthropic)
	•	Image generation via Replicate (FLUX Schnell)
	•	TTS via Kokoro locally when available, otherwise OpenAI TTS fallback
	•	Video stitching via MoviePy + ffmpeg
requirements.txt
Include:
	•	streamlit
	•	python-dotenv
	•	requests
	•	Pillow
	•	replicate
	•	openai
	•	anthropic
	•	moviepy
	•	numpy
	•	soundfile
	•	any small helpers needed (pydantic optional)
System deps (document in README)
	•	ffmpeg
	•	espeak or espeak-ng (for Kokoro if required)

Data model
Use a clear schema. Either dataclass or pydantic.
plan.json:
{
  "meta": {
    "project_name": "...",
    "created_utc": "...",
    "llm_provider": "openai|anthropic",
    "image_model": "black-forest-labs/flux-schnell"
  },
  "scenes": [
    {
      "id": 0,
      "title": "optional short label",
      "narration": "...",
      "visual_prompt": "...",
      "refinement_history": [
        {"ts_utc": "...", "feedback": "...", "new_prompt": "..."}
      ],
      "image_path": "projects/<name>/scene_000.png",
      "audio_path": "projects/<name>/scene_000.wav"
    }
  ]
}
Notes:
	•	Paths can be null until generated.
	•	Use zero-padded numbering for predictable sorting.

LLM prompts
Create two prompt files and load them at runtime.
prompts/director_system.txt
Purpose: turn clinical text into 5 to 15 scenes.Output must be valid JSON only.
Include these rules:
	•	Warm, reassuring tone.
	•	Translate jargon into plain language.
	•	No diagnosis claims, no fear language.
	•	First scene: welcoming intro.
	•	Last scene: encouraging summary and next steps.
	•	Each scene should be 2 to 4 short sentences.
	•	Visuals: flat vector medical illustration, calming palette, no text in image, no scary imagery.
	•	Use metaphors sparingly and clearly.
Director JSON output format:
{ "scenes": [ { "narration": "...", "visual_prompt": "..." } ] }
prompts/refiner_system.txt
Purpose: rewrite an existing image prompt using user feedback.Output only the new prompt text.

Image generation
Implement core/image_gen.py:
	•	Use Replicate API.
	•	Default model: black-forest-labs/flux-schnell.
	•	Always append a style suffix like:
	◦	"flat vector medical illustration, clean, minimalist, friendly, trustworthy, no text"
	•	Save image to projects/<project>/scene_XXX.png.
	•	Cache behavior:
	◦	If file exists and user did not request regen, reuse it.
	◦	Regen overwrites the file.

TTS generation
Implement core/voice_gen.py:
	•	Preferred: Kokoro local TTS.
	•	If Kokoro import or runtime fails, fallback to OpenAI TTS.
	•	Save to scene_XXX.wav (preferred) or mp3, but be consistent.
	•	Cache behavior like images.

Video assembly
Implement core/video_assembly.py:
	•	For each scene:
	◦	Load image
	◦	Load audio
	◦	Set clip duration to audio duration
	◦	Attach audio
	•	Concatenate in order.
	•	Output: projects/<project>/final.mp4
	•	fps: 24
	•	codec: libx264, audio: aac

Streamlit UI
Implement app.py with a simple 3-step layout:
Step 1: Input
	•	Project name input (sidebar)
	•	Provider selector: OpenAI or Anthropic
	•	Text area for qEEG analysis
	•	Button: “Create Storyboard”
Step 2: Edit scenes
Render each scene as a bordered card:
	•	Scene number and optional short title
	•	Narration editor (text area)
	•	Image prompt hidden in expander (text area)
	•	Refinement input (single line) + button “Apply Refinement”
	•	Buttons:
	◦	“Generate Image” / “Regenerate Image”
	◦	“Generate Audio” / “Regenerate Audio”
	•	Image preview
	•	Status badges: missing image/audio, generated, error
Important Streamlit details:
	•	Use st.session_state as the live working copy.
	•	Persist changes to plan.json after edits.
	•	Widget key stability is mandatory: use deterministic keys based on scene['id'] (NOT list index). Examples:
	◦	Good: key=f"narration_{scene['id']}"
	◦	Bad: key=f"narration_{i}"
	•	Use stable widget keys for every per-scene widget: narration, visual_prompt, refinement feedback, regen buttons.
	•	Use spinners during API calls.
	•	Fail per scene without crashing the whole app.
Step 3: Render and download
	•	Button: “Render Final Video”
	•	Show errors if any scene missing image or audio.
	•	Show the video player.
	•	Provide download button.

Environment and config
Create .env.example:
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
REPLICATE_API_TOKEN=
Read env vars via python-dotenv.

Project naming and collisions (must be deterministic)
Define and implement a clear collision policy for projects/{project_name}:
	•	Default behavior: auto-increment the folder name if it already exists.
	◦	Example: if projects/report_001 exists, create projects/report_001__02 (or report_002, pick one scheme and stick to it).
	•	Provide a sidebar option: Overwrite existing project (checkbox).
	◦	If enabled: delete and recreate the existing project folder (safe recursive delete) before writing new files.
	•	The UI must clearly display the resolved final project folder name after applying the collision policy.

Error handling
Must be scene-scoped:
	•	Missing API keys: show a clear sidebar warning.
	•	Replicate failure: show the error on that scene card.
	•	Kokoro missing deps: warn and fallback to OpenAI TTS.
	•	Video render: tell the user exactly which scene is missing assets.

Sample test input
Include this in README and optionally as a “Load Sample” button:
Patient Assessment Summary:

The qEEG analysis reveals elevated beta activity (15-30 Hz) in the frontal regions,
particularly F3 and F4. This pattern is often associated with anxiety, overthinking,
or difficulty "switching off." The amplitude asymmetry shows the left frontal region
running about 15% higher than the right.

Alpha activity (8-12 Hz) in the posterior regions (P3, P4, O1, O2) is within normal
limits, suggesting good baseline relaxation capacity when the patient is able to disengage.

Theta/Beta ratio in the central regions is slightly elevated at 2.8 (normal range 1.5-2.5),
which may correlate with reported attention difficulties.

Coherence analysis shows reduced connectivity between frontal and parietal regions,
which can impact executive function and working memory integration.

Recommendations: Neurofeedback training targeting beta downtraining at F3/F4,
combined with alpha uptraining at Pz, is recommended. Expected training duration:
20-30 sessions.

Acceptance checklist
The implementation is done when:
	•	Storyboard JSON is generated and saved.
	•	Cards render and edits persist.
	•	A single scene can generate and regenerate image and audio.
	•	Final MP4 renders correctly from per-scene assets.
	•	Streamlit plays the final video and allows download.

Implementation instructions for you (Codex / Claude Code)
	•	Produce the full repository contents.
	•	Include all files and their complete code.
	•	Include a README with:
	◦	setup instructions
	◦	system dependency install commands (macOS and Linux)
	◦	how to run
	◦	troubleshooting notes
	•	Keep V1 minimal and stable. No extra features beyond this spec.
