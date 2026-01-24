"""
qEEG Explainer Video Generator

A Streamlit app that converts qEEG analysis text into patient-friendly
slideshow videos with AI-generated images and voiceover.
"""

import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.director import generate_storyboard, refine_prompt, refine_narration
from core.image_gen import generate_scene_image, edit_image
from core.voice_gen import generate_scene_audio, KOKORO_VOICES, DEFAULT_VOICE, DEFAULT_SPEED
from core.video_assembly import assemble_video, get_video_duration, preview_scene

# Load environment variables (override shell env vars with .env file)
load_dotenv(override=True)

# Constants
PROJECTS_DIR = Path(__file__).parent / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "replicate": bool(os.getenv("REPLICATE_API_TOKEN")),
    }


def get_project_path(project_name: str, overwrite: bool = False) -> Path:
    """
    Get path for a project, handling naming collisions.

    If overwrite is False and project exists, auto-increments: project__02, project__03, etc.
    If overwrite is True, returns existing path (caller should delete if needed).
    """
    base_path = PROJECTS_DIR / project_name

    if overwrite or not base_path.exists():
        return base_path

    # Auto-increment on collision
    counter = 2
    while True:
        incremented_path = PROJECTS_DIR / f"{project_name}__{counter:02d}"
        if not incremented_path.exists():
            return incremented_path
        counter += 1


def load_plan(project_dir: Path) -> dict | None:
    """Load plan.json from a project directory."""
    plan_path = project_dir / "plan.json"
    if plan_path.exists():
        try:
            return json.loads(plan_path.read_text())
        except json.JSONDecodeError as e:
            st.error(f"Corrupted plan.json in {project_dir.name}: {e}")
            return None
    return None


def save_plan(project_dir: Path, plan: dict) -> None:
    """Save plan.json to a project directory."""
    project_dir.mkdir(parents=True, exist_ok=True)
    plan_path = project_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))


def init_session_state():
    """Initialize Streamlit session state."""
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "project_dir" not in st.session_state:
        st.session_state.project_dir = None
    if "plan" not in st.session_state:
        st.session_state.plan = None
    if "tts_voice" not in st.session_state:
        st.session_state.tts_voice = DEFAULT_VOICE
    if "tts_speed" not in st.session_state:
        st.session_state.tts_speed = DEFAULT_SPEED


def get_existing_projects() -> list[str]:
    """Get list of existing project folders."""
    if not PROJECTS_DIR.exists():
        return []
    projects = []
    for p in PROJECTS_DIR.iterdir():
        if p.is_dir() and (p / "plan.json").exists():
            projects.append(p.name)
    return sorted(projects, reverse=True)  # Most recent first


def render_sidebar():
    """Render the sidebar with API key status and project info."""
    with st.sidebar:
        st.title("qEEG Video Generator")

        # API Key Status
        st.subheader("API Keys")
        keys = check_api_keys()

        for service, configured in keys.items():
            icon = "✅" if configured else "❌"
            st.write(f"{icon} {service.title()}")

        if not all(keys.values()):
            st.warning("Some API keys are missing. Check your .env file.")

        st.divider()

        # Project selector
        st.subheader("Projects")
        existing = get_existing_projects()

        if existing:
            selected = st.selectbox(
                "Open existing project",
                options=["— New Project —"] + existing,
                key="project_selector",
            )

            if selected != "— New Project —":
                if st.button("Load Project", type="primary"):
                    project_dir = PROJECTS_DIR / selected
                    plan = load_plan(project_dir)
                    if plan:
                        st.session_state.project_dir = project_dir
                        st.session_state.plan = plan
                        st.session_state.step = 2  # Go to edit scenes
                        st.rerun()
                    else:
                        st.error("Could not load project")
        else:
            st.caption("No existing projects")

        st.divider()

        # Current project info
        if st.session_state.project_dir:
            st.subheader("Current Project")
            st.write(f"📁 {st.session_state.project_dir.name}")

            if st.session_state.plan:
                num_scenes = len(st.session_state.plan.get("scenes", []))
                st.write(f"🎬 {num_scenes} scenes")

            if st.button("Close Project"):
                st.session_state.project_dir = None
                st.session_state.plan = None
                st.session_state.step = 1
                st.rerun()

        st.divider()

        # TTS Settings
        st.subheader("Voice Settings")

        # Voice selector
        voice_options = list(KOKORO_VOICES.keys())
        voice_labels = [f"{v} - {KOKORO_VOICES[v]}" for v in voice_options]
        current_voice_idx = voice_options.index(st.session_state.tts_voice) if st.session_state.tts_voice in voice_options else 0

        selected_voice = st.selectbox(
            "Voice",
            options=voice_options,
            format_func=lambda v: f"{v} - {KOKORO_VOICES[v]}",
            index=current_voice_idx,
            key="voice_selector",
            help="Choose the narrator voice",
        )
        st.session_state.tts_voice = selected_voice

        # Speed slider
        speed = st.slider(
            "Speed",
            min_value=0.8,
            max_value=1.5,
            value=st.session_state.tts_speed,
            step=0.1,
            key="speed_slider",
            help="1.0 = normal, 1.2 = 20% faster",
        )
        st.session_state.tts_speed = speed

        st.divider()

        # Navigation
        st.subheader("Steps")
        steps = ["1. Input", "2. Edit Scenes", "3. Render"]
        for i, step_name in enumerate(steps, 1):
            if i == st.session_state.step:
                st.write(f"**→ {step_name}**")
            elif i < st.session_state.step:
                st.write(f"✓ {step_name}")
            else:
                st.write(f"○ {step_name}")


def render_step_1():
    """Step 1: Input project details and qEEG text."""
    st.header("Step 1: Create Project")

    # Project name
    project_name = st.text_input(
        "Project Name",
        value="my_video",
        help="Folder name for the project (no spaces)",
    )
    project_name = project_name.replace(" ", "_")
    # Sanitize: allow only alphanumeric, underscore, dash
    project_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name)

    # Overwrite option
    existing_path = PROJECTS_DIR / project_name
    if existing_path.exists():
        overwrite = st.checkbox(
            f"Overwrite existing project '{project_name}'",
            value=False,
            help="If unchecked, a new folder with incremented name will be created",
        )
    else:
        overwrite = False

    # LLM Provider
    keys = check_api_keys()
    available_providers = [p for p in ["anthropic", "openai"] if keys.get(p)]

    if not available_providers:
        st.error("No LLM API keys configured. Please add OPENAI_API_KEY or ANTHROPIC_API_KEY to your .env file.")
        return

    provider = st.selectbox(
        "LLM Provider",
        options=available_providers,
        help="Choose which AI to use for storyboard generation",
    )

    # Input text
    input_text = st.text_area(
        "qEEG Analysis Text",
        height=300,
        placeholder="Paste the qEEG analysis text here...",
        help="The clinical text to convert into a patient-friendly video",
    )

    # Generate button
    if st.button("Generate Storyboard", type="primary", disabled=not input_text.strip()):
        with st.spinner("Generating storyboard..."):
            try:
                # Get project path
                project_dir = get_project_path(project_name, overwrite)

                # Delete existing if overwriting
                if overwrite and project_dir.exists():
                    shutil.rmtree(project_dir)

                # Create project directory
                project_dir.mkdir(parents=True, exist_ok=True)

                # Generate storyboard
                scenes = generate_storyboard(input_text, provider=provider)

                # Create plan
                plan = {
                    "meta": {
                        "project_name": project_dir.name,
                        "created_utc": datetime.utcnow().isoformat(),
                        "llm_provider": provider,
                        "image_model": "qwen/qwen-image-2512",
                        "input_text": input_text,
                    },
                    "scenes": scenes,
                }

                # Save plan
                save_plan(project_dir, plan)

                # Update session state
                st.session_state.project_dir = project_dir
                st.session_state.plan = plan
                st.session_state.step = 2

                st.rerun()

            except Exception as e:
                st.error(f"Error generating storyboard: {e}")


def render_step_2():
    """Step 2: Edit scenes and generate assets."""
    st.header("Step 2: Edit Scenes")

    plan = st.session_state.plan
    project_dir = st.session_state.project_dir
    scenes = plan["scenes"]

    # Back button
    if st.button("← Back to Input"):
        st.session_state.step = 1
        st.rerun()

    st.divider()

    # Add scene at start button
    if st.button("➕ Add Scene at Beginning", key="add_scene_start"):
        new_scene = {
            "id": 0,
            "uid": str(uuid.uuid4())[:8],  # Permanent unique ID
            "title": "New Scene",
            "narration": "",
            "visual_prompt": "",
            "refinement_history": [],
            "image_path": None,
            "audio_path": None,
        }
        scenes.insert(0, new_scene)
        # Renumber all scenes
        for idx, s in enumerate(scenes):
            s["id"] = idx
        save_plan(project_dir, plan)
        st.rerun()

    # Scene cards
    for i, scene in enumerate(scenes):
        scene_id = scene.get("id", i)
        # Use permanent uid for widget keys (generate if missing for old projects)
        if "uid" not in scene:
            scene["uid"] = str(uuid.uuid4())[:8]
            save_plan(project_dir, plan)
        scene_uid = scene["uid"]

        with st.expander(f"Scene {scene_id + 1}: {scene['title']}", expanded=i == 0):
            # Scene management buttons at top
            mgmt_col1, mgmt_col2, mgmt_col3 = st.columns([1, 1, 4])
            with mgmt_col1:
                if st.button("🗑️ Delete", key=f"delete_{scene_uid}", type="secondary"):
                    if len(scenes) > 1:  # Don't delete last scene
                        scenes.remove(scene)
                        # Renumber remaining scenes
                        for idx, s in enumerate(scenes):
                            s["id"] = idx
                        save_plan(project_dir, plan)
                        st.rerun()
                    else:
                        st.warning("Can't delete the only scene")
            with mgmt_col2:
                if st.button("➕ Add After", key=f"add_after_{scene_uid}", type="secondary"):
                    new_scene = {
                        "id": scene_id + 1,
                        "uid": str(uuid.uuid4())[:8],  # Permanent unique ID
                        "title": "New Scene",
                        "narration": "",
                        "visual_prompt": "",
                        "refinement_history": [],
                        "image_path": None,
                        "audio_path": None,
                    }
                    # Insert after current scene
                    scenes.insert(i + 1, new_scene)
                    # Renumber all scenes
                    for idx, s in enumerate(scenes):
                        s["id"] = idx
                    save_plan(project_dir, plan)
                    st.rerun()

            st.divider()

            col1, col2 = st.columns([1, 1])

            with col1:
                # Narration
                new_narration = st.text_area(
                    "Narration",
                    value=scene["narration"],
                    height=100,
                    key=f"narration_{scene_uid}",
                )

                # Update narration if changed
                if new_narration != scene["narration"]:
                    scene["narration"] = new_narration
                    save_plan(project_dir, plan)

                st.divider()

                # Refine narration
                narration_feedback = st.text_input(
                    "Refine narration (e.g., 'more concise but keep key details')",
                    key=f"narration_feedback_{scene_uid}",
                    placeholder="Enter feedback to refine the narration",
                )

                if st.button("Refine Narration", key=f"refine_narration_{scene_uid}"):
                    if narration_feedback.strip():
                        with st.spinner("Refining narration..."):
                            try:
                                provider = plan["meta"]["llm_provider"]
                                refined = refine_narration(
                                    scene["narration"],
                                    narration_feedback,
                                    provider=provider,
                                )
                                if not refined or not refined.strip():
                                    st.error("Refinement returned empty result")
                                else:
                                    scene["narration"] = refined
                                    save_plan(project_dir, plan)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Enter feedback above first")

                st.divider()

                # Visual prompt
                new_prompt = st.text_area(
                    "Visual Prompt",
                    value=scene["visual_prompt"],
                    height=100,
                    key=f"visual_prompt_{scene_uid}",
                )

                if new_prompt != scene["visual_prompt"]:
                    scene["visual_prompt"] = new_prompt
                    save_plan(project_dir, plan)

                st.divider()

                # Refinement input for visual prompt
                refinement = st.text_input(
                    "Refine visual prompt (e.g., 'make it more abstract')",
                    key=f"refinement_{scene_uid}",
                    placeholder="Enter feedback to refine the visual prompt",
                )

                # Two refinement options
                col_a, col_b = st.columns(2)

                with col_a:
                    if st.button("Refine Prompt", key=f"refine_prompt_{scene_uid}",
                                 help="Update the text prompt, then regenerate"):
                        if refinement.strip():
                            with st.spinner("Refining prompt..."):
                                try:
                                    provider = plan["meta"]["llm_provider"]
                                    refined = refine_prompt(
                                        scene["visual_prompt"],
                                        refinement,
                                        narration=scene.get("narration", ""),
                                        provider=provider,
                                    )
                                    if not refined or not refined.strip():
                                        st.error("Refinement returned empty result")
                                    else:
                                        scene["visual_prompt"] = refined
                                        save_plan(project_dir, plan)
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.warning("Enter feedback above first")

                with col_b:
                    has_image = scene.get("image_path") and Path(scene["image_path"]).exists()
                    if st.button("Edit Image", key=f"edit_image_{scene_uid}",
                                 disabled=not has_image,
                                 help="Directly edit the existing image"):
                        if not refinement.strip():
                            st.warning("Enter edit instructions above first")
                        elif has_image:
                            with st.spinner("Editing image..."):
                                try:
                                    edited_path = project_dir / "images" / f"scene_{scene_id:03d}_edited.png"
                                    edit_image(
                                        refinement,
                                        scene["image_path"],
                                        edited_path,
                                    )
                                    # Replace original with edited
                                    edited_path.rename(scene["image_path"])
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")

            with col2:
                # Image preview and generation
                st.subheader("Image")

                if scene.get("image_path") and Path(scene["image_path"]).exists():
                    st.image(scene["image_path"], use_container_width=True)

                    if st.button("Regenerate Image", key=f"regen_image_{scene_uid}"):
                        with st.spinner("Generating image..."):
                            try:
                                path = generate_scene_image(scene, project_dir)
                                scene["image_path"] = str(path)
                                save_plan(project_dir, plan)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                else:
                    st.info("No image generated yet")

                    if st.button("Generate Image", key=f"gen_image_{scene_uid}", type="primary"):
                        with st.spinner("Generating image..."):
                            try:
                                path = generate_scene_image(scene, project_dir)
                                scene["image_path"] = str(path)
                                save_plan(project_dir, plan)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                # Audio preview and generation
                st.subheader("Audio")

                if scene.get("audio_path") and Path(scene["audio_path"]).exists():
                    st.audio(scene["audio_path"])

                    if st.button("Regenerate Audio", key=f"regen_audio_{scene_uid}"):
                        with st.spinner("Generating audio..."):
                            try:
                                path = generate_scene_audio(
                                    scene, project_dir,
                                    voice=st.session_state.tts_voice,
                                    speed=st.session_state.tts_speed,
                                )
                                scene["audio_path"] = str(path)
                                save_plan(project_dir, plan)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                else:
                    st.info("No audio generated yet")

                    if st.button("Generate Audio", key=f"gen_audio_{scene_uid}", type="primary"):
                        with st.spinner("Generating audio..."):
                            try:
                                path = generate_scene_audio(
                                    scene, project_dir,
                                    voice=st.session_state.tts_voice,
                                    speed=st.session_state.tts_speed,
                                )
                                scene["audio_path"] = str(path)
                                save_plan(project_dir, plan)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                # Scene preview (requires both image and audio)
                has_image = scene.get("image_path") and Path(scene["image_path"]).exists()
                has_audio = scene.get("audio_path") and Path(scene["audio_path"]).exists()

                if has_image and has_audio:
                    st.divider()
                    st.subheader("Preview")

                    # Show existing preview if available
                    existing_preview = scene.get("preview_path")
                    if existing_preview and Path(existing_preview).exists():
                        st.video(existing_preview)

                    if st.button("▶️ Generate Preview" if not existing_preview else "🔄 Regenerate Preview",
                                 key=f"preview_{scene_uid}"):
                        with st.spinner("Rendering preview..."):
                            try:
                                preview_path = preview_scene(scene, project_dir)
                                if preview_path:
                                    scene["preview_path"] = str(preview_path)
                                    save_plan(project_dir, plan)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

    st.divider()

    # Batch generation
    st.subheader("Batch Generation")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate All Images"):
            progress = st.progress(0)
            status = st.empty()
            failed_scenes = []

            for i, scene in enumerate(scenes):
                if not scene.get("image_path") or not Path(scene["image_path"]).exists():
                    status.text(f"Generating image {i+1}/{len(scenes)}: {scene['title'][:30]}...")
                    try:
                        path = generate_scene_image(scene, project_dir)
                        scene["image_path"] = str(path)
                        save_plan(project_dir, plan)
                    except Exception as e:
                        failed_scenes.append((scene['id'], str(e)))
                        st.warning(f"Scene {scene['id']} failed: {e}")
                progress.progress((i + 1) / len(scenes))

            status.empty()
            if failed_scenes:
                st.error(f"{len(failed_scenes)} scene(s) failed. Retry individually.")
            else:
                st.success("All images generated!")
            st.rerun()

    with col2:
        if st.button("Generate All Audio"):
            progress = st.progress(0)
            status = st.empty()
            failed_scenes = []

            for i, scene in enumerate(scenes):
                if not scene.get("audio_path") or not Path(scene["audio_path"]).exists():
                    status.text(f"Generating audio {i+1}/{len(scenes)}: {scene['title'][:30]}...")
                    try:
                        path = generate_scene_audio(
                            scene, project_dir,
                            voice=st.session_state.tts_voice,
                            speed=st.session_state.tts_speed,
                        )
                        scene["audio_path"] = str(path)
                        save_plan(project_dir, plan)
                    except Exception as e:
                        failed_scenes.append((scene['id'], str(e)))
                        st.warning(f"Scene {scene['id']} failed: {e}")
                progress.progress((i + 1) / len(scenes))

            status.empty()
            if failed_scenes:
                st.error(f"{len(failed_scenes)} scene(s) failed. Retry individually.")
            else:
                st.success("All audio generated!")
            st.rerun()

    with col3:
        # Check if all assets are ready
        all_images = all(
            scene.get("image_path") and Path(scene["image_path"]).exists()
            for scene in scenes
        )
        all_audio = all(
            scene.get("audio_path") and Path(scene["audio_path"]).exists()
            for scene in scenes
        )

        if st.button(
            "Proceed to Render →",
            type="primary",
            disabled=not (all_images and all_audio),
        ):
            st.session_state.step = 3
            st.rerun()

        if not all_images:
            st.caption("⚠️ Generate all images first")
        if not all_audio:
            st.caption("⚠️ Generate all audio first")


def render_step_3():
    """Step 3: Render final video and download."""
    st.header("Step 3: Render Video")

    plan = st.session_state.plan
    project_dir = st.session_state.project_dir
    scenes = plan["scenes"]

    # Back button
    if st.button("← Back to Scenes"):
        st.session_state.step = 2
        st.rerun()

    st.divider()

    # Preview stats
    duration = get_video_duration(scenes)
    st.write(f"**Scenes:** {len(scenes)}")
    st.write(f"**Estimated Duration:** {duration:.1f} seconds ({duration/60:.1f} minutes)")

    st.divider()

    # Render options
    st.subheader("Render Settings")

    col1, col2 = st.columns(2)
    with col1:
        fps = st.selectbox("Frame Rate", [24, 30, 60], index=0)
    with col2:
        output_name = st.text_input("Output Filename", "final_video.mp4")

    # Render button
    video_path = project_dir / output_name

    if st.button("Render Video", type="primary"):
        with st.spinner("Rendering video... This may take a while."):
            try:
                path = assemble_video(
                    scenes,
                    project_dir,
                    output_filename=output_name,
                    fps=fps,
                )
                st.success("Video rendered successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error rendering video: {e}")

    # Download section
    if video_path.exists():
        st.divider()
        st.subheader("Download")

        st.video(str(video_path))

        with open(video_path, "rb") as f:
            st.download_button(
                "Download Video",
                data=f,
                file_name=output_name,
                mime="video/mp4",
                type="primary",
            )


def render_batch_queue():
    """Render the batch processing queue tab."""
    st.header("Batch Processing Queue")
    st.info("Queue multiple projects for overnight processing")

    # List existing projects
    available_projects = get_existing_projects()

    if not available_projects:
        st.warning("No existing projects found. Create some projects first.")
        return

    # Multi-select for batch queue
    selected = st.multiselect(
        "Select projects to process",
        available_projects,
        help="Choose which projects to process in batch",
    )

    if not selected:
        st.caption("Select at least one project to start batch processing")
        return

    st.divider()

    # Batch options
    st.subheader("Processing Options")

    col1, col2 = st.columns(2)
    with col1:
        generate_images = st.checkbox("Generate missing images", value=True)
        generate_audio = st.checkbox("Generate missing audio", value=True)
    with col2:
        assemble_videos = st.checkbox("Assemble final videos", value=True)
        delay_between = st.slider(
            "Delay between projects (sec)",
            min_value=5,
            max_value=60,
            value=15,
            help="Wait time between processing each project",
        )

    st.divider()

    # Show estimated work
    st.subheader("Estimated Work")
    total_missing_images = 0
    total_missing_audio = 0

    for project_name in selected:
        project_dir = PROJECTS_DIR / project_name
        plan = load_plan(project_dir)
        if plan:
            scenes = plan.get("scenes", [])
            missing_images = sum(
                1 for s in scenes
                if not s.get("image_path") or not Path(s["image_path"]).exists()
            )
            missing_audio = sum(
                1 for s in scenes
                if not s.get("audio_path") or not Path(s["audio_path"]).exists()
            )
            total_missing_images += missing_images
            total_missing_audio += missing_audio
            st.caption(f"**{project_name}**: {missing_images} images, {missing_audio} audio missing")

    st.write(f"**Total**: {total_missing_images} images, {total_missing_audio} audio to generate")

    st.divider()

    # Start batch processing
    if st.button("Start Batch Processing", type="primary"):
        batch_progress = st.progress(0)
        batch_status = st.empty()
        results = []

        for i, project_name in enumerate(selected):
            batch_status.text(f"Processing {project_name} ({i+1}/{len(selected)})")
            project_dir = PROJECTS_DIR / project_name
            plan = load_plan(project_dir)

            if not plan:
                results.append({
                    "name": project_name,
                    "images": 0,
                    "audio": 0,
                    "video": False,
                    "errors": ["Could not load plan.json"],
                })
                continue

            scenes = plan.get("scenes", [])
            project_result = {
                "name": project_name,
                "images": 0,
                "audio": 0,
                "video": False,
                "errors": [],
            }

            # Generate missing images
            if generate_images:
                for scene in scenes:
                    if not scene.get("image_path") or not Path(scene["image_path"]).exists():
                        try:
                            path = generate_scene_image(scene, project_dir)
                            scene["image_path"] = str(path)
                            project_result["images"] += 1
                        except Exception as e:
                            project_result["errors"].append(f"Image {scene['id']}: {e}")

            # Generate missing audio
            if generate_audio:
                for scene in scenes:
                    if not scene.get("audio_path") or not Path(scene["audio_path"]).exists():
                        try:
                            path = generate_scene_audio(
                                scene, project_dir,
                                voice=st.session_state.tts_voice,
                                speed=st.session_state.tts_speed,
                            )
                            scene["audio_path"] = str(path)
                            project_result["audio"] += 1
                        except Exception as e:
                            project_result["errors"].append(f"Audio {scene['id']}: {e}")

            # Assemble video if all assets ready
            if assemble_videos:
                all_images = all(
                    s.get("image_path") and Path(s["image_path"]).exists()
                    for s in scenes
                )
                all_audio = all(
                    s.get("audio_path") and Path(s["audio_path"]).exists()
                    for s in scenes
                )

                if all_images and all_audio:
                    try:
                        assemble_video(scenes, project_dir)
                        project_result["video"] = True
                    except Exception as e:
                        project_result["errors"].append(f"Video: {e}")
                else:
                    project_result["errors"].append("Video: Missing assets, skipped")

            # Save updated plan
            save_plan(project_dir, plan)
            results.append(project_result)
            batch_progress.progress((i + 1) / len(selected))

            # Delay between projects (except for last one)
            if i < len(selected) - 1:
                batch_status.text(f"Waiting {delay_between}s before next project...")
                time.sleep(delay_between)

        batch_status.empty()

        # Show summary
        st.success("Batch processing complete!")

        st.subheader("Results Summary")
        for r in results:
            status_icon = "❌" if r["errors"] else "✅"
            video_status = "📹" if r["video"] else ""
            st.write(f"{status_icon} **{r['name']}**: {r['images']} images, {r['audio']} audio generated {video_status}")
            if r["errors"]:
                for err in r["errors"]:
                    st.caption(f"  └ {err}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="qEEG Video Generator",
        page_icon="🧠",
        layout="wide",
    )

    init_session_state()
    render_sidebar()

    # Add tabs for main workflow vs batch queue
    tab1, tab2 = st.tabs(["Project Workflow", "Batch Queue"])

    with tab1:
        # Main content based on current step
        if st.session_state.step == 1:
            render_step_1()
        elif st.session_state.step == 2:
            render_step_2()
        elif st.session_state.step == 3:
            render_step_3()

    with tab2:
        render_batch_queue()


if __name__ == "__main__":
    main()
