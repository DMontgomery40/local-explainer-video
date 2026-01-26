"""
qEEG Explainer Video Generator

A Streamlit app that converts qEEG analysis text into patient-friendly
slideshow videos with AI-generated images and voiceover.
"""

import json
import os
import queue
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.director import generate_storyboard, refine_prompt, refine_narration
from core.image_gen import generate_scene_image, edit_image
from core.voice_gen import (
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_ELEVENLABS_SIMILARITY_BOOST,
    DEFAULT_ELEVENLABS_SPEED,
    DEFAULT_ELEVENLABS_STABILITY,
    DEFAULT_ELEVENLABS_STYLE,
    DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
    DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST,
    DEFAULT_ELEVENLABS_VOICE,
    DEFAULT_EXAGGERATION,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    ELEVENLABS_VOICES,
    KOKORO_VOICES,
    generate_scene_audio,
 )
from core.video_assembly import assemble_video, get_video_duration, preview_scene
from core.qc_publish import (
    QCPublishConfig,
    QCPublishError,
    default_cliproxy_api_key,
    default_cliproxy_url,
    default_qeeg_analysis_dir,
    default_qeeg_backend_url,
    infer_patient_id,
    qc_and_publish_project,
)

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
        "dashscope": bool(os.getenv("DASHSCOPE_API_KEY")),
        "elevenlabs": bool(os.getenv("ELEVENLABS_API_KEY")),
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
    if "tts_provider" not in st.session_state:
        st.session_state.tts_provider = "kokoro"  # Default to free local TTS
    if "tts_exaggeration" not in st.session_state:
        st.session_state.tts_exaggeration = DEFAULT_EXAGGERATION

    # ElevenLabs-specific settings (kept separate from Kokoro voice/speed)
    if "tts_elevenlabs_voice" not in st.session_state:
        st.session_state.tts_elevenlabs_voice = DEFAULT_ELEVENLABS_VOICE
    if "tts_elevenlabs_speed" not in st.session_state:
        st.session_state.tts_elevenlabs_speed = float(DEFAULT_ELEVENLABS_SPEED)
    if "tts_elevenlabs_model_id" not in st.session_state:
        st.session_state.tts_elevenlabs_model_id = DEFAULT_ELEVENLABS_MODEL
    if "tts_elevenlabs_text_normalization" not in st.session_state:
        st.session_state.tts_elevenlabs_text_normalization = DEFAULT_ELEVENLABS_TEXT_NORMALIZATION
    if "tts_elevenlabs_stability" not in st.session_state:
        st.session_state.tts_elevenlabs_stability = DEFAULT_ELEVENLABS_STABILITY
    if "tts_elevenlabs_similarity_boost" not in st.session_state:
        st.session_state.tts_elevenlabs_similarity_boost = DEFAULT_ELEVENLABS_SIMILARITY_BOOST
    if "tts_elevenlabs_style" not in st.session_state:
        st.session_state.tts_elevenlabs_style = DEFAULT_ELEVENLABS_STYLE
    if "tts_elevenlabs_use_speaker_boost" not in st.session_state:
        st.session_state.tts_elevenlabs_use_speaker_boost = DEFAULT_ELEVENLABS_USE_SPEAKER_BOOST

    # Image edit model (used for both UI "Edit Image" and QC auto-fix slide text).
    if "image_edit_model" not in st.session_state:
        st.session_state.image_edit_model = (
            (os.getenv("IMAGE_EDIT_MODEL") or "").strip()
            or ("qwen-image-edit-max" if (os.getenv("DASHSCOPE_API_KEY") or "").strip() else "qwen/qwen-image-edit-2511")
        )

    # DashScope-specific image edit parameters
    if "dashscope_edit_n" not in st.session_state:
        st.session_state.dashscope_edit_n = 1
    if "dashscope_edit_seed" not in st.session_state:
        st.session_state.dashscope_edit_seed = ""  # Empty string = random
    if "dashscope_edit_negative_prompt" not in st.session_state:
        st.session_state.dashscope_edit_negative_prompt = ""
    if "dashscope_edit_prompt_extend" not in st.session_state:
        st.session_state.dashscope_edit_prompt_extend = True


def _tts_kwargs_from_state() -> dict:
    """Build kwargs for generate_scene_audio() based on current sidebar settings."""
    provider = st.session_state.tts_provider
    kwargs: dict = {"tts_provider": provider}

    if provider == "kokoro":
        kwargs.update(
            {
                "voice": st.session_state.tts_voice,
                "speed": float(st.session_state.tts_speed),
            }
        )
        return kwargs

    if provider == "elevenlabs":
        kwargs.update(
            {
                "voice": st.session_state.tts_elevenlabs_voice,
                "speed": float(st.session_state.tts_elevenlabs_speed),
                "elevenlabs_model_id": st.session_state.tts_elevenlabs_model_id,
                "elevenlabs_apply_text_normalization": st.session_state.tts_elevenlabs_text_normalization,
                "elevenlabs_stability": float(st.session_state.tts_elevenlabs_stability),
                "elevenlabs_similarity_boost": float(st.session_state.tts_elevenlabs_similarity_boost),
                "elevenlabs_style": float(st.session_state.tts_elevenlabs_style),
                "elevenlabs_use_speaker_boost": bool(st.session_state.tts_elevenlabs_use_speaker_boost),
            }
        )
        return kwargs

    # OpenAI (no extra settings here; voice is handled inside core/voice_gen.py)
    return kwargs


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

        # Only warn on keys that block core workflows
        if not keys.get("replicate"):
            st.warning("REPLICATE_API_TOKEN missing (required for image generation).")
        if not (keys.get("openai") or keys.get("anthropic")):
            st.warning("Missing LLM key (set OPENAI_API_KEY or ANTHROPIC_API_KEY).")

        st.divider()

        # Image edit settings
        st.subheader("Image Edit")
        edit_models: list[str] = []
        if keys.get("dashscope"):
            edit_models.extend(["qwen-image-edit-max", "qwen-image-edit-plus", "qwen-image-edit"])
        edit_models.append("qwen/qwen-image-edit-2511")
        # Ensure selection is valid
        if st.session_state.image_edit_model not in edit_models:
            st.session_state.image_edit_model = edit_models[0]
        st.selectbox(
            "Image edit model",
            options=edit_models,
            key="image_edit_model",
            help="Used for the per-scene 'Edit Image' button and QC auto-fix slide text edits.",
        )
        if str(st.session_state.image_edit_model).startswith("qwen-image-edit") and not keys.get("dashscope"):
            st.warning("DASHSCOPE_API_KEY missing (required for DashScope image edits).")
        if "/" in str(st.session_state.image_edit_model) and not keys.get("replicate"):
            st.warning("REPLICATE_API_TOKEN missing (required for Replicate image edits).")

        # DashScope-only controls (hidden when Replicate model is selected)
        is_dashscope_model = str(st.session_state.image_edit_model).startswith("qwen-image-edit")
        if is_dashscope_model:
            st.caption("DashScope Parameters")
            st.session_state.dashscope_edit_n = st.slider(
                "Variants (n)",
                min_value=1,
                max_value=6,
                value=int(st.session_state.dashscope_edit_n),
                key="dashscope_n_slider",
                help="Generate 1-6 output variants per edit (max/plus models only)",
            )
            seed_input = st.text_input(
                "Seed",
                value=str(st.session_state.dashscope_edit_seed),
                key="dashscope_seed_input",
                help="Integer 0-2147483647 for reproducible edits. Leave empty for random.",
            )
            st.session_state.dashscope_edit_seed = seed_input.strip()
            st.session_state.dashscope_edit_prompt_extend = st.checkbox(
                "Prompt extend",
                value=bool(st.session_state.dashscope_edit_prompt_extend),
                key="dashscope_prompt_extend_checkbox",
                help="Let DashScope expand your prompt for better results (default: on)",
            )
            st.session_state.dashscope_edit_negative_prompt = st.text_input(
                "Negative prompt",
                value=str(st.session_state.dashscope_edit_negative_prompt),
                key="dashscope_negative_prompt_input",
                help="Things to avoid in the output (max 500 chars)",
            )

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

        # TTS Provider selector
        tts_providers = {
            "kokoro": "Kokoro (Free, Local)",
            "elevenlabs": "ElevenLabs (Flash v2.5, Premium)",
            "openai": "OpenAI TTS",
        }
        provider_keys = list(tts_providers.keys())
        current_provider = st.session_state.tts_provider if st.session_state.tts_provider in provider_keys else provider_keys[0]
        selected_provider = st.selectbox(
            "TTS Provider",
            options=provider_keys,
            format_func=lambda p: tts_providers[p],
            index=provider_keys.index(current_provider),
            key="tts_provider_selector",
            help="ElevenLabs Flash v2.5 expects narration numbers to be written out as words (no digits).",
        )
        st.session_state.tts_provider = selected_provider

        # Provider-specific settings
        if selected_provider == "kokoro":
            # Voice selector (Kokoro only)
            voice_options = list(KOKORO_VOICES.keys())
            current_voice_idx = voice_options.index(st.session_state.tts_voice) if st.session_state.tts_voice in voice_options else 0

            selected_voice = st.selectbox(
                "Voice",
                options=voice_options,
                format_func=lambda v: f"{v} - {KOKORO_VOICES[v]}",
                index=current_voice_idx,
                key="kokoro_voice_selector",
                help="Choose the narrator voice",
            )
            st.session_state.tts_voice = selected_voice

            # Speed slider (Kokoro only)
            speed = st.slider(
                "Speed",
                min_value=0.8,
                max_value=1.5,
                value=st.session_state.tts_speed,
                step=0.1,
                key="kokoro_speed_slider",
                help="1.0 = normal, 1.2 = 20% faster",
            )
            st.session_state.tts_speed = speed

        elif selected_provider == "elevenlabs":
            if not keys.get("elevenlabs"):
                st.error("ELEVENLABS_API_KEY missing. Add it to .env to use ElevenLabs TTS.")
            voice_options = list(ELEVENLABS_VOICES.keys())
            current_voice_idx = (
                voice_options.index(st.session_state.tts_elevenlabs_voice)
                if st.session_state.tts_elevenlabs_voice in voice_options
                else 0
            )
            selected_voice = st.selectbox(
                "Voice",
                options=voice_options,
                format_func=lambda v: f"{v} - {ELEVENLABS_VOICES[v][1]}",
                index=current_voice_idx,
                key="elevenlabs_voice_selector",
                help="Choose the ElevenLabs narrator voice",
            )
            st.session_state.tts_elevenlabs_voice = selected_voice

            st.caption("Narration should spell out numbers (no digits) for best Flash v2.5 results.")

            # Voice settings: 0..1
            st.session_state.tts_elevenlabs_stability = st.slider(
                "Stability",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.tts_elevenlabs_stability),
                step=0.05,
                key="elevenlabs_stability",
                help="Lower = more variation, higher = more consistent/steady",
            )
            st.session_state.tts_elevenlabs_similarity_boost = st.slider(
                "Similarity Boost",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.tts_elevenlabs_similarity_boost),
                step=0.05,
                key="elevenlabs_similarity_boost",
                help="Higher = closer to the base voice",
            )
            st.session_state.tts_elevenlabs_style = st.slider(
                "Style",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.tts_elevenlabs_style),
                step=0.05,
                key="elevenlabs_style",
                help="Style exaggeration (may increase latency if > 0)",
            )
            st.session_state.tts_elevenlabs_speed = st.slider(
                "Speed",
                min_value=0.7,
                max_value=1.4,
                value=float(st.session_state.tts_elevenlabs_speed),
                step=0.05,
                key="elevenlabs_speed",
                help="1.0 = normal",
            )
            st.session_state.tts_elevenlabs_use_speaker_boost = st.checkbox(
                "Use Speaker Boost",
                value=bool(st.session_state.tts_elevenlabs_use_speaker_boost),
                key="elevenlabs_use_speaker_boost",
                help="Boosts similarity; can slightly increase latency",
            )

            st.session_state.tts_elevenlabs_text_normalization = st.selectbox(
                "Text Normalization",
                options=["auto", "on", "off"],
                index=["auto", "on", "off"].index(str(st.session_state.tts_elevenlabs_text_normalization)),
                key="elevenlabs_text_normalization",
                help="Controls ElevenLabs text normalization (may affect latency; Flash v2.5 may limit this on some plans).",
            )

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
                                    # Build kwargs for edit_image
                                    edit_kwargs: dict = {
                                        "model": st.session_state.image_edit_model,
                                    }
                                    # Add DashScope params if using a DashScope model
                                    if str(st.session_state.image_edit_model).startswith("qwen-image-edit"):
                                        edit_kwargs["n"] = int(st.session_state.dashscope_edit_n)
                                        edit_kwargs["prompt_extend"] = bool(st.session_state.dashscope_edit_prompt_extend)
                                        neg = str(st.session_state.dashscope_edit_negative_prompt).strip()
                                        edit_kwargs["negative_prompt"] = neg if neg else " "
                                        seed_str = str(st.session_state.dashscope_edit_seed).strip()
                                        if seed_str.isdigit():
                                            edit_kwargs["seed"] = int(seed_str)
                                    edit_image(
                                        refinement,
                                        scene["image_path"],
                                        edited_path,
                                        **edit_kwargs,
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
                                path = generate_scene_audio(scene, project_dir, **_tts_kwargs_from_state())
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
                                path = generate_scene_audio(scene, project_dir, **_tts_kwargs_from_state())
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
        if st.button("Generate Missing Images", key="batch_gen_images"):
            progress = st.progress(0)
            status = st.empty()
            failed_scenes = []
            generated = 0
            skipped = 0

            for i, scene in enumerate(scenes):
                has_image = scene.get("image_path") and Path(scene["image_path"]).exists()
                if not has_image:
                    status.text(f"Generating image {i+1}/{len(scenes)}: {scene['title'][:30]}...")
                    try:
                        path = generate_scene_image(scene, project_dir)
                        scene["image_path"] = str(path)
                        save_plan(project_dir, plan)
                        generated += 1
                    except Exception as e:
                        failed_scenes.append((scene['id'], str(e)))
                        st.warning(f"Scene {scene['id']} failed: {e}")
                else:
                    skipped += 1
                progress.progress((i + 1) / len(scenes))

            status.empty()
            if failed_scenes:
                st.error(
                    f"Images complete: generated {generated}, skipped {skipped}, failed {len(failed_scenes)}. "
                    "Retry failed scenes individually."
                )
            else:
                st.success(f"Images complete: generated {generated}, skipped {skipped}.")

    with col2:
        a, b = st.columns(2)
        gen_missing = a.button("Generate Missing Audio", key="batch_gen_audio")
        regen_all = b.button("Regenerate All Audio", key="batch_regen_audio", help="Overwrites existing scene audio files")
        if gen_missing or regen_all:
            progress = st.progress(0)
            status = st.empty()
            failed_scenes = []
            generated = 0
            skipped = 0

            for i, scene in enumerate(scenes):
                has_audio = scene.get("audio_path") and Path(scene["audio_path"]).exists()
                should_generate = regen_all or (not has_audio)
                if should_generate:
                    status.text(f"Generating audio {i+1}/{len(scenes)}: {scene['title'][:30]}...")
                    try:
                        path = generate_scene_audio(scene, project_dir, **_tts_kwargs_from_state())
                        scene["audio_path"] = str(path)
                        save_plan(project_dir, plan)
                        generated += 1
                    except Exception as e:
                        failed_scenes.append((scene['id'], str(e)))
                        st.warning(f"Scene {scene['id']} failed: {e}")
                else:
                    skipped += 1
                progress.progress((i + 1) / len(scenes))

            status.empty()
            if failed_scenes:
                st.error(
                    f"Audio complete: generated {generated}, skipped {skipped}, failed {len(failed_scenes)}. "
                    "Retry failed scenes individually."
                )
            else:
                msg = f"Audio complete: generated {generated}, skipped {skipped}."
                if gen_missing and generated == 0:
                    msg += " (Everything already had audio. Use 'Regenerate All Audio' to force.)"
                st.success(msg)

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

    st.divider()
    st.subheader("Regenerate Everything")
    st.caption("Regenerates ALL images (Replicate) and ALL audio (TTS) in parallel. This overwrites existing assets.")

    if st.button("Regenerate Everything (Images + Audio)", type="primary", key="regen_everything_parallel"):
        tts_kwargs = _tts_kwargs_from_state()
        total = len(scenes)
        if total == 0:
            st.warning("No scenes found.")
            return

        img_col, aud_col = st.columns(2)
        with img_col:
            st.markdown("**Images (Replicate)**")
            img_status = st.empty()
            img_progress = st.progress(0.0)
        with aud_col:
            st.markdown("**Audio (TTS)**")
            aud_status = st.empty()
            aud_progress = st.progress(0.0)

        overall = st.empty()

        q: queue.Queue = queue.Queue()
        lock = threading.Lock()

        img_failed: list[tuple[int, str]] = []
        aud_failed: list[tuple[int, str]] = []
        img_done = 0
        aud_done = 0

        def _regen_images() -> None:
            try:
                for i, scene in enumerate(scenes):
                    sid = int(scene.get("id", i))
                    try:
                        path = generate_scene_image(scene, project_dir)
                        with lock:
                            scene["image_path"] = str(path)
                            save_plan(project_dir, plan)
                        q.put(("image_ok", i + 1, total, sid))
                    except Exception as e:
                        q.put(("image_err", i + 1, total, sid, str(e)))
            finally:
                q.put(("image_done",))

        def _regen_audio() -> None:
            try:
                for i, scene in enumerate(scenes):
                    sid = int(scene.get("id", i))
                    try:
                        path = generate_scene_audio(scene, project_dir, **tts_kwargs)
                        with lock:
                            scene["audio_path"] = str(path)
                            save_plan(project_dir, plan)
                        q.put(("audio_ok", i + 1, total, sid))
                    except Exception as e:
                        q.put(("audio_err", i + 1, total, sid, str(e)))
            finally:
                q.put(("audio_done",))

        t_img = threading.Thread(target=_regen_images, daemon=True)
        t_aud = threading.Thread(target=_regen_audio, daemon=True)
        t_img.start()
        t_aud.start()

        overall.info("Running… (images + audio in parallel)")

        img_finished = False
        aud_finished = False

        # Main loop: update UI from events emitted by worker threads
        while not (img_finished and aud_finished):
            try:
                evt = q.get(timeout=0.2)
            except Exception:
                evt = None

            if not evt:
                # Keep the UI responsive during long calls
                continue

            kind = evt[0]
            if kind == "image_ok":
                _, n, tot, sid = evt
                img_progress.progress(n / tot)
                img_status.text(f"Scene {sid}: {n}/{tot}")
                img_done = n
            elif kind == "image_err":
                _, n, tot, sid, msg = evt
                img_progress.progress(n / tot)
                img_status.text(f"Scene {sid}: {n}/{tot} (error)")
                img_failed.append((sid, msg))
                img_done = n
            elif kind == "image_done":
                img_finished = True
                img_progress.progress(1.0)
                img_status.text("Done.")
            elif kind == "audio_ok":
                _, n, tot, sid = evt
                aud_progress.progress(n / tot)
                aud_status.text(f"Scene {sid}: {n}/{tot}")
                aud_done = n
            elif kind == "audio_err":
                _, n, tot, sid, msg = evt
                aud_progress.progress(n / tot)
                aud_status.text(f"Scene {sid}: {n}/{tot} (error)")
                aud_failed.append((sid, msg))
                aud_done = n
            elif kind == "audio_done":
                aud_finished = True
                aud_progress.progress(1.0)
                aud_status.text("Done.")

        # Ensure threads ended
        t_img.join(timeout=0.1)
        t_aud.join(timeout=0.1)

        if img_failed or aud_failed:
            overall.error(
                f"Regenerate complete with errors. Images failed: {len(img_failed)}. Audio failed: {len(aud_failed)}."
            )
        else:
            overall.success("Regenerate complete: all images and audio regenerated.")

        if img_failed:
            with st.expander(f"Image failures ({len(img_failed)})"):
                for sid, msg in img_failed[:50]:
                    st.write(f"- Scene {sid}: {msg}")
        if aud_failed:
            with st.expander(f"Audio failures ({len(aud_failed)})"):
                for sid, msg in aud_failed[:50]:
                    st.write(f"- Scene {sid}: {msg}")


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
        fps = st.selectbox("Frame Rate", [24, 30, 60], index=0, key="render_fps")
    with col2:
        output_name = st.text_input("Output Filename", "final_video.mp4", key="render_output_name")

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

    st.divider()
    st.subheader("QC + Publish")
    st.caption(
        "Runs a final verification pass against qEEG Council ground truth, optionally fixes slide text via image-edit (no regeneration), "
        "re-renders the video, then publishes the MP4 to qEEG Council + the clinician portal sync folder."
    )

    guessed_patient_id = infer_patient_id(project_dir.name) if project_dir else None
    patient_id = st.text_input(
        "Patient ID (MM-DD-YYYY-N)",
        value=guessed_patient_id or "",
        help="Used to locate the latest qEEG Council run (Stage 4 consolidation + Stage 1 data pack).",
        key="qc_patient_id",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        qeeg_dir = st.text_input(
            "qEEG Council repo directory",
            value=str(default_qeeg_analysis_dir()),
            help="Path to the qEEG Council repo (contains data/app.db and artifacts).",
            key="qc_qeeg_dir",
        )
        backend_url = st.text_input(
            "qEEG Council backend URL",
            value=default_qeeg_backend_url(),
            help="Used to upload the MP4 via /api/patients/{patient_uuid}/files.",
            key="qc_backend_url",
        )
    with col_b:
        cliproxy_url = st.text_input(
            "CLIProxyAPI URL",
            value=default_cliproxy_url(),
            help="Used for Gemini visual QC via OpenAI-compatible /v1/chat/completions.",
            key="qc_cliproxy_url",
        )
        cliproxy_api_key = st.text_input(
            "CLIProxyAPI key (optional)",
            value=default_cliproxy_api_key(),
            type="password",
            key="qc_cliproxy_api_key",
        )

    max_passes = st.number_input(
        "Max visual QC passes",
        min_value=1,
        max_value=10,
        value=5,
        help="Only used when auto-fix is enabled (recheck images after edits until all clear).",
        key="qc_max_passes",
    )

    auto_fix_images = st.checkbox(
        "Auto-fix slide text (image edit)",
        value=False,
        help=(
            "If enabled, QC will attempt deterministic text edits on the existing PNGs via Qwen Image Edit. "
            "If disabled, QC will only report issues (writes qc_visual_issues.json) and block publish."
        ),
        key="qc_auto_fix_images",
    )

    run_qc = st.button("Run QC + Publish", type="primary", key="qc_publish_btn")
    if run_qc:
        if not patient_id.strip():
            st.error("Enter a Patient ID (MM-DD-YYYY-N) to run QC.")
            return

        status = st.empty()
        progress = st.progress(0.0)
        log_box = st.empty()
        log_lines: list[str] = []

        def _log(msg: str) -> None:
            log_lines.append(msg)
            log_box.code("\n".join(log_lines[-250:]))

        def _phase(p: str) -> None:
            status.info(p)

        def _progress(v: float) -> None:
            try:
                progress.progress(max(0.0, min(1.0, float(v))))
            except Exception:
                pass

        try:
            # Ensure QC audio regeneration uses the same provider/settings as the UI
            tts_provider = st.session_state.tts_provider
            if tts_provider == "elevenlabs":
                qc_tts_voice = st.session_state.tts_elevenlabs_voice
                qc_tts_speed = float(st.session_state.tts_elevenlabs_speed)
                qc_elevenlabs = {
                    "elevenlabs_model_id": st.session_state.tts_elevenlabs_model_id,
                    "elevenlabs_apply_text_normalization": st.session_state.tts_elevenlabs_text_normalization,
                    "elevenlabs_stability": float(st.session_state.tts_elevenlabs_stability),
                    "elevenlabs_similarity_boost": float(st.session_state.tts_elevenlabs_similarity_boost),
                    "elevenlabs_style": float(st.session_state.tts_elevenlabs_style),
                    "elevenlabs_use_speaker_boost": bool(st.session_state.tts_elevenlabs_use_speaker_boost),
                }
            else:
                qc_tts_voice = st.session_state.tts_voice
                qc_tts_speed = float(st.session_state.tts_speed)
                qc_elevenlabs = {}

            cfg = QCPublishConfig(
                qeeg_dir=Path(qeeg_dir).expanduser(),
                backend_url=backend_url,
                cliproxy_url=cliproxy_url,
                cliproxy_api_key=cliproxy_api_key,
                max_visual_passes=int(max_passes),
                auto_fix_images=bool(auto_fix_images),
                fps=int(fps),
                output_filename=str(output_name or "final_video.mp4"),
                tts_provider=tts_provider,
                tts_voice=qc_tts_voice,
                tts_speed=qc_tts_speed,
                image_edit_model=str(st.session_state.image_edit_model),
                **qc_elevenlabs,
            )

            updated_plan, summary = qc_and_publish_project(
                project_dir=project_dir,
                plan=plan,
                patient_id=patient_id.strip(),
                config=cfg,
                log=_log,
                set_phase=_phase,
                set_progress=_progress,
            )
            st.session_state.plan = updated_plan
            save_plan(project_dir, updated_plan)

            st.success("QC + Publish complete.")
            st.write(f"**Run:** {summary.run_id}")
            st.write(f"**Video:** {summary.video_path}")
            if summary.portal_copy_path:
                st.write(f"**Portal folder:** {summary.portal_copy_path}")
            st.write(f"**Backend upload:** {'ok' if summary.backend_upload_ok else 'failed'}")
        except QCPublishError as e:
            st.error(str(e))
            issues_path = project_dir / "qc_visual_issues.json"
            if issues_path.exists():
                try:
                    payload = json.loads(issues_path.read_text(encoding="utf-8"))
                    issues = payload.get("issues", []) if isinstance(payload, dict) else []
                    if isinstance(issues, list) and issues:
                        st.info(f"Visual QC issues written to: {issues_path}")
                        rows = []
                        for item in issues:
                            if not isinstance(item, dict):
                                continue
                            scene_id = item.get("scene_id")
                            slide_num = item.get("slide_num")
                            repls = item.get("replacements") or []
                            if isinstance(repls, list) and repls:
                                for r in repls:
                                    if not isinstance(r, dict):
                                        continue
                                    rows.append(
                                        {
                                            "scene_id": scene_id,
                                            "slide": slide_num,
                                            "from": r.get("from"),
                                            "to": r.get("to"),
                                            "why": r.get("why"),
                                            "where": r.get("where"),
                                        }
                                    )
                            else:
                                rows.append(
                                    {
                                        "scene_id": scene_id,
                                        "slide": slide_num,
                                        "from": None,
                                        "to": None,
                                        "why": None,
                                        "where": None,
                                    }
                                )
                        if rows:
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                        with st.expander("Raw qc_visual_issues.json"):
                            st.code(json.dumps(payload, indent=2), language="json")
                except Exception as e2:
                    st.warning(f"Could not read visual QC report ({issues_path}): {e2}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


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
                            path = generate_scene_audio(scene, project_dir, **_tts_kwargs_from_state())
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
