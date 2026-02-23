"""LLM-based storyboard generation for qEEG explainer videos."""

import json
import re
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping

import anthropic
import openai

# Singleton LLM clients
_openai_client = None
_anthropic_client = None


def _get_openai_client():
    """Get or create singleton OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client


def _get_anthropic_client():
    """Get or create singleton Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


# Cached prompts
_PROMPTS: dict[str, str] = {}

_ELECTRODE_LABEL_RE = re.compile(
    r"\b(?:Fp1|Fp2|Fpz|F7|F3|Fz|F4|F8|T3|C3|Cz|C4|T4|T5|P3|Pz|P4|T6|O1|O2|Oz|A1|A2|T7|T8|P7|P8)\b",
    flags=re.IGNORECASE,
)
_BLENDER_HINT_TERMS: tuple[str, ...] = (
    "electrode",
    "topomap",
    "topography",
    "coherence",
    "connectivity",
    "brain map",
    "eeg",
)


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory (cached)."""
    if name not in _PROMPTS:
        prompt_path = Path(__file__).parent.parent / "prompts" / f"{name}.txt"
        _PROMPTS[name] = prompt_path.read_text(encoding="utf-8")
    return _PROMPTS[name]


def _build_director_system_prompt() -> str:
    """
    Build the director system prompt, optionally appending Blender-scene skill guidance.

    The base prompt still works on its own; the Blender skill file can be added/iterated
    without changing Python code.
    """
    base = load_prompt("director_system").rstrip()
    skill_name = "director_blender_skill"
    skill_path = Path(__file__).parent.parent / "prompts" / f"{skill_name}.txt"
    if not skill_path.exists():
        return base
    skill = load_prompt(skill_name).strip()
    if not skill:
        return base
    return f"{base}\n\n{skill}"


def generate_storyboard(
    input_text: str,
    provider: Literal["openai", "anthropic"] = "openai",
) -> list[dict]:
    """
    Generate a video storyboard from qEEG analysis text.

    Args:
        input_text: The qEEG analysis text to convert
        provider: LLM provider to use ("openai" or "anthropic")

    Returns:
        List of scene dictionaries with id, title, narration, visual_prompt
    """
    system_prompt = _build_director_system_prompt()
    user_prompt = f"""Create a storyboard for an explainer video based on this qEEG analysis.

CRITICAL: Total narration must be 950-1,100 words (6-7 min video).
Target ~15 scenes, ~50 words average per content scene. Count as you go.

---
{input_text}
---

Return the storyboard as a JSON array of scenes."""

    if provider == "openai":
        return _generate_with_openai(system_prompt, user_prompt)
    elif provider == "anthropic":
        return _generate_with_anthropic(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _generate_with_openai(system_prompt: str, user_prompt: str) -> list[dict]:
    """Generate storyboard using OpenAI GPT-5.1 Responses API."""
    client = _get_openai_client()

    response = client.responses.create(
        model="gpt-5.1",
        instructions=system_prompt,
        input=user_prompt,
        text={
            "format": {
                "type": "json_object"
            }
        },
        temperature=0.7,
    )

    content = response.output_text
    result = json.loads(content)

    # Handle both direct array and wrapped object responses
    if isinstance(result, list):
        scenes = result
    elif isinstance(result, dict) and "scenes" in result:
        scenes = result["scenes"]
    else:
        # Try to find any array in the response
        for value in result.values():
            if isinstance(value, list):
                scenes = value
                break
        else:
            raise ValueError("Could not find scenes array in response")

    return _validate_scenes(scenes)


def _generate_with_anthropic(system_prompt: str, user_prompt: str) -> list[dict]:
    """Generate storyboard using Anthropic Claude Sonnet 4 with extended thinking."""
    client = _get_anthropic_client()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        thinking={
            "type": "enabled",
            "budget_tokens": 4096,
        },
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    # With extended thinking, find the text block (not thinking block)
    content = None
    for block in response.content:
        if block.type == "text":
            content = block.text
            break

    if not content:
        raise ValueError("No text response from model")

    # Extract JSON from response (Claude may wrap it in markdown)
    if "```json" in content:
        start = content.index("```json") + 7
        end = content.index("```", start)
        content = content[start:end].strip()
    elif "```" in content:
        start = content.index("```") + 3
        end = content.index("```", start)
        content = content[start:end].strip()

    result = json.loads(content)

    # Handle both direct array and wrapped object responses
    if isinstance(result, list):
        scenes = result
    elif isinstance(result, dict) and "scenes" in result:
        scenes = result["scenes"]
    else:
        for value in result.values():
            if isinstance(value, list):
                scenes = value
                break
        else:
            raise ValueError("Could not find scenes array in response")

    return _validate_scenes(scenes)


def _looks_like_blender_scene(scene: Mapping[str, Any]) -> bool:
    backend = str(scene.get("render_backend") or "").strip().lower()
    if backend == "blender":
        return True
    if isinstance(scene.get("blender"), Mapping):
        return True

    text_parts = []
    for key in ("visual_prompt", "title", "subtitle"):
        value = scene.get(key)
        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip().lower())
    text = " ".join(text_parts)
    if not text:
        return False
    if _ELECTRODE_LABEL_RE.search(text):
        return True
    if "brain" in text and any(term in text for term in _BLENDER_HINT_TERMS):
        return True
    return False


def _validate_scenes(scenes: list[dict]) -> list[dict]:
    """Validate and normalize scene data."""
    validated = []
    for i, scene in enumerate(scenes):
        narration = scene.get("narration", "").strip()
        visual_prompt = scene.get("visual_prompt", "").strip()

        if not narration:
            raise ValueError(f"Scene {i+1} has empty narration")
        if not visual_prompt:
            raise ValueError(f"Scene {i+1} has empty visual prompt")

        normalized: dict[str, Any] = {
            "id": scene.get("id", i),
            "uid": scene.get("uid", str(uuid.uuid4())[:8]),
            "title": scene.get("title", f"Scene {i + 1}"),
            "narration": narration,
            "visual_prompt": visual_prompt,
            "refinement_history": scene.get("refinement_history", []),
            "image_path": scene.get("image_path"),
            "audio_path": scene.get("audio_path"),
        }

        # Preserve optional structured fields the director may emit for Blender scenes.
        for key in (
            "render_backend",
            "subtitle",
            "footer",
            "band",
            "metric",
            "session_index",
            "qeeg_extract",
            "electrode_values",
            "coherence_edges",
            "scene_type",
            "style_preset",
            "camera_preset",
        ):
            if key in scene:
                normalized[key] = scene.get(key)
        blender_payload = scene.get("blender")
        if isinstance(blender_payload, Mapping):
            normalized["blender"] = dict(blender_payload)

        if _looks_like_blender_scene(normalized):
            normalized["render_backend"] = "blender"

        validated.append(normalized)
    return validated


def refine_prompt(
    original_prompt: str,
    feedback: str,
    narration: str = "",
    provider: Literal["openai", "anthropic"] = "openai",
) -> str:
    """
    Refine an image prompt based on user feedback.

    Args:
        original_prompt: The current image prompt
        feedback: User's requested changes
        narration: The scene narration for context
        provider: LLM provider to use

    Returns:
        Refined prompt string
    """
    system_prompt = load_prompt("refiner_system")

    narration_context = f"\nScene narration (for context): {narration}\n" if narration else ""

    user_prompt = f"""Original prompt: {original_prompt}
{narration_context}
User feedback: {feedback}

Please provide the refined prompt."""

    if provider == "openai":
        client = _get_openai_client()
        response = client.responses.create(
            model="gpt-5.1",
            instructions=system_prompt,
            input=user_prompt,
            temperature=0.7,
        )
        return response.output_text.strip()

    elif provider == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        text_block = next((b for b in response.content if hasattr(b, 'text')), None)
        if not text_block:
            raise ValueError("No text response from model")
        return text_block.text.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")


def refine_narration(
    original_narration: str,
    feedback: str,
    provider: Literal["openai", "anthropic"] = "openai",
) -> str:
    """
    Refine a scene narration based on user feedback.

    Args:
        original_narration: The current narration text
        feedback: User's requested changes
        provider: LLM provider to use

    Returns:
        Refined narration string
    """
    system_prompt = load_prompt("refiner_narration_system")

    user_prompt = f"""Original narration: {original_narration}

User feedback: {feedback}

Please provide the refined narration."""

    if provider == "openai":
        client = _get_openai_client()
        response = client.responses.create(
            model="gpt-5.1",
            instructions=system_prompt,
            input=user_prompt,
            temperature=0.7,
        )
        return response.output_text.strip()

    elif provider == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        text_block = next((b for b in response.content if hasattr(b, 'text')), None)
        if not text_block:
            raise ValueError("No text response from model")
        return text_block.text.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")
