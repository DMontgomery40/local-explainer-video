"""LLM-based storyboard generation for qEEG explainer videos."""

import json
import os
import uuid
from pathlib import Path
from typing import Literal

import anthropic
import openai

from core.template_pipeline import (
    SceneTyperError,
    annotate_scenes_with_types,
    use_template_pipeline,
    validate_scenes as validate_typed_scenes,
)

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


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory (cached)."""
    if name not in _PROMPTS:
        prompt_path = Path(__file__).parent.parent / "prompts" / f"{name}.txt"
        _PROMPTS[name] = prompt_path.read_text()
    return _PROMPTS[name]


def _allow_scene_typer_fallback() -> bool:
    raw = str(os.getenv("ALLOW_DOWNSTREAM_SCENE_TYPER_FALLBACK", "true")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _extract_json_payload(raw_text: str) -> dict | list:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.find("```", start)
        text = text[start:end if end != -1 else None].strip()
    elif "```" in text:
        start = text.index("```") + len("```")
        end = text.find("```", start)
        text = text[start:end if end != -1 else None].strip()

    candidates = [text]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            fragment = text[start : end + 1].strip()
            if fragment and fragment not in candidates:
                candidates.append(fragment)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON payload from model response")


def _scenes_have_structured_payloads(scenes: list[dict]) -> bool:
    for scene in scenes:
        if not isinstance(scene, dict):
            return False
        if not scene.get("scene_type"):
            return False
        if not isinstance(scene.get("structured_data"), dict):
            return False
    return True


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
    system_prompt = load_prompt("director_system")
    template_mode = use_template_pipeline()
    output_contract = (
        "Return the storyboard as a JSON object with key `scenes`.\n"
        "Each scene must include: id, title, narration, scene_type, structured_data.\n"
        "Include visual_prompt only when needed (especially atmospheric scenes)."
        if template_mode
        else "Return the storyboard as a JSON array of scenes."
    )

    user_prompt = f"""Create a storyboard for an explainer video based on this qEEG analysis.

CRITICAL: Total narration must be 950-1,100 words (6-7 min video).
Target ~15 scenes, ~50 words average per content scene. Count as you go.

---
{input_text}
---

{output_contract}"""
    if provider == "openai":
        scenes = _generate_with_openai(
            system_prompt,
            user_prompt,
            require_visual_prompt=not template_mode,
        )
    elif provider == "anthropic":
        scenes = _generate_with_anthropic(
            system_prompt,
            user_prompt,
            require_visual_prompt=not template_mode,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if template_mode:
        # Structured-first path: if the director already produced typed scenes,
        # validate and use them directly.
        if _scenes_have_structured_payloads(scenes):
            return validate_typed_scenes(scenes)

        if not _allow_scene_typer_fallback():
            raise ValueError(
                "Template pipeline requires `scene_type + structured_data` from director output. "
                "Downstream scene typer fallback is disabled."
            )

        try:
            typed = annotate_scenes_with_types(
                scenes=scenes,
                input_text=input_text,
                provider=provider,
            )
        except SceneTyperError as exc:
            raise ValueError(f"Downstream scene typer fallback failed: {exc}") from exc
        return validate_typed_scenes(typed)

    return scenes


def _generate_with_openai(
    system_prompt: str,
    user_prompt: str,
    *,
    require_visual_prompt: bool,
) -> list[dict]:
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

    return _validate_scenes(scenes, require_visual_prompt=require_visual_prompt)


def _generate_with_anthropic(
    system_prompt: str,
    user_prompt: str,
    *,
    require_visual_prompt: bool,
) -> list[dict]:
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

    result = _extract_json_payload(content)

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

    return _validate_scenes(scenes, require_visual_prompt=require_visual_prompt)


def _validate_scenes(
    scenes: list[dict],
    *,
    require_visual_prompt: bool,
) -> list[dict]:
    """Validate and normalize scene data."""
    validated = []
    for i, scene in enumerate(scenes):
        narration = scene.get("narration", "").strip()
        visual_prompt = scene.get("visual_prompt", "").strip()

        if not narration:
            raise ValueError(f"Scene {i+1} has empty narration")
        if require_visual_prompt and not visual_prompt:
            raise ValueError(f"Scene {i+1} has empty visual prompt")

        normalized_scene = {
            "id": scene.get("id", i),
            "uid": scene.get("uid", str(uuid.uuid4())[:8]),
            "title": scene.get("title", f"Scene {i + 1}"),
            "narration": narration,
            "visual_prompt": visual_prompt,
            "refinement_history": scene.get("refinement_history", []),
            "image_path": scene.get("image_path"),
            "audio_path": scene.get("audio_path"),
        }
        if scene.get("scene_type") is not None:
            normalized_scene["scene_type"] = scene.get("scene_type")
        if scene.get("structured_data") is not None:
            normalized_scene["structured_data"] = scene.get("structured_data")
        validated.append(normalized_scene)
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
