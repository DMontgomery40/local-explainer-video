"""Image generation using Replicate API (Qwen Image)."""

from pathlib import Path
import traceback
import sys

import replicate
from replicate import Client
import requests

from core.rate_limiter import image_limiter


def _log(msg):
    """Debug logging to stderr (visible in Streamlit terminal)."""
    print(f"[IMAGE_GEN] {msg}", file=sys.stderr, flush=True)


# Replicate client with timeout to prevent indefinite hangs
# 120s timeout covers both cold-start queue delays and generation time
_replicate_client = None


def _get_replicate_client() -> Client:
    """Get or create the Replicate client with timeout."""
    global _replicate_client
    if _replicate_client is None:
        _replicate_client = Client(timeout=120)  # 2 minute timeout
        _log("Created Replicate client with 120s timeout")
    return _replicate_client


# Style suffix appended to every image prompt - this is the only context the image model gets
STYLE_SUFFIX = ", patient-friendly medical education video, warm and reassuring aesthetic, premium healthcare feel, soft lighting, modern and approachable, never clinical or scary, 16:9 aspect ratio"

def generate_image(
    prompt: str,
    output_path: str | Path,
    model: str = "qwen/qwen-image-2512",
    apply_style: bool = True,
    seed: int | None = None,
) -> Path:
    """
    Generate an image using Qwen Image 2512 model.

    $0.02/image, ~7s, strongest text rendering available.

    Args:
        prompt: The image generation prompt
        output_path: Where to save the generated image
        model: Replicate model identifier
        apply_style: Whether to append the style suffix

    Returns:
        Path to the saved image
    """
    _log(f"generate_image() called")
    _log(f"  output_path: {output_path}")
    _log(f"  model: {model}")
    _log(f"  prompt (first 100 chars): {prompt[:100]}...")

    output_path = Path(output_path)
    _log(f"  Creating parent dir: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style suffix for consistent aesthetic
    full_prompt = f"{prompt}{STYLE_SUFFIX}" if apply_style else prompt
    _log(f"  Full prompt length: {len(full_prompt)} chars")

    # Run the model with rate limiting and retry
    def _call_replicate():
        client = _get_replicate_client()
        _log(f"  Calling client.run() with model: {model} (120s timeout)")
        inputs: dict = {
            "prompt": full_prompt,
            "aspect_ratio": "16:9",
            "output_format": "png",
            "go_fast": True,
            "guidance": 4,
            "num_inference_steps": 40,
        }
        if seed is not None:
            inputs["seed"] = int(seed)
        return client.run(
            model,
            input=inputs,
        )

    try:
        _log(f"  Starting API call with retry...")
        output = image_limiter.call_with_retry(_call_replicate)
        _log(f"  API call succeeded, output type: {type(output)}")
        _log(f"  Output value: {output}")
    except Exception as e:
        _log(f"  API CALL FAILED: {type(e).__name__}: {e}")
        _log(f"  Traceback:\n{traceback.format_exc()}")
        raise

    # Handle different output formats from Replicate
    if isinstance(output, list):
        image_url = output[0]
        _log(f"  Output was list, first element: {image_url}")
    elif hasattr(output, 'url'):
        image_url = output.url
        _log(f"  Output had .url attribute: {image_url}")
    else:
        image_url = str(output)
        _log(f"  Output stringified: {image_url}")

    # Download and save the image
    _log(f"  Downloading image from URL...")
    try:
        response = requests.get(image_url, timeout=(5, 60))  # (connect, read)
        _log(f"  Download response status: {response.status_code}")
        response.raise_for_status()
        _log(f"  Downloaded {len(response.content)} bytes")
    except Exception as e:
        _log(f"  DOWNLOAD FAILED: {type(e).__name__}: {e}")
        _log(f"  Traceback:\n{traceback.format_exc()}")
        raise

    try:
        output_path.write_bytes(response.content)
        _log(f"  Saved to: {output_path}")
        _log(f"  File exists: {output_path.exists()}, size: {output_path.stat().st_size if output_path.exists() else 'N/A'}")
    except Exception as e:
        _log(f"  FILE WRITE FAILED: {type(e).__name__}: {e}")
        _log(f"  Traceback:\n{traceback.format_exc()}")
        raise

    return output_path


def edit_image(
    prompt: str,
    input_image_path: str | Path,
    output_path: str | Path,
    model: str = "qwen/qwen-image-edit-2511",
    seed: int | None = None,
) -> Path:
    """
    Edit an existing image using Qwen Image Edit model.

    $0.03/edit, ~4.5s, preserves style while making targeted changes.

    Args:
        prompt: Text instruction describing the edit
        input_image_path: Path to the image to edit
        output_path: Where to save the edited image

    Returns:
        Path to the edited image
    """
    input_image_path = Path(input_image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Upload the input image - model expects image as file handle
    # Note: We need to define the call inside the retry loop to re-open the file on retry
    def _call_replicate_edit():
        client = _get_replicate_client()
        with open(input_image_path, "rb") as f:
            inputs: dict = {
                "prompt": prompt,
                "image": [f],  # Model expects array
                "aspect_ratio": "16:9",
                "output_format": "png",
                "go_fast": True,
            }
            if seed is not None:
                inputs["seed"] = int(seed)
            return client.run(
                model,
                input=inputs,
            )

    output = image_limiter.call_with_retry(_call_replicate_edit)

    # Handle output
    if isinstance(output, list):
        image_url = output[0]
    elif hasattr(output, 'url'):
        image_url = output.url
    else:
        image_url = str(output)

    # Download and save
    response = requests.get(image_url, timeout=(5, 60))  # (connect, read)
    response.raise_for_status()
    output_path.write_bytes(response.content)

    return output_path


def generate_scene_image(
    scene: dict,
    project_dir: Path,
) -> Path:
    """
    Generate an image for a specific scene.

    Args:
        scene: Scene dictionary with 'id' and 'visual_prompt'
        project_dir: Project directory for saving assets

    Returns:
        Path to the generated image
    """
    _log(f"=== generate_scene_image() START ===")
    _log(f"  scene keys: {list(scene.keys())}")
    _log(f"  project_dir: {project_dir}")

    scene_id = scene.get("id")
    _log(f"  scene_id: {scene_id}")

    prompt = scene.get("visual_prompt")
    if not prompt:
        _log(f"  ERROR: No visual_prompt in scene!")
        raise ValueError(f"Scene {scene_id} has no visual_prompt")

    _log(f"  visual_prompt length: {len(prompt)}")

    output_path = project_dir / "images" / f"scene_{scene_id:03d}.png"
    _log(f"  output_path: {output_path}")

    try:
        result = generate_image(prompt, output_path)
        _log(f"=== generate_scene_image() SUCCESS: {result} ===")
        return result
    except Exception as e:
        _log(f"=== generate_scene_image() FAILED: {type(e).__name__}: {e} ===")
        raise
