"""Image generation using Replicate API (Qwen Image)."""

from pathlib import Path

import replicate
import requests

from core.rate_limiter import image_limiter


# Style suffix appended to every image prompt - this is the only context the image model gets
STYLE_SUFFIX = ", patient-friendly medical education video, warm and reassuring aesthetic, premium healthcare feel, soft lighting, modern and approachable, never clinical or scary, 16:9 aspect ratio"

def generate_image(
    prompt: str,
    output_path: str | Path,
    model: str = "qwen/qwen-image-2512",
    apply_style: bool = True,
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style suffix for consistent aesthetic
    full_prompt = f"{prompt}{STYLE_SUFFIX}" if apply_style else prompt

    # Run the model with rate limiting and retry
    def _call_replicate():
        return replicate.run(
            model,
            input={
                "prompt": full_prompt,
                "aspect_ratio": "16:9",
                "output_format": "png",
                "go_fast": True,
                "guidance": 4,
                "num_inference_steps": 40,
            }
        )

    output = image_limiter.call_with_retry(_call_replicate)

    # Handle different output formats from Replicate
    if isinstance(output, list):
        image_url = output[0]
    elif hasattr(output, 'url'):
        image_url = output.url
    else:
        image_url = str(output)

    # Download and save the image
    response = requests.get(image_url, timeout=120)
    response.raise_for_status()

    output_path.write_bytes(response.content)

    return output_path


def edit_image(
    prompt: str,
    input_image_path: str | Path,
    output_path: str | Path,
    model: str = "qwen/qwen-image-edit-2511",
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
        with open(input_image_path, "rb") as f:
            return replicate.run(
                model,
                input={
                    "prompt": prompt,
                    "image": [f],  # Model expects array
                    "aspect_ratio": "16:9",
                    "output_format": "png",
                    "go_fast": True,
                }
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
    response = requests.get(image_url, timeout=120)
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
    scene_id = scene["id"]
    prompt = scene["visual_prompt"]

    output_path = project_dir / "images" / f"scene_{scene_id:03d}.png"

    return generate_image(prompt, output_path)
