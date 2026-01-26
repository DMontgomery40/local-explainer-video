"""Image generation/editing using Replicate (Qwen Image) and DashScope (Alibaba Model Studio)."""

from pathlib import Path
import base64
import os
import traceback
import sys
from contextlib import ExitStack

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

# DashScope (Alibaba Model Studio) endpoints for Qwen image edit models.
# NOTE: Beijing + Singapore have separate API keys and endpoints (cross-region calls fail).
_DASHSCOPE_ENDPOINT_SINGAPORE = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
_DASHSCOPE_ENDPOINT_BEIJING = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def _dashscope_endpoint() -> str:
    override = (os.getenv("DASHSCOPE_ENDPOINT") or "").strip()
    if override:
        return override
    region = (os.getenv("DASHSCOPE_REGION") or "").strip().upper()
    if region in {"BEIJING", "BJ", "CN"}:
        return _DASHSCOPE_ENDPOINT_BEIJING
    # Default to the "intl" endpoint (Singapore) because this repo is commonly run outside CN.
    return _DASHSCOPE_ENDPOINT_SINGAPORE


def _dashscope_api_key() -> str:
    key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        raise ValueError("DASHSCOPE_API_KEY is not set (required for DashScope qwen-image-edit-* models).")
    return key


def _data_uri_for_image(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "gif": "image/gif",
    }.get(ext, "application/octet-stream")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _infer_dashscope_size(path: Path) -> str | None:
    """Infer an explicit DashScope size from an image, if within allowed bounds."""
    try:
        from PIL import Image  # Pillow is already a project dependency

        with Image.open(path) as im:
            w, h = im.size
        if 512 <= int(w) <= 2048 and 512 <= int(h) <= 2048:
            return f"{int(w)}*{int(h)}"
    except Exception:
        return None
    return None


def _extract_dashscope_image_urls(payload: dict) -> list[str]:
    """
    DashScope response example:
      { "output": { "choices": [ { "message": { "content": [ {"image": "https://...png"}, ... ]}}]}}
    """
    urls: list[str] = []
    output = payload.get("output")
    if not isinstance(output, dict):
        return urls
    choices = output.get("choices")
    if not isinstance(choices, list):
        return urls
    for c in choices:
        if not isinstance(c, dict):
            continue
        msg = c.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            img = part.get("image")
            if isinstance(img, str) and img.strip():
                urls.append(img.strip())
    return urls


def _default_image_edit_model() -> str:
    """
    Default selection:
    - If IMAGE_EDIT_MODEL is set, use it verbatim (e.g., 'qwen-image-edit-max' or 'qwen/qwen-image-edit-2511')
    - Else, if DASHSCOPE_API_KEY is set, default to DashScope 'qwen-image-edit-max'
    - Else, default to Replicate 'qwen/qwen-image-edit-2511'
    """
    provider = (os.getenv("IMAGE_EDIT_PROVIDER") or "").strip().lower()
    env_model = (os.getenv("IMAGE_EDIT_MODEL") or "").strip()
    if provider in {"replicate", "rep"}:
        return env_model or "qwen/qwen-image-edit-2511"
    if provider in {"dashscope", "alibaba", "modelstudio"}:
        return env_model or "qwen-image-edit-max"
    if env_model:
        return env_model
    if (os.getenv("DASHSCOPE_API_KEY") or "").strip():
        return "qwen-image-edit-max"
    return "qwen/qwen-image-edit-2511"


def _edit_image_replicate(
    *,
    prompt: str,
    input_image_paths: list[Path],
    output_path: Path,
    model: str,
    seed: int | None,
) -> Path:
    # Note: We need to define the call inside the retry loop to re-open the files on retry
    def _call_replicate_edit():
        client = _get_replicate_client()
        with ExitStack() as stack:
            files = [stack.enter_context(open(p, "rb")) for p in input_image_paths]
            inputs: dict = {
                "prompt": prompt,
                "image": files,  # Model expects an array
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
    elif hasattr(output, "url"):
        image_url = output.url
    else:
        image_url = str(output)

    # Download and save
    response = requests.get(image_url, timeout=(5, 60))  # (connect, read)
    response.raise_for_status()
    output_path.write_bytes(response.content)

    return output_path


def _edit_image_dashscope(
    *,
    prompt: str,
    input_image_paths: list[Path],
    output_path: Path,
    model: str,
    n: int,
    size: str | None,
    prompt_extend: bool,
    negative_prompt: str,
    watermark: bool,
    seed: int | None,
) -> Path:
    if not (1 <= int(n) <= 6):
        raise ValueError("DashScope qwen-image-edit-max/qwen-image-edit-plus supports n in [1, 6].")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    endpoint = _dashscope_endpoint()
    api_key = _dashscope_api_key()

    inferred_size = size or _infer_dashscope_size(input_image_paths[-1])
    params: dict = {
        "n": int(n),
        "negative_prompt": negative_prompt if isinstance(negative_prompt, str) else " ",
        "prompt_extend": bool(prompt_extend),
        "watermark": bool(watermark),
    }
    if inferred_size:
        params["size"] = inferred_size
    if seed is not None:
        # DashScope supports seed 0-2147483647 for reproducible edits
        params["seed"] = int(seed) % 2147483648

    content = [{"image": _data_uri_for_image(p)} for p in input_image_paths]
    content.append({"text": prompt})

    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": params,
    }

    def _call_dashscope() -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        _log(f"  DashScope POST model={model} n={params.get('n')} size={params.get('size', '(default)')}")
        resp = requests.post(endpoint, headers=headers, json=body, timeout=(10, 240))
        _log(f"  DashScope response status: {resp.status_code}")
        try:
            payload = resp.json()
        except Exception:
            # Bubble up HTTP errors with some context
            snippet = (resp.text or "")[:500]
            raise RuntimeError(f"DashScope returned non-JSON (HTTP {resp.status_code}): {snippet}") from None

        if resp.status_code != 200:
            msg = payload.get("message") if isinstance(payload, dict) else None
            code = payload.get("code") if isinstance(payload, dict) else None
            raise RuntimeError(f"DashScope image edit failed (HTTP {resp.status_code}, code={code}): {msg}")

        if isinstance(payload, dict) and str(payload.get("code") or "").strip():
            raise RuntimeError(f"DashScope image edit failed (code={payload.get('code')}): {payload.get('message')}")

        return payload if isinstance(payload, dict) else {}

    payload = image_limiter.call_with_retry(_call_dashscope)
    urls = _extract_dashscope_image_urls(payload)
    if not urls:
        raise RuntimeError(f"DashScope returned no image URLs. Top-level keys: {list(payload.keys())}")

    # Save first image to output_path; if n>1, also save additional images next to it.
    first_written = False
    for idx, url in enumerate(urls[: int(n)]):
        out = (
            output_path
            if idx == 0
            else output_path.with_name(f"{output_path.stem}__n{idx + 1}{output_path.suffix}")
        )
        r = requests.get(url, timeout=(5, 120))
        r.raise_for_status()
        out.write_bytes(r.content)
        if not first_written:
            first_written = True
    return output_path


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
    input_image_path: str | Path | list[str | Path],
    output_path: str | Path,
    model: str | None = None,
    seed: int | None = None,
    *,
    # DashScope-only controls (ignored by Replicate models)
    n: int = 1,
    size: str | None = None,
    prompt_extend: bool = True,
    negative_prompt: str = " ",
    watermark: bool = False,
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
    # Normalize input images (DashScope supports 1-3 images; Replicate model accepts an array too)
    if isinstance(input_image_path, (list, tuple)):
        input_image_paths = [Path(p) for p in input_image_path]
    else:
        input_image_paths = [Path(input_image_path)]

    chosen_model = (model or "").strip() or _default_image_edit_model()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # DashScope model names look like: qwen-image-edit-max, qwen-image-edit-max-2026-01-16, qwen-image-edit-plus, qwen-image-edit
    if chosen_model.startswith("qwen-image-edit"):
        return _edit_image_dashscope(
            prompt=prompt,
            input_image_paths=input_image_paths,
            output_path=output_path,
            model=chosen_model,
            n=int(n),
            size=size,
            prompt_extend=bool(prompt_extend),
            negative_prompt=negative_prompt,
            watermark=bool(watermark),
            seed=seed,
        )

    # Otherwise assume Replicate model id (e.g., "qwen/qwen-image-edit-2511")
    return _edit_image_replicate(
        prompt=prompt,
        input_image_paths=input_image_paths,
        output_path=output_path,
        model=chosen_model,
        seed=seed,
    )


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
