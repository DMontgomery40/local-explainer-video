"""Image generation/editing using Replicate (Qwen/Imagen) and DashScope."""

from pathlib import Path
import base64
import os
import re
import traceback
import sys
from contextlib import ExitStack

import replicate
from replicate import Client
import requests

from core.rate_limiter import image_limiter

# Target dimensions for all generated/edited images (16:9 aspect ratio)
TARGET_WIDTH = 1664
TARGET_HEIGHT = 928
TARGET_SIZE_DASHSCOPE = f"{TARGET_WIDTH}*{TARGET_HEIGHT}"

# Replicate text-to-image models exposed in this app.
DEFAULT_IMAGE_GEN_MODEL = "qwen/qwen-image-2512"
IMAGEN_4_MODEL = "google/imagen-4"

# Marker Claude can emit in visual_prompt to force EEG map img2img conditioning.
EEG_10_20_REF_MARKER = "[[USE_EEG_10_20_REF]]"
EEG_10_20_REFERENCE_IMAGE = Path(__file__).parent.parent / "prompts" / "qEEG-site-mapping-reference.JPG"


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


def _ensure_png(path: Path) -> Path:
    """Convert image to PNG if it's not already (Replicate sometimes returns WebP)."""
    import subprocess
    result = subprocess.run(["file", str(path)], capture_output=True, text=True)
    if "Web/P" in result.stdout or "RIFF" in result.stdout:
        tmp = path.with_suffix(".tmp.png")
        subprocess.run(["ffmpeg", "-y", "-i", str(path), "-f", "image2", str(tmp)],
                      capture_output=True, check=True)
        tmp.replace(path)
        _log(f"  Converted WebP to PNG: {path}")
    return path


# Style suffix appended to every image prompt - this is the only context the image model gets.
STYLE_SUFFIX = (
    ", patient-friendly medical education video, warm and reassuring aesthetic, premium healthcare feel, "
    "soft lighting, modern and approachable, never clinical or scary, 16:9 aspect ratio"
)

# Optional anatomy reminder for scenes that depict EEG topography/electrodes.
# This is intentionally guidance so composition and camera angle can still vary.
EEG_10_20_GUIDE_SUFFIX = (
    ", if electrodes or brain-map labels are shown, follow standard EEG 10-20 topology as anatomical guidance: "
    "front landmark is nasion and rear landmark is inion; left/right ear references are A1/A2; "
    "front row Fp1 Fp2; next row F7 F3 Fz F4 F8; middle row T3 C3 Cz C4 T4; "
    "posterior row T5 P3 Pz P4 T6; occipital row O1 O2; preserve relative neighborhoods"
)

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
    # Accept both DASHSCOPE_API_KEY and ALIBABA_API_KEY (same platform, different naming)
    key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_API_KEY") or "").strip()
    if not key:
        raise ValueError("DASHSCOPE_API_KEY (or ALIBABA_API_KEY) is not set (required for DashScope qwen-image-edit-* models).")
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


def _normalize_model_slug(model: str | None) -> str:
    """Strip an optional Replicate version suffix (owner/model:version -> owner/model)."""
    return str(model or "").split(":", 1)[0].strip().lower()


def _is_qwen_generation_model(model: str | None) -> bool:
    normalized = _normalize_model_slug(model)
    return normalized.startswith("qwen/qwen-image")


def _strip_eeg_marker(prompt: str) -> tuple[str, bool]:
    text = str(prompt or "")
    has_marker = EEG_10_20_REF_MARKER in text
    if not has_marker:
        return text, False
    cleaned = text.replace(EEG_10_20_REF_MARKER, " ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, True


def _prompt_has_eeg_topology_keywords(prompt: str) -> bool:
    """Fallback detector when marker is missing."""
    text = str(prompt or "").lower()
    if not text:
        return False
    keyword_patterns = [
        r"\beeg\b",
        r"\b10-20\b",
        r"\belectrode",
        r"\btopograph",
        r"\bbrain map\b",
        r"\bconnectivity\b",
        r"\bcoherence\b",
        r"\bfp1\b|\bfp2\b|\bf7\b|\bf3\b|\bfz\b|\bf4\b|\bf8\b",
        r"\bt3\b|\bc3\b|\bcz\b|\bc4\b|\bt4\b|\bt5\b|\bp3\b|\bpz\b|\bp4\b|\bt6\b",
        r"\bo1\b|\bo2\b|\ba1\b|\ba2\b",
    ]
    return any(re.search(p, text) for p in keyword_patterns)


def _is_text_dense_prompt(prompt: str) -> bool:
    text = str(prompt or "")
    if not text:
        return False
    quote_count = text.count('"')
    numeric_tokens = re.findall(r"\b\d+(?:\.\d+)?(?:ms|hz|%|µv|uv|sec|s)?\b", text, flags=re.IGNORECASE)
    lowered = text.lower()
    text_keywords = [
        "title",
        "subtitle",
        "label",
        "caption",
        "timeline",
        "chart",
        "infographic",
        "session",
        "ms",
        "hz",
        "µv",
        "uv",
        "%",
    ]
    keyword_hits = sum(1 for k in text_keywords if k in lowered)
    return quote_count >= 4 or len(numeric_tokens) >= 4 or keyword_hits >= 3


def default_qwen_guidance_for_prompt(prompt: str) -> float:
    """
    Adaptive guidance policy:
    - text-dense technical prompts: 8.0
    - conceptual/brain prompts: 10.0
    """
    return 8.0 if _is_text_dense_prompt(prompt) else 10.0


def _default_qwen_negative_prompt(prompt: str) -> str:
    base = [
        "blurry",
        "low resolution",
        "jpeg artifacts",
        "noisy image",
        "oversaturated",
        "overprocessed",
        "watermark",
        "logo",
        "signature",
        "border",
        "cropped text",
        "cut-off text",
        "illegible text",
        "gibberish text",
        "misspelled words",
        "duplicated words",
        "random characters",
    ]

    text_dense = [
        "incorrect numbers",
        "wrong units",
        "inconsistent labels",
        "extra unlabeled numbers",
        "malformed typography",
        "overlapping text blocks",
        "warped letters",
        "mirrored text",
    ]
    concept = [
        "anatomical distortions",
        "deformed hands",
        "extra limbs",
        "uncanny face",
        "plastic skin",
    ]
    selected = base + (text_dense if _is_text_dense_prompt(prompt) else concept)
    # Keep prompt concise to avoid over-constraining/attenuation.
    merged = ", ".join(dict.fromkeys(selected))
    if len(merged) > 450:
        merged = merged[:450].rstrip(", ")
    return merged or " "


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
    if (os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_API_KEY") or "").strip():
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
                "go_fast": False,
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

    # Ensure PNG format (Replicate may return WebP despite output_format: png)
    _ensure_png(output_path)

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

    # Always use target dimensions for consistent output (size arg can override)
    inferred_size = size or TARGET_SIZE_DASHSCOPE
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


def _build_generate_inputs(
    model: str,
    full_prompt: str,
    seed: int | None,
    *,
    guidance: float | None,
    negative_prompt: str | None,
) -> dict:
    """
    Build model-specific inputs for Replicate image generation.
    """
    normalized = _normalize_model_slug(model)
    if normalized == IMAGEN_4_MODEL:
        if seed is not None:
            _log("  seed provided but ignored for google/imagen-4 (not in schema)")
        return {
            "prompt": full_prompt,
            "aspect_ratio": "16:9",
            "output_format": "png",
            "safety_filter_level": "block_only_high",
        }

    resolved_guidance = float(guidance) if guidance is not None else default_qwen_guidance_for_prompt(full_prompt)
    resolved_negative = str(negative_prompt) if isinstance(negative_prompt, str) else _default_qwen_negative_prompt(full_prompt)
    if not resolved_negative.strip():
        # Recommended by Qwen examples when not using a negative prompt.
        resolved_negative = " "

    inputs: dict = {
        "prompt": full_prompt,
        "aspect_ratio": "16:9",
        "output_format": "png",
        "go_fast": False,
        "guidance": max(0.0, min(10.0, resolved_guidance)),
        "num_inference_steps": 50,
        "negative_prompt": resolved_negative,
    }
    if seed is not None:
        inputs["seed"] = int(seed)
    return inputs


def generate_image(
    prompt: str,
    output_path: str | Path,
    model: str = DEFAULT_IMAGE_GEN_MODEL,
    apply_style: bool = True,
    use_eeg_10_20_guide: bool = False,
    seed: int | None = None,
    *,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    reference_image_path: str | Path | None = None,
    reference_strength: float | None = None,
) -> Path:
    """
    Generate an image using a Replicate text-to-image model.
    """
    _log("generate_image() called")
    _log(f"  output_path: {output_path}")
    _log(f"  model: {model}")
    _log(f"  prompt (first 100 chars): {prompt[:100]}...")
    _log(f"  guidance override: {guidance}")
    _log(f"  has reference image: {bool(reference_image_path)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply style suffix for consistent aesthetic.
    full_prompt = f"{prompt}{STYLE_SUFFIX}" if apply_style else prompt
    if use_eeg_10_20_guide:
        full_prompt = f"{full_prompt}{EEG_10_20_GUIDE_SUFFIX}"
        _log("  Added EEG 10-20 text guidance suffix")
    _log(f"  Full prompt length: {len(full_prompt)} chars")

    normalized = _normalize_model_slug(model)

    def _call_replicate():
        client = _get_replicate_client()
        with ExitStack() as stack:
            _log(f"  Calling client.run() with model: {model} (120s timeout)")
            inputs = _build_generate_inputs(
                model,
                full_prompt,
                seed,
                guidance=guidance,
                negative_prompt=negative_prompt,
            )
            if _is_qwen_generation_model(model) and reference_image_path:
                ref_path = Path(reference_image_path)
                if ref_path.exists():
                    inputs["image"] = stack.enter_context(open(ref_path, "rb"))
                    # Replicate schema: strength in [0, 1], 1.0 is full destruction.
                    resolved_strength = 1.0 if reference_strength is None else float(reference_strength)
                    inputs["strength"] = max(0.0, min(1.0, resolved_strength))
                    _log(f"  Using img2img reference: {ref_path.name} (strength={inputs['strength']})")
                else:
                    _log(f"  Reference image missing, skipping img2img: {ref_path}")
            elif normalized == IMAGEN_4_MODEL and reference_image_path:
                _log("  Reference image requested but ignored for Imagen 4 (text-only schema)")
            return client.run(
                model,
                input=inputs,
            )

    try:
        output = image_limiter.call_with_retry(_call_replicate)
        _log(f"  API call succeeded, output type: {type(output)}")
    except Exception as e:
        _log(f"  API CALL FAILED: {type(e).__name__}: {e}")
        _log(f"  Traceback:\n{traceback.format_exc()}")
        raise

    # Handle different output formats from Replicate.
    if isinstance(output, list):
        image_url = output[0]
    elif hasattr(output, "url"):
        image_url = output.url
    else:
        image_url = str(output)

    try:
        response = requests.get(image_url, timeout=(5, 60))  # (connect, read)
        response.raise_for_status()
    except Exception as e:
        _log(f"  DOWNLOAD FAILED: {type(e).__name__}: {e}")
        _log(f"  Traceback:\n{traceback.format_exc()}")
        raise

    try:
        output_path.write_bytes(response.content)
    except Exception as e:
        _log(f"  FILE WRITE FAILED: {type(e).__name__}: {e}")
        _log(f"  Traceback:\n{traceback.format_exc()}")
        raise

    # Ensure PNG format (Replicate may return WebP despite output_format: png).
    _ensure_png(output_path)
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
    model: str = DEFAULT_IMAGE_GEN_MODEL,
    use_eeg_10_20_guide: bool = False,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    force_use_eeg_ref: bool | None = None,
) -> Path:
    """
    Generate an image for a specific scene.

    Args:
        scene: Scene dictionary with 'id' and 'visual_prompt'
        project_dir: Project directory for saving assets
        model: Replicate image model id
        use_eeg_10_20_guide: Whether to append 10-20 textual guidance
        guidance: Optional guidance override
        negative_prompt: Optional negative prompt override
        force_use_eeg_ref: Optional explicit toggle for EEG reference img2img

    Returns:
        Path to the generated image
    """
    _log("=== generate_scene_image() START ===")
    _log(f"  scene keys: {list(scene.keys())}")
    _log(f"  project_dir: {project_dir}")

    scene_id = scene.get("id")
    _log(f"  scene_id: {scene_id}")

    prompt_raw = scene.get("visual_prompt")
    if not prompt_raw:
        _log("  ERROR: No visual_prompt in scene!")
        raise ValueError(f"Scene {scene_id} has no visual_prompt")
    prompt, has_marker = _strip_eeg_marker(str(prompt_raw))

    _log(f"  visual_prompt length: {len(prompt)}")
    _log(f"  image model: {model}")
    _log(f"  EEG 10-20 textual guide: {use_eeg_10_20_guide}")
    _log(f"  marker present: {has_marker}")

    output_path = project_dir / "images" / f"scene_{scene_id:03d}.png"
    _log(f"  output_path: {output_path}")

    use_reference = False
    if _is_qwen_generation_model(model):
        if force_use_eeg_ref is not None:
            use_reference = bool(force_use_eeg_ref)
        else:
            # Primary trigger is explicit marker from Claude. Fallback helps with legacy prompts.
            use_reference = has_marker or _prompt_has_eeg_topology_keywords(prompt)

    ref_path: Path | None = None
    ref_strength: float | None = None
    if use_reference:
        ref_path = EEG_10_20_REFERENCE_IMAGE
        ref_strength = 1.0

    guidance_value = float(guidance) if guidance is not None else default_qwen_guidance_for_prompt(prompt)
    negative_value = negative_prompt if isinstance(negative_prompt, str) else _default_qwen_negative_prompt(prompt)

    try:
        result = generate_image(
            prompt,
            output_path,
            model=model,
            use_eeg_10_20_guide=use_eeg_10_20_guide,
            guidance=guidance_value,
            negative_prompt=negative_value,
            reference_image_path=ref_path,
            reference_strength=ref_strength,
        )
        _log(f"=== generate_scene_image() SUCCESS: {result} ===")
        return result
    except Exception as e:
        _log(f"=== generate_scene_image() FAILED: {type(e).__name__}: {e} ===")
        raise
