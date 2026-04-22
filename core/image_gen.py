"""Still-image generation and editing helpers.

Prompt-bearing still scenes now prefer the local Codex CLI path using native
`gpt-image-2` generation. Deterministic Remotion still rendering remains
available as a compatibility fallback for scenes that do not carry a prompt.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from core.scene_modes import scene_is_cathode_motion

TARGET_WIDTH = 1664
TARGET_HEIGHT = 928
TARGET_ASPECT_RATIO = "16:9"
TARGET_SIZE_DASHSCOPE = f"{TARGET_WIDTH}*{TARGET_HEIGHT}"
DEFAULT_IMAGE_GEN_MODEL = "gpt-image-2"

_DASHSCOPE_ENDPOINT_SINGAPORE = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
_DASHSCOPE_ENDPOINT_BEIJING = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
_replicate_client = None


def _log(msg: str) -> None:
    print(f"[IMAGE_GEN] {msg}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _codex_binary() -> str:
    return (os.getenv("CODEX_BINARY") or "codex").strip()


def _codex_cli_available() -> bool:
    binary = _codex_binary()
    if not binary:
        return False
    if os.sep in binary:
        return Path(binary).expanduser().exists()
    return shutil.which(binary) is not None


def _codex_runner_model() -> str | None:
    value = (
        os.getenv("LOCAL_EXPLAINER_CODEX_IMAGE_RUNNER_MODEL")
        or os.getenv("CODEX_IMAGE_RUNNER_MODEL")
        or ""
    ).strip()
    return value or None


def _image_log_dir(output_path: Path) -> Path:
    if output_path.parent.name == "images":
        project_dir = output_path.parent.parent
    else:
        project_dir = output_path.parent
    log_dir = project_dir / "image_generation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _ensure_png(path: Path) -> Path:
    """Convert image to PNG if the provider returned a different format."""
    result = subprocess.run(["file", str(path)], capture_output=True, text=True, check=False)
    if "Web/P" in result.stdout or "RIFF" in result.stdout:
        tmp = path.with_suffix(".tmp.png")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-f", "image2", str(tmp)],
            capture_output=True,
            check=True,
        )
        tmp.replace(path)
        _log(f"Converted WebP to PNG: {path}")
    return path


def _normalize_image_to_target(path: Path) -> tuple[int, int]:
    from PIL import Image, ImageFilter, ImageOps

    with Image.open(path) as img:
        source = img.convert("RGB")
        original_size = source.size

        if original_size == (TARGET_WIDTH, TARGET_HEIGHT):
            return original_size

        background = ImageOps.fit(
            source,
            (TARGET_WIDTH, TARGET_HEIGHT),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        background = background.filter(ImageFilter.GaussianBlur(radius=28))
        background = Image.blend(
            background,
            Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0)),
            0.32,
        )

        foreground = ImageOps.contain(
            source,
            (TARGET_WIDTH, TARGET_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )
        offset = (
            (TARGET_WIDTH - foreground.width) // 2,
            (TARGET_HEIGHT - foreground.height) // 2,
        )
        background.paste(foreground, offset)
        background.save(path, format="PNG", optimize=True)
        return original_size


def _render_scene_still(*, family: str, props: dict[str, Any], output_path: Path) -> Path:
    from .remotion_bridge import render_scene_still

    return render_scene_still(
        family=family,
        props=props,
        output_path=output_path.with_suffix(".png"),
    )


def build_codex_image_prompt(
    *,
    prompt: str,
    output_path: Path,
    image_model: str = DEFAULT_IMAGE_GEN_MODEL,
    title: str = "",
) -> str:
    trimmed_title = str(title or "").strip()
    title_line = f"Scene title: {trimmed_title}\n" if trimmed_title else ""
    return (
        f"Work in {_repo_root()}.\n"
        "Use only Codex's built-in native image generation capability in this session.\n"
        "Do not use skill scripts, wrappers, Python API clients, or any API-key-based image generation workflow.\n"
        "Do not inspect `~/.codex/skills`, `.env` files, config files, or search the filesystem for `OPENAI_API_KEY` or any other secret.\n"
        "If the built-in native image generation capability is unavailable, stop immediately and fail.\n\n"
        f"{title_line}"
        "Generate exactly one still PNG from the following prompt.\n"
        f"- Image model: {image_model}\n"
        f"- Fixed render constraints: landscape {TARGET_ASPECT_RATIO}, widescreen slide, target frame {TARGET_WIDTH}x{TARGET_HEIGHT}, no square composition, no portrait composition, keep all text and important graphics fully visible inside safe margins.\n"
        "- Use the existing prompt text as the core content prompt. Do not otherwise rewrite, summarize, or refine it.\n"
        "- Any quoted on-screen text or branded term must be rendered exactly and case-sensitively.\n"
        f"- Copy the generated PNG to {output_path}.\n"
        "- Do not modify any other repo files.\n"
        f"- After finishing, verify {output_path} exists and report its file size and dimensions.\n\n"
        "Prompt:\n"
        f"{prompt.strip()}\n"
    )


def _run_codex_exec_image(
    *,
    prompt: str,
    output_path: Path,
    runner_model: str | None = None,
) -> Path:
    if not _codex_cli_available():
        raise RuntimeError(
            "Local Codex CLI is not available. Install/configure `codex` before generating still images."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = _image_log_dir(output_path)
    stem = output_path.stem
    jsonl_path = log_dir / f"{stem}.codex.jsonl"
    final_message_path = log_dir / f"{stem}.final.txt"

    cmd = [
        _codex_binary(),
        "exec",
        "--json",
        "--ignore-user-config",
        "-C",
        str(_repo_root()),
        "-s",
        "danger-full-access",
        "-c",
        'approval_policy="never"',
        "-o",
        str(final_message_path),
    ]
    resolved_runner_model = runner_model or _codex_runner_model()
    if resolved_runner_model:
        cmd.extend(["-m", resolved_runner_model])
    cmd.append("-")

    _log(f"Calling local Codex image generation for {output_path.name}")
    with jsonl_path.open("w", encoding="utf-8") as stdout_handle:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"codex exec failed while generating {output_path.name}; inspect {jsonl_path}"
        )
    if not output_path.exists():
        raise RuntimeError(
            f"codex exec finished without writing {output_path}; inspect {jsonl_path} and {final_message_path}"
        )

    _ensure_png(output_path)
    _normalize_image_to_target(output_path)
    return output_path


def generate_image(
    prompt: str,
    output_path: str | Path,
    model: str = DEFAULT_IMAGE_GEN_MODEL,
    *,
    title: str = "",
    runner_model: str | None = None,
    **_: Any,
) -> Path:
    """Generate a still image through local Codex native image generation."""
    output_path = Path(output_path)
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Image generation requires a non-empty prompt")

    codex_prompt = build_codex_image_prompt(
        prompt=prompt,
        output_path=output_path,
        image_model=str(model or DEFAULT_IMAGE_GEN_MODEL).strip() or DEFAULT_IMAGE_GEN_MODEL,
        title=title,
    )
    return _run_codex_exec_image(
        prompt=codex_prompt,
        output_path=output_path,
        runner_model=runner_model,
    )


def _get_replicate_client():
    global _replicate_client
    if _replicate_client is None:
        from replicate import Client

        _replicate_client = Client(timeout=120)
    return _replicate_client


def _dashscope_endpoint() -> str:
    override = (os.getenv("DASHSCOPE_ENDPOINT") or "").strip()
    if override:
        return override
    region = (os.getenv("DASHSCOPE_REGION") or "").strip().upper()
    if region in {"BEIJING", "BJ", "CN"}:
        return _DASHSCOPE_ENDPOINT_BEIJING
    return _DASHSCOPE_ENDPOINT_SINGAPORE


def _dashscope_api_key() -> str:
    key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "DASHSCOPE_API_KEY (or ALIBABA_API_KEY) is not set (required for DashScope qwen-image-edit-* models)."
        )
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


def _extract_dashscope_image_urls(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    output = payload.get("output")
    if not isinstance(output, dict):
        return urls
    choices = output.get("choices")
    if not isinstance(choices, list):
        return urls
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            image_url = part.get("image")
            if isinstance(image_url, str) and image_url.strip():
                urls.append(image_url.strip())
    return urls


def _default_image_edit_model() -> str:
    env_model = (os.getenv("IMAGE_EDIT_MODEL") or "").strip()
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
    import requests

    from core.rate_limiter import image_limiter

    def _call_replicate_edit():
        client = _get_replicate_client()
        with ExitStack() as stack:
            files = [stack.enter_context(open(path, "rb")) for path in input_image_paths]
            inputs: dict[str, Any] = {
                "prompt": prompt,
                "image": files,
                "aspect_ratio": TARGET_ASPECT_RATIO,
                "output_format": "png",
                "go_fast": False,
            }
            if seed is not None:
                inputs["seed"] = int(seed)
            return client.run(model, input=inputs)

    output = image_limiter.call_with_retry(_call_replicate_edit)
    if isinstance(output, list):
        image_url = output[0]
    elif hasattr(output, "url"):
        image_url = output.url
    else:
        image_url = str(output)

    response = requests.get(image_url, timeout=(5, 60))
    response.raise_for_status()
    output_path.write_bytes(response.content)
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
    import requests

    from core.rate_limiter import image_limiter

    if not 1 <= int(n) <= 6:
        raise ValueError("DashScope qwen-image-edit-max/qwen-image-edit-plus supports n in [1, 6].")

    endpoint = _dashscope_endpoint()
    api_key = _dashscope_api_key()
    params: dict[str, Any] = {
        "n": int(n),
        "negative_prompt": negative_prompt if isinstance(negative_prompt, str) else " ",
        "prompt_extend": bool(prompt_extend),
        "watermark": bool(watermark),
        "size": size or TARGET_SIZE_DASHSCOPE,
    }
    if seed is not None:
        params["seed"] = int(seed) % 2147483648

    content = [{"image": _data_uri_for_image(path)} for path in input_image_paths]
    content.append({"text": prompt})
    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": params,
    }

    def _call_dashscope() -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        response = requests.post(endpoint, headers=headers, json=body, timeout=(10, 240))
        try:
            payload = response.json()
        except Exception:
            snippet = (response.text or "")[:500]
            raise RuntimeError(
                f"DashScope returned non-JSON (HTTP {response.status_code}): {snippet}"
            ) from None

        if response.status_code != 200:
            msg = payload.get("message") if isinstance(payload, dict) else None
            code = payload.get("code") if isinstance(payload, dict) else None
            raise RuntimeError(
                f"DashScope image edit failed (HTTP {response.status_code}, code={code}): {msg}"
            )

        if isinstance(payload, dict) and str(payload.get("code") or "").strip():
            raise RuntimeError(
                f"DashScope image edit failed (code={payload.get('code')}): {payload.get('message')}"
            )
        return payload if isinstance(payload, dict) else {}

    payload = image_limiter.call_with_retry(_call_dashscope)
    urls = _extract_dashscope_image_urls(payload)
    if not urls:
        raise RuntimeError(f"DashScope returned no image URLs. Top-level keys: {list(payload.keys())}")

    first_url = urls[0]
    response = requests.get(first_url, timeout=(5, 120))
    response.raise_for_status()
    output_path.write_bytes(response.content)
    _ensure_png(output_path)
    return output_path


def edit_image(
    prompt: str,
    input_image_path: str | Path | list[str | Path],
    output_path: str | Path,
    model: str | None = None,
    seed: int | None = None,
    *,
    n: int = 1,
    size: str | None = None,
    prompt_extend: bool = True,
    negative_prompt: str = " ",
    watermark: bool = False,
) -> Path:
    """Edit an existing image using DashScope or Replicate Qwen image-edit models."""
    if isinstance(input_image_path, (list, tuple)):
        input_image_paths = [Path(path) for path in input_image_path]
    else:
        input_image_paths = [Path(input_image_path)]

    chosen_model = (model or "").strip() or _default_image_edit_model()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    return _edit_image_replicate(
        prompt=prompt,
        input_image_paths=input_image_paths,
        output_path=output_path,
        model=chosen_model,
        seed=seed,
    )


def generate_scene_image(
    scene: dict[str, Any],
    project_dir: str | Path,
    model: str = DEFAULT_IMAGE_GEN_MODEL,
    **kwargs: Any,
) -> Path:
    """Generate a still image for a scene.

    Prompt-bearing scenes use the local Codex `gpt-image-2` path. Scenes without
    prompts fall back to deterministic Remotion still rendering so older plans
    without `visual_prompt` remain usable.
    """
    if scene_is_cathode_motion(scene):
        raise ValueError(
            f"Scene {scene.get('id', 0)} is a Cathode motion/template scene and should not be generated through the still-image path."
        )

    project_dir = Path(project_dir)
    scene_id = int(scene.get("id", 0))
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_path = images_dir / f"scene_{scene_id:03d}.png"

    visual_prompt = str(scene.get("visual_prompt") or "").strip()
    if visual_prompt:
        result = generate_image(
            visual_prompt,
            output_path,
            model=model or DEFAULT_IMAGE_GEN_MODEL,
            title=str(scene.get("title") or ""),
            runner_model=str(kwargs.get("runner_model") or "").strip() or None,
        )
        scene["image_path"] = str(result)
        return result

    composition = scene.get("composition") if isinstance(scene.get("composition"), dict) else {}
    family = str(composition.get("family") or "narration_slide")
    props = composition.get("props") if isinstance(composition.get("props"), dict) else {}
    if not props:
        props = {
            "headline": str(scene.get("title") or ""),
            "body": "",
        }

    result = _render_scene_still(
        family=family,
        props=props,
        output_path=output_path,
    )
    scene["image_path"] = str(result)
    return result
