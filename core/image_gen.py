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
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from core.scene_modes import scene_is_cathode_motion
from core.generation_receipts import paid_bytes, status_code, atomic_bytes, request_digest

TARGET_WIDTH = 1664
TARGET_HEIGHT = 928
TARGET_ASPECT_RATIO = "16:9"
TARGET_SIZE_DASHSCOPE = f"{TARGET_WIDTH}*{TARGET_HEIGHT}"
DEFAULT_IMAGE_GEN_MODEL = "gpt-image-2"

# The one rule that stops prompt text becoming picture text.
#
# A visual prompt ends with art direction — "premium medical infographic, luminous
# and precise". When that trails a corner-label instruction with nothing closing the
# label, the image model reads the whole tail as the caption to paint, and then
# invents a brand emblem to sit beside it because the result reads like a logo
# lockup. Fourteen of fourteen slides in AN_04-08-1986's explainer shipped that way:
# the patient identifier followed by the style sentence, under a different made-up
# crest each time. Nobody ever asked for either. The identifier alone is wanted and
# useful; everything else on that label is a bug.
#
# Stated positively and applied to every image, this holds whatever the planner wrote.
TEXT_DISCIPLINE = (
    "Render as on-screen text only the words this prompt places inside quotation marks. "
    "Words describing style, finish, quality or mood are art direction for how the picture "
    "should look — express them in the artwork and never draw them as letters. "
    "Draw only the logos, emblems, badges, crests or brand marks this prompt explicitly asks "
    "for; invent none."
)

_DASHSCOPE_ENDPOINT_SINGAPORE = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
_DASHSCOPE_ENDPOINT_BEIJING = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
_replicate_client = None


def _log(msg: str) -> None:
    print(f"[IMAGE_GEN] {msg}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


_runtime_scope: ContextVar = ContextVar("codex_runtime_scope", default=None)


@contextmanager
def codex_runtime_scope():
    token = _runtime_scope.set({})
    try:
        yield
    finally:
        _runtime_scope.reset(token)


def resolve_codex_runtime() -> dict[str, str]:
    """Check the configured executable or working PATH candidates per job."""
    scope = _runtime_scope.get()
    if scope is not None and "runtime" in scope:
        return scope["runtime"]
    configured = (os.getenv("CODEX_BINARY") or "").strip()
    candidates = ([str(Path(configured).expanduser())] if configured else
                  [str(Path(entry) / "codex") for entry in os.get_exec_path()])
    failures = []
    for candidate in dict.fromkeys(candidates):
        resolved = shutil.which(candidate)
        if not resolved:
            failures.append(candidate)
            continue
        try:
            result = subprocess.run([resolved, "--version"], capture_output=True,
                                    text=True, timeout=15, check=True)
            version = result.stdout.strip()
            if not version:
                raise RuntimeError("Empty version output")
            runtime = {"path": str(Path(resolved).absolute()), "version": version}
            if scope is not None:
                scope["runtime"] = runtime
            return runtime
        except (OSError, subprocess.SubprocessError, RuntimeError):
            failures.append(candidate)
    if configured:
        raise RuntimeError(f"Configured Codex executable failed its version check: {configured}")
    raise FileNotFoundError("No working Codex executable on PATH")


def _codex_binary() -> str:
    return resolve_codex_runtime()["path"]


def _codex_cli_available() -> bool:
    try:
        resolve_codex_runtime()
        return True
    except FileNotFoundError:
        return False


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


def _target_aspect_ratio(width: int, height: int) -> str:
    if width == 1080 and height == 1920:
        return "9:16"
    if width == TARGET_WIDTH and height == TARGET_HEIGHT:
        return TARGET_ASPECT_RATIO
    return f"{width}:{height}"


def _render_constraint_text(width: int, height: int) -> str:
    aspect = _target_aspect_ratio(width, height)
    if height > width:
        return (
            f"portrait {aspect}, vertical phone-first frame, target frame {width}x{height}, "
            "no square composition, no landscape composition, keep all text and important graphics fully visible inside safe margins"
        )
    return (
        f"landscape {aspect}, widescreen slide, target frame {width}x{height}, "
        "no square composition, no portrait composition, keep all text and important graphics fully visible inside safe margins"
    )


def _normalize_image_to_target(path: Path, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT) -> tuple[int, int]:
    from PIL import Image, ImageFilter, ImageOps

    with Image.open(path) as img:
        source = img.convert("RGB")
        original_size = source.size

        if original_size == (width, height):
            return original_size

        background = ImageOps.fit(
            source,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        background = background.filter(ImageFilter.GaussianBlur(radius=28))
        background = Image.blend(
            background,
            Image.new("RGB", (width, height), (0, 0, 0)),
            0.32,
        )

        foreground = ImageOps.contain(
            source,
            (width, height),
            method=Image.Resampling.LANCZOS,
        )
        offset = (
            (width - foreground.width) // 2,
            (height - foreground.height) // 2,
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
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> str:
    trimmed_title = str(title or "").strip()
    title_line = f"Scene title: {trimmed_title}\n" if trimmed_title else ""
    return (
        f"Work in {_repo_root()}.\n"
        "Use only Codex's built-in native image generation capability in this session.\n"
        "Do not use skill scripts, wrappers, Python API clients, or any API-key-based image generation workflow.\n"
        "Do not inspect `~/.codex/skills`, `.env` files, config files, or search the filesystem for `OPENAI_API_KEY` or any other secret.\n"
        "Do not search `/tmp`, `/var/folders`, Downloads, Desktop, this repo, or any other folder for recent PNGs. Use only the image artifact you generate in this turn.\n"
        "If the built-in native image generation capability is unavailable, stop immediately and fail.\n\n"
        f"{title_line}"
        "Generate exactly one still PNG from the following prompt.\n"
        f"- Image model: {image_model}\n"
        f"- Fixed render constraints: {_render_constraint_text(target_width, target_height)}.\n"
        "- Use the existing prompt text as the core content prompt. Do not otherwise rewrite, summarize, or refine it.\n"
        "- Any quoted on-screen text or branded term must be rendered exactly and case-sensitively.\n"
        f"- {TEXT_DISCIPLINE}\n"
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
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> Path:
    runtime = resolve_codex_runtime()
    # This file belongs to this exact intended invocation, never a canonical PNG.
    identity = {"provider": "codex", "prompt": prompt, "runner_model": runner_model or _codex_runner_model(),
                "width": target_width, "height": target_height}
    raw_path = output_path.parent / ".codex-output" / request_digest(identity) / "raw.png"
    raw_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    dispatch_prompt = prompt.replace(str(output_path), str(raw_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = _image_log_dir(output_path)
    stem = output_path.stem
    jsonl_path = log_dir / f"{stem}.codex.jsonl"
    final_message_path = log_dir / f"{stem}.final.txt"

    # Honor the user's codex config (auth, default model, image tool); running with
    # --ignore-user-config made codex fall back to a model the account rejects.
    cmd = [
        runtime["path"],
        "exec",
        "--json",
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

    def dispatch() -> bytes:
        raw_path.unlink(missing_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as stdout_handle:
            proc = subprocess.run(cmd, input=dispatch_prompt, text=True,
                                  stdout=stdout_handle, stderr=subprocess.STDOUT, check=False)
        if not raw_path.is_file():
            raise RuntimeError(f"Codex returned {proc.returncode} without exact new output; inspect {jsonl_path}")
        return raw_path.read_bytes()

    def recover() -> bytes | None:
        if not raw_path.is_file():
            return None
        import io
        from PIL import Image
        raw = raw_path.read_bytes()
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
        except Exception:
            return None
        return raw

    raw = paid_bytes(identity, dispatch, output_path=output_path, provenance=runtime, recover=recover)
    atomic_bytes(output_path, raw)

    _ensure_png(output_path)
    _normalize_image_to_target(output_path, target_width, target_height)
    return output_path


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _scene_target_dimensions(scene: dict[str, Any], kwargs: dict[str, Any]) -> tuple[int, int]:
    width = _positive_int(kwargs.get("target_width") or scene.get("target_width") or scene.get("render_width"), TARGET_WIDTH)
    height = _positive_int(kwargs.get("target_height") or scene.get("target_height") or scene.get("render_height"), TARGET_HEIGHT)
    orientation = str(
        kwargs.get("orientation")
        or scene.get("orientation")
        or scene.get("render_orientation")
        or scene.get("aspect_ratio")
        or scene.get("render_aspect_ratio")
        or ""
    ).strip().lower()
    if (orientation in {"portrait", "vertical", "9:16"} or "9:16" in orientation) and height <= width:
        return 1080, 1920
    if (orientation in {"landscape", "horizontal", "16:9"} or "16:9" in orientation) and width <= height:
        return TARGET_WIDTH, TARGET_HEIGHT
    return width, height


def _image_quality() -> str:
    return (os.getenv("LOCAL_EXPLAINER_IMAGE_QUALITY") or "high").strip() or "high"


def _generate_image_openai(
    *,
    prompt: str,
    output_path: Path,
    model: str,
    target_width: int,
    target_height: int,
) -> Path:
    """Generate one still via the OpenAI gpt-image API and write it to output_path.

    This mirrors the deterministic, file-based approach used by the sibling cathode
    pipeline (scripts/generate_openai_image.py): call the image API directly and write
    the returned PNG to an exact path. It replaces the previous Codex native-image-tool
    path, which returned images only as in-session artifacts with no filesystem handle
    and so failed non-deterministically during headless renders.
    """
    import base64

    import openai

    output_path.parent.mkdir(parents=True, exist_ok=True)
    portrait = int(target_height) > int(target_width)
    orientation_hint = (
        "Vertical 9:16 portrait composition; keep the subject and any text centered and within mobile safe zones.\n"
        if portrait
        else "Horizontal 16:9 landscape composition.\n"
    )
    full_prompt = f"{orientation_hint}{prompt.strip()}\n\n{TEXT_DISCIPLINE}"
    # gpt-image only accepts a fixed set of sizes; pick by orientation, then normalize to
    # the exact scene target. A non-standard size raises, so we fall back to "auto".
    size = "1024x1536" if portrait else "1536x1024"
    quality = _image_quality()
    # The clinic routes text models through OpenRouter via OPENAI_BASE_URL, but OpenRouter
    # does not serve the image endpoint — image generation must hit the real OpenAI API.
    base_url = (os.getenv("OPENAI_IMAGE_BASE_URL") or "https://api.openai.com/v1").strip()
    api_key = (os.getenv("OPENAI_IMAGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    resolved_model = (os.getenv("LOCAL_EXPLAINER_IMAGE_MODEL") or "gpt-image-1").strip() or "gpt-image-1"

    client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0) if api_key else openai.OpenAI(base_url=base_url, max_retries=0)

    def _call(size_value: str):
        request = {"provider": "openai", "base_url": base_url, "model": resolved_model,
                   "prompt": full_prompt, "size": size_value, "quality": quality,
                   "output_format": "png", "width": target_width, "height": target_height}
        def dispatch():
            result = client.images.generate(model=resolved_model, prompt=full_prompt,
                                            size=size_value, quality=quality, output_format="png")
            payload = result.data[0].b64_json if getattr(result, "data", None) else None
            if not payload:
                raise RuntimeError(f"OpenAI returned no image payload for {output_path.name}")
            # Save the acknowledged encoded response before decoding/conversion.
            return payload.encode("ascii")
        encoded = paid_bytes(request, dispatch, output_path=output_path,
                             key="image-auto" if size_value == "auto" else "paid")
        return base64.b64decode(encoded, validate=True)

    try:
        raw = _call(size)
    except Exception as exc:
        body = getattr(exc, "body", {})
        error = body.get("error", body) if isinstance(body, dict) else {}
        if status_code(exc) not in {400, 422} or error.get("param") != "size":
            raise
        raw = _call("auto")
    atomic_bytes(output_path, raw)

    _ensure_png(output_path)
    _normalize_image_to_target(output_path, target_width, target_height)
    return output_path


def generate_image(
    prompt: str,
    output_path: str | Path,
    model: str = DEFAULT_IMAGE_GEN_MODEL,
    *,
    title: str = "",
    runner_model: str | None = None,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    **_: Any,
) -> Path:
    """Generate a still image, preferring the local Codex CLI (subscription-covered).

    The Codex native image tool is the primary path; the metered OpenAI gpt-image
    API is only a fallback when Codex is unavailable before dispatch. Set
    LOCAL_EXPLAINER_IMAGE_PROVIDER=openai to force the API path explicitly.
    """
    output_path = Path(output_path)
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Image generation requires a non-empty prompt")

    resolved_model = str(model or DEFAULT_IMAGE_GEN_MODEL).strip() or DEFAULT_IMAGE_GEN_MODEL
    if resolved_model == "qwen/qwen-image-2512":
        return _generate_image_qwen(prompt=prompt, output_path=output_path, model=resolved_model,
                                    target_width=target_width, target_height=target_height)
    provider = (os.getenv("LOCAL_EXPLAINER_IMAGE_PROVIDER") or "codex").strip().lower()
    if provider != "openai" and _codex_cli_available():
        codex_prompt = build_codex_image_prompt(
            prompt=prompt,
            output_path=output_path,
            image_model=resolved_model,
            title=title,
            target_width=target_width,
            target_height=target_height,
        )
        return _run_codex_exec_image(
            prompt=codex_prompt,
            output_path=output_path,
            runner_model=runner_model,
            target_width=target_width,
            target_height=target_height,
        )

    return _generate_image_openai(
        prompt=prompt,
        output_path=output_path,
        model=resolved_model,
        target_width=target_width,
        target_height=target_height,
    )


def _generate_image_qwen(*, prompt, output_path, model, target_width, target_height):
    """The explicit Qwen action uses Replicate and retains its original response."""
    import io
    import json
    import requests
    from PIL import Image

    inputs = {"prompt": prompt + "\n" + TEXT_DISCIPLINE,
              "aspect_ratio": "1:1" if target_width == target_height else "9:16" if target_height > target_width else "16:9",
              "output_format": "png", "go_fast": False}
    request = {"provider": "replicate", "model": model, "input": inputs,
               "target_width": target_width, "target_height": target_height}

    def dispatch():
        output = _get_replicate_client().run(model, input=inputs)
        items = output if isinstance(output, list) else [output]
        urls = [str(item.url if hasattr(item, "url") else item) for item in items]
        if not urls or not urls[0].startswith("https://"):
            raise RuntimeError("Qwen returned no image URL")
        return json.dumps(urls).encode()

    # Keep the paid acknowledgement before the separate, retryable output download.
    response = paid_bytes(request, dispatch, output_path=output_path, key="qwen-provider",
                          provenance={"provider": "replicate", "model": model})
    url = json.loads(response)[0]
    def download():
        downloaded = requests.get(url, timeout=(5, 60))
        downloaded.raise_for_status()
        raw = downloaded.content
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        return raw

    # Download recovery repeats only the original free read, never prediction creation.
    raw = paid_bytes({"source_request": request_digest(request), "url": url}, download,
                     output_path=output_path, key="qwen-image", recover=download)
    atomic_bytes(output_path, raw)
    _ensure_png(output_path)
    _normalize_image_to_target(output_path, target_width, target_height)
    return output_path


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
    target_width, target_height = _scene_target_dimensions(scene, kwargs)
    if visual_prompt:
        result = generate_image(
            visual_prompt,
            output_path,
            model=model or DEFAULT_IMAGE_GEN_MODEL,
            title=str(scene.get("title") or ""),
            runner_model=str(kwargs.get("runner_model") or "").strip() or None,
            target_width=target_width,
            target_height=target_height,
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
