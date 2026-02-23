"""Core modules for qEEG Explainer Video Generator."""

from typing import Any, Callable


def _missing_dependency(name: str, error: Exception) -> Callable[..., Any]:
    def _raise(*_: Any, **__: Any) -> Any:
        raise ModuleNotFoundError(
            f"{name} is unavailable because an optional dependency failed to import: {error}"
        ) from error

    return _raise


from .director import generate_storyboard
from .image_gen import generate_image, edit_image

try:
    from .voice_gen import generate_audio
except Exception as exc:  # pragma: no cover - defensive import guard for lightweight test envs.
    generate_audio = _missing_dependency("generate_audio", exc)

try:
    from .video_assembly import assemble_video
except Exception as exc:  # pragma: no cover - defensive import guard for lightweight test envs.
    assemble_video = _missing_dependency("assemble_video", exc)

__all__ = [
    "generate_storyboard",
    "generate_image",
    "generate_audio",
    "assemble_video",
]
