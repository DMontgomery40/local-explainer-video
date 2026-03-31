"""Lazy exports for explainer-video pipeline helpers."""


def generate_storyboard(*args, **kwargs):
    from .director import generate_storyboard as _generate_storyboard

    return _generate_storyboard(*args, **kwargs)


def generate_storyboard_api(*args, **kwargs):
    from .director import generate_storyboard_api as _generate_storyboard_api

    return _generate_storyboard_api(*args, **kwargs)


def generate_image(*args, **kwargs):
    from .image_gen import generate_image as _generate_image

    return _generate_image(*args, **kwargs)


def edit_image(*args, **kwargs):
    from .image_gen import edit_image as _edit_image

    return _edit_image(*args, **kwargs)


def generate_audio(*args, **kwargs):
    from .voice_gen import generate_audio as _generate_audio

    return _generate_audio(*args, **kwargs)


def assemble_video(*args, **kwargs):
    from .video_assembly import assemble_video as _assemble_video

    return _assemble_video(*args, **kwargs)


def render_plan_scenes(*args, **kwargs):
    from .remotion_bridge import render_plan_scenes as _render_plan_scenes

    return _render_plan_scenes(*args, **kwargs)


__all__ = [
    "generate_storyboard",
    "generate_storyboard_api",
    "generate_image",
    "edit_image",
    "generate_audio",
    "assemble_video",
    "render_plan_scenes",
]
