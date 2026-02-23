from pathlib import Path

from core.visual_gen import BLENDER_QEEG_MARKER, generate_scene_visual, is_blender_scene


def test_is_blender_scene_by_backend_flag() -> None:
    scene = {"id": 1, "render_backend": "blender", "visual_prompt": "plain prompt"}
    assert is_blender_scene(scene) is True


def test_is_blender_scene_by_prompt_marker() -> None:
    scene = {"id": 2, "visual_prompt": f"render this map {BLENDER_QEEG_MARKER}"}
    assert is_blender_scene(scene) is True


def test_generate_scene_visual_routes_to_blender(monkeypatch) -> None:
    scene = {"id": 3, "visual_prompt": f"scene with {BLENDER_QEEG_MARKER}"}
    project_dir = Path("/tmp/project")
    expected = project_dir / "images" / "scene_003.png"

    calls: list[tuple[str, dict]] = []

    def fake_blender(*args, **kwargs):
        calls.append(("blender", kwargs))
        return expected

    def fake_image(*args, **kwargs):
        calls.append(("image", kwargs))
        return expected

    monkeypatch.setattr("core.visual_gen.render_blender_scene", fake_blender)
    monkeypatch.setattr("core.visual_gen.generate_scene_image", fake_image)

    out = generate_scene_visual(scene, project_dir, model="qwen/qwen-image-2512")
    assert out == expected
    assert calls and calls[0][0] == "blender"


def test_generate_scene_visual_falls_back_to_image(monkeypatch) -> None:
    scene = {"id": 4, "visual_prompt": "non-blender visual"}
    project_dir = Path("/tmp/project")
    expected = project_dir / "images" / "scene_004.png"
    calls: list[tuple[str, dict]] = []

    def fake_blender(*args, **kwargs):
        calls.append(("blender", kwargs))
        return expected

    def fake_image(*args, **kwargs):
        calls.append(("image", kwargs))
        return expected

    monkeypatch.setattr("core.visual_gen.render_blender_scene", fake_blender)
    monkeypatch.setattr("core.visual_gen.generate_scene_image", fake_image)

    out = generate_scene_visual(scene, project_dir, model="google/imagen-4")
    assert out == expected
    assert calls and calls[0][0] == "image"
