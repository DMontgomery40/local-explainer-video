from __future__ import annotations

from core.scene_modes import plan_has_cathode_motion_scenes, scene_is_cathode_motion


def test_scene_is_cathode_motion_accepts_scene_type_motion():
    assert scene_is_cathode_motion({"scene_type": "motion"}) is True


def test_scene_is_cathode_motion_accepts_native_composition_mode():
    assert scene_is_cathode_motion({"composition": {"mode": "native"}}) is True


def test_scene_is_cathode_motion_accepts_native_remotion_manifestation():
    assert scene_is_cathode_motion({"composition": {"manifestation": "native_remotion"}}) is True


def test_scene_is_cathode_motion_rejects_prompt_bearing_image_scene():
    assert scene_is_cathode_motion({"scene_type": "image", "visual_prompt": "Exact prompt"}) is False


def test_plan_has_cathode_motion_scenes_supports_scene_lists_and_plan_dicts():
    scenes = [
        {"scene_type": "image", "visual_prompt": "Prompt"},
        {"composition": {"mode": "native"}},
    ]

    assert plan_has_cathode_motion_scenes(scenes) is True
    assert plan_has_cathode_motion_scenes({"scenes": scenes}) is True
    assert plan_has_cathode_motion_scenes({"scenes": [{"scene_type": "image"}]}) is False
