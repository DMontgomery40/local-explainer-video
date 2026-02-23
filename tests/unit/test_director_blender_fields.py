from core.director import _build_director_system_prompt, _validate_scenes


def test_validate_scenes_preserves_blender_payload_and_backend() -> None:
    scenes = [
        {
            "id": 3,
            "title": "Alpha coherence",
            "narration": "We review alpha coherence now.",
            "visual_prompt": "Brain map with C3 and P4 coherence links",
            "render_backend": "blender",
            "blender": {
                "extract": {"session_index": 2, "band": "alpha", "metric": "coherence"},
                "animation": {"enabled": True, "duration_sec": 4, "fps": 24},
            },
            "scene_type": "coherence_map",
        }
    ]

    out = _validate_scenes(scenes)
    assert len(out) == 1
    scene = out[0]
    assert scene["render_backend"] == "blender"
    assert scene["scene_type"] == "coherence_map"
    assert scene["blender"]["extract"]["session_index"] == 2
    assert scene["blender"]["animation"]["enabled"] is True


def test_validate_scenes_auto_tags_blender_for_eeg_label_prompt() -> None:
    scenes = [
        {
            "id": 1,
            "title": "Central-parietal synchronization",
            "narration": "The central and parietal sites become synchronized.",
            "visual_prompt": "Top-down brain with C3 Cz C4 P3 Pz P4 labels and coherence lines",
        }
    ]
    out = _validate_scenes(scenes)
    assert out[0]["render_backend"] == "blender"


def test_validate_scenes_defaults_non_brain_to_template_pack() -> None:
    scenes = [
        {
            "id": 2,
            "title": "Roadmap",
            "narration": "We will walk through the next five steps.",
            "visual_prompt": "A clean roadmap panel with milestones and timeline markers",
        }
    ]
    out = _validate_scenes(scenes)
    assert out[0]["render_backend"] == "template_pack"


def test_director_system_prompt_contains_blender_skill_routing_block() -> None:
    prompt = _build_director_system_prompt()
    assert "blender-mcp-qeeg-runtime" in prompt
    assert '"render_backend": "blender"' in prompt
