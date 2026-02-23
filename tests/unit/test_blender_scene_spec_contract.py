from core.blender_gen import _build_scene_spec


def test_scene_spec_includes_contract_and_style_defaults() -> None:
    scene = {
        "id": 4,
        "uid": "abcd1234",
        "title": "Alpha map",
        "visual_prompt": "[[BLENDER_QEEG]]",
        "blender": {
            "electrode_values": {"c3": 2.1, "p4": -1.3},
            "coherence_edges": [{"a": "c3", "b": "p4", "value": 0.44}],
        },
    }

    spec = _build_scene_spec(scene=scene, data_pack=None)
    assert spec["contract_version"] == "brain_basemodel_v1"
    assert spec["scene_uid"] == "abcd1234"
    assert spec["style"] == {
        "lighting_preset": "clinical_glow",
        "camera_preset": "three_quarter_left",
        "palette": "teal-amber",
    }
    assert spec["electrode_values"] == {"C3": 2.1, "P4": -1.3}
    assert spec["coherence_edges"] == [{"a": "C3", "b": "P4", "value": 0.44}]


def test_scene_spec_style_normalizes_invalid_presets() -> None:
    scene = {
        "id": 5,
        "title": "Beta coherence",
        "style_preset": "unknown_style",
        "camera_preset": "invalid_camera",
        "blender": {
            "style": {
                "lighting_preset": "focus_contrast",
                "camera_preset": "top_center",
                "palette": "",
            }
        },
    }

    spec = _build_scene_spec(scene=scene, data_pack=None)
    assert spec["style"]["lighting_preset"] == "focus_contrast"
    assert spec["style"]["camera_preset"] == "top_center"
    assert spec["style"]["palette"] == "teal-amber"


def test_scene_spec_extract_hints_sanitized() -> None:
    scene = {
        "id": 6,
        "title": "Session-specific map",
        "blender": {
            "extract": {
                "session_index": "2",
                "band": "alpha",
                "metric": "coherence",
                "electrode_path": "sessions.1.channels",
                "coherence_path": "sessions.1.edges",
                "junk": "ignored",
            }
        },
    }

    spec = _build_scene_spec(scene=scene, data_pack=None)
    assert spec["extract"] == {
        "session_index": 2,
        "band": "alpha",
        "metric": "coherence",
        "electrode_path": "sessions.1.channels",
        "coherence_path": "sessions.1.edges",
    }
