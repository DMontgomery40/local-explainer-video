from __future__ import annotations

from core.template_pipeline import detect_qwen_fallback_scene_ids


def test_detect_qwen_fallback_scene_ids() -> None:
    plan = {
        "scenes": [
            {"id": 0, "render_mode": "template_deterministic"},
            {"id": 1, "fallback_to_qwen": True},
            {"id": 2, "render_mode": "qwen_fallback"},
            {"id": 2, "render_mode": "qwen_fallback"},
        ]
    }
    assert detect_qwen_fallback_scene_ids(plan) == [1, 2]
