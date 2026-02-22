from __future__ import annotations

import pytest

from core.template_pipeline import (
    TemplateSelectionError,
    load_manifest,
    select_template,
)


def _scene(scene_type: str, structured_data: dict) -> dict:
    return {
        "id": 1,
        "uid": "t1",
        "title": "Title",
        "narration": "Narration",
        "scene_type": scene_type,
        "structured_data": structured_data,
    }


def test_selector_uses_predicate_for_bar_count() -> None:
    manifest = load_manifest()
    spec = select_template(
        _scene(
            "bar_volume_chart",
            {
                "metric": "P300",
                "bars": [{"label": "S1", "value": 1}, {"label": "S2", "value": 2}],
            },
        ),
        manifest,
        production_only=False,
    )
    assert spec.template_id == "bar_volume_chart_2panel_v1"


def test_selector_production_mode_fails_without_production_ready_coverage() -> None:
    manifest = load_manifest()
    with pytest.raises(TemplateSelectionError, match="production-ready"):
        select_template(
            _scene("multi_session_trend", {"metric": "M", "points": []}),
            manifest,
            production_only=True,
        )


def test_selector_no_silent_prod_fallback_even_when_scene_type_exists() -> None:
    manifest = load_manifest()
    with pytest.raises(TemplateSelectionError, match="No production template matched selector|production-ready"):
        select_template(
            _scene(
                "bar_volume_chart",
                {
                    "metric": "P300",
                    "bars": [{"label": "S1", "value": 1}, {"label": "S2", "value": 2}, {"label": "S3", "value": 3}],
                    "trend": "ascending",
                },
            ),
            manifest,
            production_only=True,
        )

