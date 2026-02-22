from __future__ import annotations

import pytest

from core.template_pipeline import ALL_SCENE_TYPES, validate_scene


@pytest.mark.parametrize("scene_type", list(ALL_SCENE_TYPES))
def test_every_scene_type_accepts_minimal_structured_payload(scene_type: str) -> None:
    scene = {
        "id": 1,
        "uid": "test",
        "title": "Test Scene",
        "narration": "Test narration",
        "scene_type": scene_type,
        "structured_data": {},
        "visual_prompt": "",
    }
    validated = validate_scene(scene)
    assert validated["scene_type"] == scene_type
    assert isinstance(validated["structured_data"], dict)


def test_multi_session_target_band_accepts_legacy_low_high_keys() -> None:
    scene = {
        "id": 7,
        "uid": "legacy-target-band",
        "title": "Trend",
        "narration": "Trend narration",
        "scene_type": "multi_session_trend",
        "structured_data": {
            "metric": "P300",
            "points": [{"label": "Session 1", "value": 6.2}],
            "target_band": {"low": 6.0, "high": 14.0},
        },
    }
    validated = validate_scene(scene)
    band = validated["structured_data"]["target_band"]
    assert band["min"] == 6.0
    assert band["max"] == 14.0


def test_table_dashboard_rows_accept_range_lists_in_cells() -> None:
    scene = {
        "id": 11,
        "uid": "table-range-cell",
        "title": "Table",
        "narration": "Table narration",
        "scene_type": "table_dashboard",
        "structured_data": {
            "columns": ["test", "actual_sec", "target_sec"],
            "rows": [
                {"test": "Trail Making A", "actual_sec": 73, "target_sec": [74, 127]},
            ],
        },
    }
    validated = validate_scene(scene)
    row = validated["structured_data"]["rows"][0]
    assert row["target_sec"] == [74, 127]


def test_coherence_network_edges_accept_pair_key() -> None:
    scene = {
        "id": 12,
        "uid": "coherence-pair",
        "title": "Coherence",
        "narration": "Coherence narration",
        "scene_type": "coherence_network_map",
        "structured_data": {
            "nodes": [{"id": "C3"}, {"id": "C4"}],
            "edges": [{"pair": "C3_C4", "value": 0.59}],
        },
    }
    validated = validate_scene(scene)
    edge = validated["structured_data"]["edges"][0]
    assert edge["from"] == "C3"
    assert edge["to"] == "C4"


def test_quality_alert_accepts_string_flags() -> None:
    scene = {
        "id": 13,
        "uid": "quality-string-flags",
        "title": "Quality",
        "narration": "Quality narration",
        "scene_type": "quality_alert",
        "structured_data": {
            "flags": ["black_X_sites_low_rare_responses"],
            "impacted_metrics": ["P300"],
        },
    }
    validated = validate_scene(scene)
    flags = validated["structured_data"]["flags"]
    assert flags[0]["kind"] == "black_X_sites_low_rare_responses"
    assert flags[0]["severity"] == "warning"


def test_dotplot_variability_accepts_sessions_dict_with_latencies() -> None:
    scene = {
        "id": 14,
        "uid": "dotplot-dict",
        "title": "Dotplot",
        "narration": "Dotplot narration",
        "scene_type": "dotplot_variability",
        "structured_data": {
            "sessions": {
                "one": {"latencies": [50.0, 60.0, 70.0]},
            },
        },
    }
    validated = validate_scene(scene)
    sessions = validated["structured_data"]["sessions"]
    assert sessions[0]["label"] == "one"
    assert sessions[0]["mean"] == 60.0
    assert sessions[0]["spread"] == 20.0
