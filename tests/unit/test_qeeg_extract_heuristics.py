from core.qeeg_extract import extract_qeeg_visual_data


def test_heuristic_prefers_band_relevant_electrode_map() -> None:
    data_pack = {
        "derived": {
            "theta": {
                "topomap": {
                    "Fp1": 9.0,
                    "Fp2": 8.0,
                    "F3": 7.0,
                    "F4": 6.0,
                    "C3": 5.0,
                }
            },
            "alpha": {
                "zscore_map": {
                    "Fp1": 1.1,
                    "Fp2": -0.8,
                    "F3": 0.2,
                    "F4": -0.1,
                    "C3": 0.4,
                    "C4": 0.5,
                }
            },
        }
    }
    scene = {
        "title": "Alpha map",
        "visual_prompt": "Show alpha band topography",
        "band": "alpha",
        "metric": "zscore",
    }
    electrode_values, coherence_edges = extract_qeeg_visual_data(data_pack, scene=scene)
    assert coherence_edges == []
    assert electrode_values["Fp1"] == 1.1
    assert electrode_values["Fp2"] == -0.8
    assert electrode_values["C4"] == 0.5


def test_heuristic_extracts_coherence_edges_from_nested_list() -> None:
    data_pack = {
        "derived": {
            "alpha": {
                "coherence_network": {
                    "edges": [
                        {"a": "F3", "b": "C3", "value": 0.82},
                        {"a": "F4", "b": "C4", "value": 0.76},
                        {"a": "Fp1", "b": "Fp2", "value": 0.65},
                    ]
                }
            }
        }
    }
    scene = {"visual_prompt": "alpha coherence network", "band": "alpha"}
    electrode_values, coherence_edges = extract_qeeg_visual_data(data_pack, scene=scene)
    assert electrode_values == {}
    assert coherence_edges == [
        {"a": "C3", "b": "F3", "value": 0.82},
        {"a": "C4", "b": "F4", "value": 0.76},
        {"a": "Fp1", "b": "Fp2", "value": 0.65},
    ]


def test_heuristic_falls_back_to_fact_rows_when_needed() -> None:
    data_pack = {
        "facts": [
            {"fact_type": "p300_cp_site", "session_index": 1, "site": "C3", "uv": 5.7},
            {"fact_type": "p300_cp_site", "session_index": 1, "site": "F3", "uv": 4.2},
            {"fact_type": "coherence_pair", "session_index": 1, "site_a": "F3", "site_b": "C3", "coherence": 0.66},
            {"fact_type": "coherence_pair", "session_index": 2, "site_a": "F3", "site_b": "F4", "coherence": 0.41},
        ]
    }
    cfg = {"session_index": 1}
    electrode_values, coherence_edges = extract_qeeg_visual_data(data_pack, config=cfg)
    assert electrode_values == {"C3": 5.7, "F3": 4.2}
    assert coherence_edges == [{"a": "C3", "b": "F3", "value": 0.66}]


def test_heuristic_session_filter_applies_to_nested_electrode_maps() -> None:
    data_pack = {
        "sessions": [
            {
                "session_index": 1,
                "derived": {
                    "alpha": {
                        "topomap": {
                            "F3": 1.0,
                            "F4": 2.0,
                            "C3": 3.0,
                            "C4": 4.0,
                        }
                    }
                },
            },
            {
                "session_index": 2,
                "derived": {
                    "alpha": {
                        "topomap": {
                            "F3": 10.0,
                            "F4": 20.0,
                            "C3": 30.0,
                            "C4": 40.0,
                        }
                    }
                },
            },
        ]
    }
    scene = {"band": "alpha", "metric": "zscore"}
    s1_values, _ = extract_qeeg_visual_data(data_pack, scene=scene, config={"session_index": 1})
    s2_values, _ = extract_qeeg_visual_data(data_pack, scene=scene, config={"session_index": 2})

    assert s1_values == {"C3": 3.0, "C4": 4.0, "F3": 1.0, "F4": 2.0}
    assert s2_values == {"C3": 30.0, "C4": 40.0, "F3": 10.0, "F4": 20.0}


def test_heuristic_session_filter_applies_to_nested_coherence_edges() -> None:
    data_pack = {
        "sessions": [
            {
                "session_index": 1,
                "derived": {
                    "alpha": {
                        "coherence_network": {
                            "edges": [
                                {"a": "F3", "b": "C3", "value": 0.81},
                                {"a": "F4", "b": "C4", "value": 0.74},
                            ]
                        }
                    }
                },
            },
            {
                "session_index": 2,
                "derived": {
                    "alpha": {
                        "coherence_network": {
                            "edges": [
                                {"a": "F3", "b": "F4", "value": 0.42},
                                {"a": "C3", "b": "C4", "value": 0.65},
                            ]
                        }
                    }
                },
            },
        ]
    }
    scene = {"band": "alpha", "visual_prompt": "alpha coherence"}
    _, s1_edges = extract_qeeg_visual_data(data_pack, scene=scene, config={"session_index": 1})
    _, s2_edges = extract_qeeg_visual_data(data_pack, scene=scene, config={"session_index": 2})

    assert s1_edges == [
        {"a": "C3", "b": "F3", "value": 0.81},
        {"a": "C4", "b": "F4", "value": 0.74},
    ]
    assert s2_edges == [
        {"a": "C3", "b": "C4", "value": 0.65},
        {"a": "F3", "b": "F4", "value": 0.42},
    ]
