from __future__ import annotations

from pathlib import Path

import pytest

from core.local_planner import (
    CathodeResourceMap,
    DEFAULT_CLAUDE_BINARY,
    build_cathode_resource_map,
    build_claude_command,
    build_codex_command,
    build_storyboard_prompt,
    normalize_storyboard_runner,
    validate_cathode_ready_scenes,
)


def test_normalize_storyboard_runner_maps_legacy_provider_names():
    assert normalize_storyboard_runner("openai") == "codex"
    assert normalize_storyboard_runner("anthropic") == "claude"
    assert normalize_storyboard_runner("codex") == "codex"
    assert normalize_storyboard_runner("claude") == "claude"


def test_build_codex_command_uses_exec_and_explicit_cathode_access(tmp_path):
    resource_map = build_cathode_resource_map()
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "last.json"

    command = build_codex_command(schema_path, output_path, resource_map.cathode_root)

    assert command[:6] == ["codex", "--search", "-a", "never", "-s", "read-only"]
    assert "exec" in command
    assert "-C" in command
    assert str(resource_map.cathode_root) in command
    assert str(schema_path) in command
    assert str(output_path) in command


def test_build_claude_command_uses_explicit_binary_and_json_schema():
    resource_map = build_cathode_resource_map()
    command = build_claude_command('{"type":"object"}', resource_map.cathode_root)

    assert command[0] == str(DEFAULT_CLAUDE_BINARY)
    assert "-p" in command
    assert "--output-format" in command
    assert "json" in command
    assert "--json-schema" in command
    assert "--add-dir" in command
    assert str(resource_map.cathode_root) in command
    assert "--no-session-persistence" in command


def test_build_storyboard_prompt_includes_real_cathode_paths_and_art_first_note(monkeypatch):
    resource_map = CathodeResourceMap(
        cathode_root=Path("/Users/davidmontgomery/cathode"),
        template_backgrounds_dir=Path("/Users/davidmontgomery/cathode/template_deck/backgrounds"),
        text_zones_json=Path("/Users/davidmontgomery/cathode/template_deck/text_zones.json"),
        template_layout_map_ts=Path("/Users/davidmontgomery/cathode/frontend/src/remotion/templateLayoutMap.ts"),
        clinical_template_prompt=Path("/Users/davidmontgomery/cathode/prompts/director_clinical_template_system_prompt.txt"),
        scene_family_contracts=Path("/Users/davidmontgomery/cathode/skills/cathode-remotion-development/references/scene-family-contracts.md"),
        remotion_architecture=Path("/Users/davidmontgomery/cathode/skills/cathode-remotion-development/references/cathode-remotion-architecture.md"),
        codex_skill=Path("/Users/davidmontgomery/.codex/skills/remotion/SKILL.md"),
        claude_skill=Path("/Users/davidmontgomery/.claude/skills/remotion/SKILL.md"),
        quality_bar_paths=(),
    )

    monkeypatch.setattr("core.local_planner.build_cathode_resource_map", lambda: resource_map)
    monkeypatch.setattr(
        "core.local_planner._background_ids",
        lambda _resource_map: ["metric_improvement", "clinical_explanation"],
    )
    monkeypatch.setattr(
        "core.local_planner._read_text",
        lambda path: f"CONTENTS FOR {path}",
    )
    prompt = build_storyboard_prompt("Patient data here", "SYSTEM")

    assert "/Users/davidmontgomery/cathode/template_deck/text_zones.json" in prompt
    assert "/Users/davidmontgomery/cathode/frontend/src/remotion/templateLayoutMap.ts" in prompt
    assert "/Users/davidmontgomery/.codex/skills/remotion/SKILL.md" in prompt
    assert "art-first" in prompt
    assert '"scenes"' in prompt


def test_validate_cathode_ready_scenes_accepts_motion_template_scene():
    scenes = validate_cathode_ready_scenes(
        {
            "scenes": [
                {
                    "title": "Signal Timing",
                    "narration": "The timing stayed inside the target range across all three sessions.",
                    "scene_type": "motion",
                    "visual_prompt": None,
                    "on_screen_text": ["P300 Signal Timing", "Within Range"],
                    "composition": {
                        "family": "metric_improvement",
                        "mode": "native",
                        "props": {
                            "background_id": "metric_improvement",
                            "headline": "P300 Signal Timing",
                            "metric_name": "P300 Timing",
                            "stages": [
                                {"value": "276 ms", "label": "Session 1"},
                                {"value": "308 ms", "label": "Session 2"},
                                {"value": "320 ms", "label": "Session 3"},
                            ],
                            "delta": "Always inside target",
                            "direction": "stable",
                        },
                    },
                }
            ]
        }
    )

    scene = scenes[0]
    assert scene["scene_type"] == "motion"
    assert scene["visual_prompt"] == ""
    assert scene["motion"]["template_id"] == "metric_improvement"
    assert scene["composition"]["manifestation"] == "native_remotion"
    assert scene["composition"]["props"]["background_id"] == "metric_improvement"


def test_validate_cathode_ready_scenes_rejects_unknown_background_id():
    with pytest.raises(ValueError, match="unknown Cathode background_id"):
        validate_cathode_ready_scenes(
            {
                "scenes": [
                    {
                        "title": "Bad Background",
                        "narration": "Narration is present.",
                        "scene_type": "motion",
                        "visual_prompt": None,
                        "on_screen_text": [],
                        "composition": {
                            "family": "metric_improvement",
                            "mode": "native",
                            "props": {
                                "background_id": "not_a_real_background",
                                "headline": "Oops",
                            },
                        },
                    }
                ]
            }
        )
