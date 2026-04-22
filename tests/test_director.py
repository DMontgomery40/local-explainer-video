from __future__ import annotations

from core.director import (
    COMPACTION_BETA,
    FILES_API_BETA,
    FAST_MODE_BETA,
    OUTPUT_300K_BETA,
    build_storyboard_batch_request,
    build_storyboard_api_kwargs,
    load_prompt,
    resolve_refinement_runner,
    resolve_anthropic_request_config,
    resolve_director_reference_file_ids,
    resolve_remotion_skill_config,
)


def test_resolve_remotion_skill_config_reads_primary_env(monkeypatch):
    monkeypatch.setenv("REMOTION_SKILL_ID", "skill_primary")
    monkeypatch.delenv("DIRECTOR_REMOTION_SKILL_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_REMOTION_SKILL_ID", raising=False)
    monkeypatch.setenv("REMOTION_SKILL_VERSION", "42")

    skill_id, skill_version = resolve_remotion_skill_config()

    assert skill_id == "skill_primary"
    assert skill_version == "42"


def test_resolve_remotion_skill_config_supports_aliases(monkeypatch):
    monkeypatch.delenv("REMOTION_SKILL_ID", raising=False)
    monkeypatch.setenv("DIRECTOR_REMOTION_SKILL_ID", "skill_alias")
    monkeypatch.delenv("REMOTION_SKILL_VERSION", raising=False)

    skill_id, skill_version = resolve_remotion_skill_config()

    assert skill_id == "skill_alias"
    assert skill_version == "latest"


def test_build_storyboard_api_kwargs_includes_custom_skill_container():
    kwargs, resolved_skill_id = build_storyboard_api_kwargs(
        "hello",
        system_prompt="system",
        model="claude-sonnet-4-6",
        skill_id="skill_123",
        skill_version="99",
        reference_image_file_ids=[],
    )

    assert resolved_skill_id == "skill_123"
    assert kwargs["betas"] == [COMPACTION_BETA, "code-execution-2025-08-25", "skills-2025-10-02"]
    assert kwargs["container"]["skills"] == [
        {
            "type": "custom",
            "skill_id": "skill_123",
            "version": "99",
        }
    ]
    assert kwargs["tools"] == [{"type": "code_execution_20250825", "name": "code_execution"}]


def test_build_storyboard_api_kwargs_without_skill_is_plain_messages_request(monkeypatch):
    monkeypatch.delenv("REMOTION_SKILL_ID", raising=False)
    monkeypatch.delenv("DIRECTOR_REMOTION_SKILL_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_REMOTION_SKILL_ID", raising=False)
    monkeypatch.delenv("DIRECTOR_REFERENCE_IMAGE_FILE_IDS", raising=False)
    monkeypatch.delenv("DIRECTOR_CONTACT_SHEET_FILE_IDS", raising=False)

    kwargs, resolved_skill_id = build_storyboard_api_kwargs(
        "hello",
        system_prompt="system",
        model="claude-sonnet-4-6",
        reference_image_file_ids=[],
    )

    assert resolved_skill_id is None
    assert kwargs["betas"] == [COMPACTION_BETA]
    assert "container" not in kwargs
    assert "tools" not in kwargs


def test_resolve_director_reference_file_ids_reads_csv_env(monkeypatch):
    monkeypatch.setenv("DIRECTOR_REFERENCE_IMAGE_FILE_IDS", "file_img_a, file_img_b")

    image_ids = resolve_director_reference_file_ids()

    assert image_ids == ["file_img_a", "file_img_b"]


def test_build_storyboard_api_kwargs_attaches_reference_files(monkeypatch):
    monkeypatch.delenv("REMOTION_SKILL_ID", raising=False)
    monkeypatch.delenv("DIRECTOR_REMOTION_SKILL_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_REMOTION_SKILL_ID", raising=False)

    kwargs, resolved_skill_id = build_storyboard_api_kwargs(
        "hello",
        system_prompt="system",
        model="claude-sonnet-4-6",
        reference_image_file_ids=["file_img_a", "file_img_b"],
    )

    assert resolved_skill_id is None
    assert kwargs["betas"] == [COMPACTION_BETA, FILES_API_BETA]
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "file", "file_id": "file_img_a"}},
                {"type": "image", "source": {"type": "file", "file_id": "file_img_b"}},
                {"type": "text", "text": "hello"},
            ],
        }
    ]


def test_resolve_anthropic_request_config_uses_adaptive_thinking_and_effort():
    config = resolve_anthropic_request_config(model="claude-sonnet-4-6")

    assert config["thinking"] == {"type": "adaptive"}
    assert config["output_config"] == {"effort": "high"}
    assert config["max_tokens"] == 64000
    assert COMPACTION_BETA in config["extra_betas"]


def test_resolve_anthropic_request_config_clamps_invalid_sonnet_max_effort():
    config = resolve_anthropic_request_config(model="claude-sonnet-4-6", effort="max")

    assert config["output_config"] == {"effort": "high"}


def test_resolve_anthropic_request_config_adds_fast_mode_for_opus():
    config = resolve_anthropic_request_config(model="claude-opus-4-6", speed="fast")

    assert config["speed"] == "fast"
    assert FAST_MODE_BETA in config["extra_betas"]
    assert COMPACTION_BETA in config["extra_betas"]


def test_build_storyboard_batch_request_enables_300k_beta():
    request = build_storyboard_batch_request(
        "hello",
        system_prompt="system",
        model="claude-sonnet-4-6",
        custom_id="storyboard-1",
    )

    assert request["custom_id"] == "storyboard-1"
    assert request["params"]["max_tokens"] == 300000
    assert OUTPUT_300K_BETA in request["params"]["betas"]
    assert COMPACTION_BETA in request["params"]["betas"]


def test_load_prompt_inlines_reference_successful_plan():
    prompt = load_prompt("director_system")

    assert "{{REFERENCE_SUCCESSFUL_PLAN_04_08_1997_0}}" not in prompt
    assert '"project_name": "04-08-1997-0"' in prompt


def test_resolve_refinement_runner_prefers_codex_for_api_projects(monkeypatch):
    monkeypatch.setattr("core.local_planner.available_storyboard_runners", lambda: ["claude", "codex"])

    assert resolve_refinement_runner("api") == "codex"


def test_resolve_refinement_runner_falls_back_to_claude_for_api_projects(monkeypatch):
    monkeypatch.setattr("core.local_planner.available_storyboard_runners", lambda: ["claude"])

    assert resolve_refinement_runner("api") == "claude"
