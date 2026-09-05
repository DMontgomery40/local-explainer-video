from __future__ import annotations

from core.director_claude_local import (
    build_claude_print_command,
    build_claude_print_env,
    build_local_storyboard_prompt,
    local_reference_contact_sheet_paths,
    resolve_claude_code_max_output_tokens,
    storyboard_output_schema,
)


def test_local_reference_contact_sheet_paths_exist():
    paths = local_reference_contact_sheet_paths()

    assert len(paths) >= 1
    assert all(path.exists() for path in paths)


def test_build_local_storyboard_prompt_mentions_local_mode_and_contact_sheets():
    prompt = build_local_storyboard_prompt("example input")

    assert "claude -p" in prompt
    assert "reference contact sheets" in prompt
    assert "example input" in prompt
    assert "04-08-1997-0" in prompt


def test_storyboard_output_schema_requires_scenes():
    schema = storyboard_output_schema()

    assert schema["required"] == ["scenes"]
    assert schema["properties"]["scenes"]["type"] == "array"


def test_build_claude_print_command_uses_current_cli_flags():
    command = build_claude_print_command('{"type":"object"}')

    assert command[0].endswith("claude")
    assert "-p" in command
    assert "--output-format" in command
    assert "json" in command
    assert "--json-schema" in command
    assert "--add-dir" in command
    assert "--dangerously-skip-permissions" in command
    assert "--no-session-persistence" in command


def test_resolve_claude_code_max_output_tokens_defaults_to_model_cap(monkeypatch):
    monkeypatch.delenv("CLAUDE_LOCAL_REMOTION_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("DIRECTOR_ANTHROPIC_MAX_TOKENS", raising=False)

    assert resolve_claude_code_max_output_tokens("claude-sonnet-4-6") == 64000


def test_build_claude_print_env_sets_claude_code_max_output_tokens(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("DIRECTOR_ANTHROPIC_MAX_TOKENS", "64000")

    env = build_claude_print_env("claude-sonnet-4-6")

    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"


def test_build_claude_print_env_prefers_subscription_auth_by_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("CLAUDE_LOCAL_REMOTION_USE_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_API_KEY", raising=False)

    env = build_claude_print_env("claude-opus-4-6")

    assert "ANTHROPIC_API_KEY" not in env


def test_build_claude_print_env_can_explicitly_keep_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("CLAUDE_LOCAL_REMOTION_USE_API_KEY", "true")

    env = build_claude_print_env("claude-opus-4-6")

    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"
