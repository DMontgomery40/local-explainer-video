from __future__ import annotations

from types import SimpleNamespace

import core.director as director


class _FakeAnthropicClient:
    def __init__(self, content_text: str) -> None:
        self._content_text = content_text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._content_text)])


def test_anthropic_parser_handles_unclosed_json_fence(monkeypatch) -> None:
    content = (
        "Here is the storyboard.\n"
        "```json\n"
        '{"scenes":[{"id":1,"title":"A","narration":"N","visual_prompt":"V"}]}'
    )
    monkeypatch.setattr(director, "_get_anthropic_client", lambda: _FakeAnthropicClient(content))
    scenes = director._generate_with_anthropic("sys", "user", require_visual_prompt=False)
    assert len(scenes) == 1
    assert scenes[0]["narration"] == "N"


def test_anthropic_parser_handles_wrapped_json_without_fence(monkeypatch) -> None:
    content = (
        "Draft output follows.\n"
        'payload={"scenes":[{"id":2,"title":"B","narration":"N2","visual_prompt":"V2"}]}\n'
        "done."
    )
    monkeypatch.setattr(director, "_get_anthropic_client", lambda: _FakeAnthropicClient(content))
    scenes = director._generate_with_anthropic("sys", "user", require_visual_prompt=False)
    assert len(scenes) == 1
    assert scenes[0]["title"] == "B"
