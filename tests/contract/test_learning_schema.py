from __future__ import annotations

from pathlib import Path

from core.template_pipeline.learning_log import (
    ENTRY_ID_PATTERN,
    REGRESSION_TEST_PATTERN,
    parse_learning_markdown,
)


def test_learning_sections_present() -> None:
    path = Path("/Users/davidmontgomery/local-explainer-video/.codex/LEARNING.md")
    text = path.read_text(encoding="utf-8")
    assert "## Active Guardrails" in text
    assert "## Open Learnings" in text
    assert "## Resolved Learnings" in text
    assert "## Regression Learnings" in text


def test_entry_id_and_regression_reference_format() -> None:
    path = Path("/Users/davidmontgomery/local-explainer-video/.codex/LEARNING.md")
    doc = parse_learning_markdown(path)
    for entry in doc.entries:
        assert ENTRY_ID_PATTERN.match(entry.entry_id)
        if entry.status == "RESOLVED":
            assert REGRESSION_TEST_PATTERN.match(entry.regression_test_added)

