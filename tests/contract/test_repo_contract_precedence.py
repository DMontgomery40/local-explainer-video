from __future__ import annotations

from pathlib import Path


def test_repo_contract_declares_precedence_and_advisory_docs() -> None:
    path = Path("/Users/davidmontgomery/local-explainer-video/.codex/REPO_CONTRACT.md")
    text = path.read_text(encoding="utf-8")
    assert "Document Precedence" in text
    assert "HANDOFF-deterministic-template-pipeline.md" in text
    assert "HANDOFF_NEXT_AGENT_DETERMINISTIC_TEMPLATE_PIPELINE_2026-02-22.md" in text
    assert "Advisory-Only Inputs" in text

