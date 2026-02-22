#!/usr/bin/env python3
"""Migrate an existing plan.json to deterministic scene typing schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.template_pipeline import (
    SCHEMA_VERSION,
    annotate_scenes_with_types,
    validate_scenes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate project plan.json to deterministic template schema.")
    parser.add_argument("--project", required=True, help="Project folder name under ./projects or full path.")
    parser.add_argument(
        "--provider",
        default="",
        choices=["", "openai", "anthropic"],
        help="Override scene-typer provider. Defaults to plan.meta.llm_provider.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    project_arg = Path(args.project)
    project_dir = project_arg if project_arg.exists() else (repo_root / "projects" / args.project)
    if not project_dir.exists():
        raise SystemExit(f"Project not found: {project_dir}")

    plan_path = project_dir / "plan.json"
    if not plan_path.exists():
        raise SystemExit(f"Missing plan.json: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scenes = plan.get("scenes")
    if not isinstance(scenes, list):
        raise SystemExit("plan.json missing scenes array")

    meta = plan.setdefault("meta", {})
    provider = str(args.provider or meta.get("llm_provider") or "openai")
    if provider not in {"openai", "anthropic"}:
        provider = "openai"
    input_text = str(meta.get("input_text") or "")

    typed = annotate_scenes_with_types(
        scenes=[s for s in scenes if isinstance(s, dict)],
        input_text=input_text,
        provider=provider,  # type: ignore[arg-type]
    )
    plan["scenes"] = validate_scenes(typed)
    meta["schema_version"] = SCHEMA_VERSION
    meta["use_template_pipeline"] = True
    meta["render_pipeline"] = "template_deterministic"

    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Migrated plan: {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
