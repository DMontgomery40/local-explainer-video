#!/usr/bin/env python3
"""Run the local Claude Code storyboard lane and save a plan under projects/."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.director_claude_local import DEFAULT_CLAUDE_MODEL, generate_storyboard_claude_local


def _normalize_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_scene in enumerate(scenes, start=1):
        scene = dict(raw_scene)
        scene["id"] = int(scene.get("id") or index)
        scene["uid"] = str(scene.get("uid") or uuid.uuid4())[:8]
        normalized.append(scene)
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local `claude -p` storyboard generation")
    parser.add_argument("input_path", help="Path to patient/source markdown file")
    parser.add_argument("--project-prefix", required=True, help="Prefix for projects/ output folder")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source_path = Path(args.input_path).expanduser().resolve()
    input_text = source_path.read_text(encoding="utf-8")
    project_name = f"{args.project_prefix}__claude_local_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    project_dir = REPO_ROOT / "projects" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "source_input.md").write_text(input_text, encoding="utf-8")

    scenes = _normalize_scenes(
        generate_storyboard_claude_local(input_text, project_dir=project_dir)
    )
    plan = {
        "meta": {
            "project_name": project_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "llm_provider": "claude-local-print",
            "claude_local_model": DEFAULT_CLAUDE_MODEL,
            "source_markdown_path": str(source_path),
        },
        "scenes": scenes,
    }
    plan_path = project_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_plan_path={plan_path}")
    print(f"scene_count={len(scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
