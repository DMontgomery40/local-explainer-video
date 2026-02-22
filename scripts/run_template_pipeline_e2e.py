#!/usr/bin/env python3
"""Run full deterministic pipeline E2E for a patient analysis input."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.director import generate_storyboard
from core.image_gen import generate_scene_image
from core.template_pipeline import SCHEMA_VERSION, validate_scenes


RUN_STATE: dict[str, object] = {
    "status": "not_started",
    "status_artifact": None,
    "project_dir": None,
}


def _sanitize(name: str) -> str:
    v = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name or "").strip())
    return v or "template_e2e"


def _load_input_text(source_project: Path) -> str:
    plan_path = source_project / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing source plan.json: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    text = str((plan.get("meta") or {}).get("input_text") or "").strip()
    if not text:
        raise ValueError(f"Source project has no meta.input_text: {plan_path}")
    return text


def _write_contact_sheet(project_dir: Path, scenes: list[dict]) -> Path:
    from PIL import Image, ImageDraw

    cards: list[tuple[int, Path, str]] = []
    for i, scene in enumerate(scenes):
        sid = int(scene.get("id", i))
        p = Path(str(scene.get("image_path") or ""))
        if not p.exists():
            continue
        label = f"{sid:03d} {scene.get('scene_type','?')} | {scene.get('template_id','?')}"
        cards.append((sid, p, label))

    if not cards:
        raise RuntimeError("No images found for contact sheet")

    thumb_w, thumb_h = 416, 232
    cols = 4
    rows = (len(cards) + cols - 1) // cols
    cell_h = thumb_h + 54
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (12, 18, 26))
    draw = ImageDraw.Draw(sheet)

    for idx, (_, path, label) in enumerate(cards):
        col = idx % cols
        row = idx // cols
        x = col * thumb_w
        y = row * cell_h
        img = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(img, (x, y))
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + cell_h), fill=(7, 13, 20))
        draw.text((x + 8, y + thumb_h + 10), label, fill=(220, 232, 244))

    out = project_dir / "contact_sheet_template_pipeline.png"
    sheet.save(out, format="PNG")
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _mark_failed(reason: str, exc: Exception | None = None) -> None:
    artifact = RUN_STATE.get("status_artifact")
    if not artifact:
        return
    payload = {
        "status": "failed",
        "updated_utc": _utc_now(),
        "reason": reason,
    }
    if exc is not None:
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(limit=5)
    _write_status(Path(str(artifact)), payload)


def _signal_handler(signum: int, _frame: object) -> None:
    _mark_failed(f"aborted_by_signal_{signum}")
    raise SystemExit(130)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic template pipeline E2E.")
    parser.add_argument(
        "--source-project",
        required=True,
        help="Existing project id/path to source patient analysis text from plan.meta.input_text",
    )
    parser.add_argument(
        "--provider",
        default="",
        choices=["", "openai", "anthropic"],
        help="Storyboard provider override (default: openai)",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Output project folder name suffix.",
    )
    parser.add_argument(
        "--template-mode",
        default="development",
        choices=["development", "production"],
        help="Template selection mode: production hard-fails on uncovered archetypes.",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    repo = _repo_root()
    load_dotenv(repo / ".env", override=True)
    os.environ["USE_TEMPLATE_PIPELINE"] = "true"
    os.environ["ALLOW_TEMPLATE_FALLBACK_TO_QWEN"] = "false"
    os.environ["TEMPLATE_PIPELINE_MODE"] = str(args.template_mode)

    src_arg = Path(args.source_project)
    source_project = src_arg if src_arg.exists() else (repo / "projects" / args.source_project)
    if not source_project.exists():
        raise SystemExit(f"Source project not found: {source_project}")

    src_plan = json.loads((source_project / "plan.json").read_text(encoding="utf-8"))
    input_text = _load_input_text(source_project)
    provider = args.provider or "openai"
    if provider not in {"openai", "anthropic"}:
        provider = "openai"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize(args.name) if args.name else "template_e2e"
    out_name = f"{source_project.name}__{suffix}_{timestamp}"
    project_dir = repo / "projects" / out_name
    project_dir.mkdir(parents=True, exist_ok=True)
    status_artifact = project_dir / "artifacts" / "template_e2e_status.json"
    RUN_STATE["status"] = "running"
    RUN_STATE["status_artifact"] = str(status_artifact)
    RUN_STATE["project_dir"] = str(project_dir)
    _write_status(
        status_artifact,
        {
            "status": "running",
            "started_utc": _utc_now(),
            "project_dir": str(project_dir),
            "source_project": source_project.name,
            "provider": provider,
            "template_mode": args.template_mode,
        },
    )

    try:
        scenes = generate_storyboard(input_text, provider=provider)  # includes downstream typing in template mode
        scenes = validate_scenes(scenes)

        for i, scene in enumerate(scenes):
            scene.setdefault("id", i)
            out = generate_scene_image(scene, project_dir)
            scene["image_path"] = str(out)

        if not scenes:
            raise RuntimeError("Storyboard produced zero scenes")
        if not any(Path(str(s.get("image_path") or "")).exists() for s in scenes):
            raise RuntimeError("No rendered scene images were produced")

        plan = {
            "meta": {
                "project_name": out_name,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "llm_provider": provider,
                "schema_version": SCHEMA_VERSION,
                "use_template_pipeline": True,
                "render_pipeline": "template_deterministic",
                "input_text": input_text,
                "source_project": source_project.name,
                "template_mode": args.template_mode,
            },
            "scenes": scenes,
        }
        (project_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        sheet = _write_contact_sheet(project_dir, scenes)
        _write_status(
            status_artifact,
            {
                "status": "completed",
                "updated_utc": _utc_now(),
                "project_dir": str(project_dir),
                "scene_count": len(scenes),
                "contact_sheet": str(sheet),
                "plan_path": str(project_dir / "plan.json"),
                "template_mode": args.template_mode,
            },
        )
        RUN_STATE["status"] = "completed"
        print(f"PROJECT={project_dir}")
        print(f"SCENES={len(scenes)}")
        print(f"CONTACT_SHEET={sheet}")
        print(f"STATUS_ARTIFACT={status_artifact}")
        return 0
    except Exception as exc:
        _mark_failed("exception_during_e2e_run", exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
