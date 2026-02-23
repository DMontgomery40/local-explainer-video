#!/usr/bin/env python3
"""Build a deterministic template pack from existing project images using Qwen image edit."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.image_gen import edit_image
from core.template_pack import (
    TemplatePackEntry,
    classify_scene_archetype,
    default_template_manifest_path,
    write_template_manifest,
)

BRAIN_MARKER = "[[BLENDER_QEEG]]"


@dataclass(frozen=True)
class SceneCandidate:
    project: str
    scene_id: int
    image_path: Path
    archetype: str


def _looks_like_brain_scene(scene: dict[str, Any]) -> bool:
    backend = str(scene.get("render_backend") or "").strip().lower()
    if backend == "blender":
        return True

    text = " ".join(
        str(scene.get(k) or "")
        for k in ("title", "subtitle", "visual_prompt", "narration", "scene_type")
    ).lower()

    if BRAIN_MARKER in text:
        return True

    keywords = (
        "electrode",
        "topomap",
        "topography",
        "coherence",
        "connectivity",
        "brain map",
        "10-20",
        "c3",
        "cz",
        "pz",
        "fp1",
        "fp2",
    )
    return any(k in text for k in keywords)


def _resolve_image_path(project_dir: Path, scene: dict[str, Any], index: int) -> Path | None:
    raw = scene.get("image_path")
    if raw:
        p = Path(str(raw))
        if not p.is_absolute():
            p = (project_dir / p).resolve()
        if p.exists():
            return p

    fallback = project_dir / "images" / f"scene_{index:03d}.png"
    return fallback if fallback.exists() else None


def _iter_candidates(projects_dir: Path) -> list[SceneCandidate]:
    rows: list[SceneCandidate] = []
    for plan_path in sorted(projects_dir.glob("*/plan.json")):
        project_dir = plan_path.parent
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        scenes = plan.get("scenes")
        if not isinstance(scenes, list):
            continue

        for idx, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue
            if _looks_like_brain_scene(scene):
                continue

            image_path = _resolve_image_path(project_dir, scene, idx)
            if not image_path:
                continue

            scene_id = int(scene.get("id") or idx)
            rows.append(
                SceneCandidate(
                    project=project_dir.name,
                    scene_id=scene_id,
                    image_path=image_path,
                    archetype=classify_scene_archetype(scene),
                )
            )
    return rows


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def _remove_text_prompt(archetype: str) -> str:
    return (
        "Create a clean, text-free background template from this slide. "
        "Remove every word, number, symbol, chart label, and axis text while preserving the scene composition, "
        "lighting, perspective, and non-text graphical elements. "
        "Fill former text regions naturally so they are empty and reusable. "
        "Do not add any new text, logos, or watermarks. "
        f"Keep the overall style suitable for a {archetype.replace('_', ' ')} panel in a clinical explainer video."
    )


def build_template_pack(
    *,
    projects_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    model: str,
    fallback_model: str,
    limit_per_archetype: int,
    dry_run: bool,
    overwrite: bool,
    continue_on_error: bool,
) -> None:
    candidates = _iter_candidates(projects_dir)
    print(f"Found {len(candidates)} non-brain scene candidates")

    grouped: dict[str, list[SceneCandidate]] = {}
    for c in candidates:
        grouped.setdefault(c.archetype, []).append(c)

    selected: list[SceneCandidate] = []
    for archetype, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda r: (r.project, r.scene_id, str(r.image_path)))
        picked = ordered[:limit_per_archetype]
        selected.extend(picked)
        print(f"  {archetype}: selected {len(picked)}/{len(rows)}")

    entries: list[TemplatePackEntry] = []
    for i, row in enumerate(selected, start=1):
        archetype_dir = output_dir / row.archetype
        archetype_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{_sanitize_name(row.project)}_scene_{row.scene_id:03d}"
        out_path = archetype_dir / f"{stem}.png"
        template_id = f"{row.archetype}__{stem}"

        render_ok = False

        if out_path.exists() and not overwrite:
            print(f"[{i}/{len(selected)}] keep existing {out_path}")
            render_ok = True
        elif dry_run:
            print(f"[{i}/{len(selected)}] [dry-run] edit -> {out_path}")
            render_ok = True
        else:
            print(f"[{i}/{len(selected)}] editing {row.image_path} -> {out_path}")
            try:
                edit_image(
                    prompt=_remove_text_prompt(row.archetype),
                    input_image_path=row.image_path,
                    output_path=out_path,
                    model=model,
                    prompt_extend=False,
                    negative_prompt="text, words, letters, numbers, logos, watermark",
                    watermark=False,
                )
                render_ok = True
            except Exception as e:
                text = str(e)
                if fallback_model and fallback_model != model and model.startswith("qwen-image-edit"):
                    print(
                        "    primary model failed "
                        f"({text[:120]}); retrying with fallback model: {fallback_model}"
                    )
                    try:
                        edit_image(
                            prompt=_remove_text_prompt(row.archetype),
                            input_image_path=row.image_path,
                            output_path=out_path,
                            model=fallback_model,
                            prompt_extend=False,
                            negative_prompt="text, words, letters, numbers, logos, watermark",
                            watermark=False,
                        )
                        render_ok = True
                    except Exception as fallback_error:
                        if continue_on_error:
                            print(f"    fallback failed; skipping template: {fallback_error}")
                            render_ok = False
                        else:
                            raise
                else:
                    if continue_on_error:
                        print(f"    render failed; skipping template: {e}")
                        render_ok = False
                    else:
                        raise

        if render_ok:
            entries.append(
                TemplatePackEntry(
                    template_id=template_id,
                    archetype=row.archetype,
                    path=out_path.resolve(),
                    source_project=row.project,
                    source_scene_id=row.scene_id,
                )
            )
        else:
            print(f"    skipped manifest entry for {out_path} (not rendered)")

    write_template_manifest(manifest_path, entries)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Total templates in manifest: {len(entries)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-dir", type=Path, default=Path("projects"))
    parser.add_argument("--output-dir", type=Path, default=Path("templates/patient_pack"))
    parser.add_argument("--manifest", type=Path, default=default_template_manifest_path())
    parser.add_argument("--model", type=str, default="qwen-image-edit-max")
    parser.add_argument(
        "--fallback-model",
        type=str,
        default="qwen/qwen-image-edit-2511",
        help="Fallback edit model when primary model access is denied.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first render error instead of skipping failed templates.",
    )
    parser.add_argument("--limit-per-archetype", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_template_pack(
        projects_dir=args.projects_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        model=args.model,
        fallback_model=args.fallback_model,
        limit_per_archetype=max(1, int(args.limit_per_archetype)),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        continue_on_error=not bool(args.fail_fast),
    )


if __name__ == "__main__":
    main()
