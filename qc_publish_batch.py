#!/usr/bin/env python3
"""
Batch runner for the QC + Publish pipeline.

Example:
  python3.10 qc_publish_batch.py --auto-fix-images
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from core.qc_publish import (
    QCPublishConfig,
    QCPublishError,
    default_cliproxy_api_key,
    default_cliproxy_url,
    default_qeeg_analysis_dir,
    default_qeeg_backend_url,
    infer_patient_id,
    qc_and_publish_project,
)


_SUFFIX_RE = re.compile(r"__([0-9]+)$")


def _parse_project_version(name: str) -> int:
    m = _SUFFIX_RE.search(name or "")
    if not m:
        return 1
    try:
        return max(1, int(m.group(1)))
    except Exception:
        return 1


def load_plan(project_dir: Path) -> dict:
    plan_path = project_dir / "plan.json"
    return json.loads(plan_path.read_text(encoding="utf-8"))


def save_plan(project_dir: Path, plan: dict) -> None:
    plan_path = project_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def discover_latest_patient_projects(projects_dir: Path) -> list[tuple[str, Path]]:
    """
    Return [(patient_id, project_dir)] choosing the latest version per patient.

    Latest is selected by:
    1) Highest __NN suffix (base folder counts as 1)
    2) Newest mtime as tie-breaker
    """
    best: dict[str, tuple[int, float, Path]] = {}
    for child in projects_dir.iterdir():
        if not child.is_dir():
            continue
        patient_id = infer_patient_id(child.name)
        if not patient_id:
            continue
        version = _parse_project_version(child.name)
        try:
            mtime = child.stat().st_mtime
        except Exception:
            mtime = 0.0

        current = best.get(patient_id)
        candidate = (version, mtime, child)
        if current is None or candidate[:2] > current[:2]:
            best[patient_id] = candidate

    chosen = [(pid, info[2]) for pid, info in best.items()]
    # Process newest projects first to match typical workflow.
    return sorted(chosen, key=lambda x: x[1].stat().st_mtime if x[1].exists() else 0.0, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch QC + publish over valid patient projects.")
    parser.add_argument("--projects-dir", default="projects", help="Path to the projects directory.")
    parser.add_argument("--qeeg-dir", default=str(default_qeeg_analysis_dir()), help="Path to qEEG-analysis repo.")
    parser.add_argument("--backend-url", default=default_qeeg_backend_url(), help="qEEG Council backend URL.")
    parser.add_argument("--cliproxy-url", default=default_cliproxy_url(), help="CLIProxyAPI base URL.")
    parser.add_argument("--cliproxy-api-key", default=default_cliproxy_api_key(), help="CLIProxyAPI key.")
    parser.add_argument("--output", default="final_video.mp4", help="Output filename within each project directory.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-visual-passes", type=int, default=5)
    parser.add_argument(
        "--auto-fix-images",
        action="store_true",
        help="Apply deterministic slide text fixes via image-edit (otherwise report issues only).",
    )
    parser.add_argument(
        "--image-edit-model",
        default="",
        help='Override image edit model (e.g., "qwen-image-edit-max" for DashScope or "qwen/qwen-image-edit-2511" for Replicate).',
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch as soon as a project fails QC/publish.",
    )
    parser.add_argument(
        "--results-json",
        default="qc_batch_results.json",
        help="Write a JSON summary of outcomes to this path.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    projects_dir = (root / args.projects_dir).resolve() if not Path(args.projects_dir).is_absolute() else Path(args.projects_dir).resolve()
    if not projects_dir.exists():
        raise SystemExit(f"Projects dir not found: {projects_dir}")

    projects = discover_latest_patient_projects(projects_dir)
    if not projects:
        print(f"No valid patient projects found under: {projects_dir}")
        return 0

    print(f"Found {len(projects)} patient project(s) (latest version per patient).")

    results: list[dict] = []
    started_at = time.time()

    for idx, (patient_id, project_dir) in enumerate(projects, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(projects)}] {patient_id} -> {project_dir.name}")

        try:
            plan = load_plan(project_dir)
        except Exception as e:
            msg = f"Failed loading plan.json: {e}"
            print(msg)
            results.append({"patient_id": patient_id, "project": str(project_dir), "ok": False, "error": msg})
            if args.stop_on_error:
                break
            continue

        cfg = QCPublishConfig(
            qeeg_dir=Path(args.qeeg_dir).expanduser(),
            backend_url=str(args.backend_url).rstrip("/"),
            cliproxy_url=str(args.cliproxy_url).rstrip("/"),
            cliproxy_api_key=str(args.cliproxy_api_key or ""),
            max_visual_passes=int(args.max_visual_passes),
            auto_fix_images=bool(args.auto_fix_images),
            fps=int(args.fps),
            output_filename=str(args.output),
            image_edit_model=(str(args.image_edit_model).strip() or None),
        )

        def log(msg: str) -> None:
            print(msg, flush=True)

        t0 = time.time()
        try:
            updated_plan, summary = qc_and_publish_project(
                project_dir=project_dir,
                plan=plan,
                patient_id=patient_id,
                config=cfg,
                log=log,
            )
            save_plan(project_dir, updated_plan)
            elapsed_s = round(time.time() - t0, 1)
            print(f"OK ({elapsed_s}s): portal_copy={summary.portal_copy_path}, backend_upload_ok={summary.backend_upload_ok}")
            summary_dict = asdict(summary)
            summary_dict["video_path"] = str(summary.video_path)
            summary_dict["portal_copy_path"] = str(summary.portal_copy_path) if summary.portal_copy_path else None
            results.append(
                {
                    "patient_id": patient_id,
                    "project": str(project_dir),
                    "ok": True,
                    "elapsed_s": elapsed_s,
                    "summary": summary_dict,
                }
            )
        except QCPublishError as e:
            elapsed_s = round(time.time() - t0, 1)
            print(f"FAILED ({elapsed_s}s): {e}")
            results.append({"patient_id": patient_id, "project": str(project_dir), "ok": False, "elapsed_s": elapsed_s, "error": str(e)})
            if args.stop_on_error:
                break
        except Exception as e:
            elapsed_s = round(time.time() - t0, 1)
            print(f"FAILED ({elapsed_s}s): unexpected error: {e}")
            results.append(
                {"patient_id": patient_id, "project": str(project_dir), "ok": False, "elapsed_s": elapsed_s, "error": f"Unexpected error: {e}"}
            )
            if args.stop_on_error:
                break

    total_s = round(time.time() - started_at, 1)
    ok_count = sum(1 for r in results if r.get("ok"))
    fail_count = len(results) - ok_count

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "projects_dir": str(projects_dir),
        "auto_fix_images": bool(args.auto_fix_images),
        "elapsed_s": total_s,
        "ok": ok_count,
        "failed": fail_count,
        "results": results,
    }

    results_path = (root / args.results_json).resolve() if not Path(args.results_json).is_absolute() else Path(args.results_json).resolve()
    try:
        results_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n" + "-" * 80)
        print(f"Wrote batch results: {results_path}")
    except Exception as e:
        print(f"Failed writing results JSON ({results_path}): {e}")

    print(f"Done. ok={ok_count}, failed={fail_count}, elapsed={total_s}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
