#!/usr/bin/env python3
"""
CLI entrypoint for the QC + publish pipeline.

Example:
  python3.10 qc_publish.py --project 09-05-1954-0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.qc_publish import (
    QCPublishConfig,
    default_cliproxy_api_key,
    default_cliproxy_url,
    default_qeeg_analysis_dir,
    default_qeeg_backend_url,
    infer_patient_id,
    qc_and_publish_project,
)


def load_plan(project_dir: Path) -> dict:
    plan_path = project_dir / "plan.json"
    return json.loads(plan_path.read_text(encoding="utf-8"))


def save_plan(project_dir: Path, plan: dict) -> None:
    plan_path = project_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QC + publish for a local-explainer-video project.")
    parser.add_argument(
        "--project",
        required=True,
        help="Project folder name under ./projects/ (or an absolute/relative path to a project dir).",
    )
    parser.add_argument("--patient-id", default="", help="Override patient id (MM-DD-YYYY-N).")
    parser.add_argument("--qeeg-dir", default=str(default_qeeg_analysis_dir()), help="Path to qEEG-analysis repo.")
    parser.add_argument("--backend-url", default=default_qeeg_backend_url(), help="qEEG Council backend URL.")
    parser.add_argument("--cliproxy-url", default=default_cliproxy_url(), help="CLIProxyAPI base URL.")
    parser.add_argument("--cliproxy-api-key", default=default_cliproxy_api_key(), help="CLIProxyAPI key.")
    parser.add_argument("--output", default="final_video.mp4", help="Output filename within the project directory.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-visual-passes", type=int, default=5)
    parser.add_argument(
        "--auto-fix-images",
        action="store_true",
        help="Apply deterministic slide text fixes via image-edit (otherwise report issues only).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    proj_arg = Path(args.project)
    project_dir = proj_arg if proj_arg.exists() else (root / "projects" / args.project)
    if not project_dir.exists():
        raise SystemExit(f"Project directory not found: {project_dir}")

    patient_id = (args.patient_id or "").strip() or infer_patient_id(project_dir.name) or ""
    if not patient_id:
        raise SystemExit("Unable to infer patient id; provide --patient-id MM-DD-YYYY-N")

    plan = load_plan(project_dir)
    cfg = QCPublishConfig(
        qeeg_dir=Path(args.qeeg_dir).expanduser(),
        backend_url=str(args.backend_url).rstrip("/"),
        cliproxy_url=str(args.cliproxy_url).rstrip("/"),
        cliproxy_api_key=str(args.cliproxy_api_key or ""),
        max_visual_passes=int(args.max_visual_passes),
        auto_fix_images=bool(args.auto_fix_images),
        fps=int(args.fps),
        output_filename=str(args.output),
    )

    def log(msg: str) -> None:
        print(msg, flush=True)

    updated_plan, summary = qc_and_publish_project(
        project_dir=project_dir,
        plan=plan,
        patient_id=patient_id,
        config=cfg,
        log=log,
    )
    save_plan(project_dir, updated_plan)

    print("\n---\nQC + Publish complete:")
    print(f"patient_id: {summary.patient_id}")
    print(f"run_id: {summary.run_id}")
    print(f"video_path: {summary.video_path}")
    print(f"portal_copy_path: {summary.portal_copy_path}")
    print(f"backend_upload_ok: {summary.backend_upload_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
