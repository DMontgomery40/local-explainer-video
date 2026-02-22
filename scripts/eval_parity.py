#!/usr/bin/env python3
"""Evaluate weighted deterministic parity confidence against release threshold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("artifacts/parity_eval/latest/parity_report.json")

WEIGHTS = {
    "qeeg_placement": 0.20,
    "cinematic_quality": 0.30,
    "text_blend": 0.20,
    "indistinguishable_vibe": 0.20,
    "runtime_window": 0.10,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _runtime_score(runtime_minutes: float) -> float:
    # Requirement is strict: full score only when runtime is between 5 and 7 minutes.
    return 100.0 if 5.0 <= float(runtime_minutes) <= 7.0 else 0.0


def _load_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics file must be a JSON object")
    return payload


def evaluate(metrics: dict[str, Any]) -> dict[str, Any]:
    qeeg = _clamp_score(float(metrics.get("qeeg_placement", 0.0)))
    cinematic = _clamp_score(float(metrics.get("cinematic_quality", 0.0)))
    text_blend = _clamp_score(float(metrics.get("text_blend", 0.0)))
    vibe = _clamp_score(float(metrics.get("indistinguishable_vibe", 0.0)))
    runtime_minutes = float(metrics.get("runtime_minutes", 0.0))
    runtime_window_score = _runtime_score(runtime_minutes)

    weighted = (
        (WEIGHTS["qeeg_placement"] * qeeg)
        + (WEIGHTS["cinematic_quality"] * cinematic)
        + (WEIGHTS["text_blend"] * text_blend)
        + (WEIGHTS["indistinguishable_vibe"] * vibe)
        + (WEIGHTS["runtime_window"] * runtime_window_score)
    )

    return {
        "qeeg_placement": qeeg,
        "cinematic_quality": cinematic,
        "text_blend": text_blend,
        "indistinguishable_vibe": vibe,
        "runtime_minutes": runtime_minutes,
        "runtime_window_score": runtime_window_score,
        "weights": WEIGHTS,
        "confidence_score": round(weighted, 3),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate parity confidence score.")
    parser.add_argument("--metrics-path", default="")
    parser.add_argument("--qeeg-placement", type=float)
    parser.add_argument("--cinematic-quality", type=float)
    parser.add_argument("--text-blend", type=float)
    parser.add_argument("--indistinguishable-vibe", type=float)
    parser.add_argument("--runtime-minutes", type=float)
    parser.add_argument("--require-confidence", type=float, default=84.7)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    metrics_path = Path(args.metrics_path) if str(args.metrics_path).strip() else None
    metrics = _load_metrics(metrics_path)

    overrides = {
        "qeeg_placement": args.qeeg_placement,
        "cinematic_quality": args.cinematic_quality,
        "text_blend": args.text_blend,
        "indistinguishable_vibe": args.indistinguishable_vibe,
        "runtime_minutes": args.runtime_minutes,
    }
    for key, value in overrides.items():
        if value is not None:
            metrics[key] = float(value)

    required = [
        "qeeg_placement",
        "cinematic_quality",
        "text_blend",
        "indistinguishable_vibe",
        "runtime_minutes",
    ]
    missing = [k for k in required if k not in metrics]
    if missing:
        print(json.dumps({"status": "failed", "error": f"missing metrics: {', '.join(missing)}"}, indent=2))
        return 2

    result = evaluate(metrics)
    threshold = float(args.require_confidence)
    passed = float(result["confidence_score"]) >= threshold

    report = {
        "created_utc": _utc_now(),
        "threshold": threshold,
        "passed": passed,
        "inputs": metrics,
        "result": result,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "passed" if passed else "failed",
                "confidence_score": result["confidence_score"],
                "threshold": threshold,
                "output_path": str(output_path.resolve()),
            },
            indent=2,
        )
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
