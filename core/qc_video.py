"""Video-specific QC helpers with minimal dependencies."""

from __future__ import annotations


def video_qc_sample_times(duration_s: float | None) -> list[tuple[str, float]]:
    """Deterministic frame sample points for animated scene visual QC."""
    if duration_s is None or duration_s <= 0.0:
        return [("start", 0.0)]
    d = max(0.0, float(duration_s))
    if d < 0.35:
        return [("start", 0.0), ("mid", max(0.0, d * 0.5))]
    return [
        ("start", 0.0),
        ("mid", max(0.0, d * 0.5)),
        ("end", max(0.0, d - 0.10)),
    ]
