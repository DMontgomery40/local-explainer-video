"""Validate agent-declared scene claims against Stage 1 _data_pack.json.

Design rule: Code may validate and execute. Code may NOT decide what scene to make.

The agent emits scene_claims.json with explicit factual claims. This module checks
those claims against the data pack. No interpretation, no inference, no scene routing.

Claim types:
  - numeric_value: exact match against data pack (tolerance for rounding)
  - directional: verify direction is correct in data
  - threshold: verify value is within/outside claimed range
  - comparative: verify percentage/comparison math
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimResult:
    scene_id: int | str
    claim_type: str
    metric: str
    passed: bool
    detail: str


@dataclass
class ClaimsQCResult:
    passed: bool
    results: list[ClaimResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _resolve_metric_in_data_pack(data_pack: dict[str, Any], metric: str, session_index: int | None = None) -> Any | None:
    """Look up a metric name in the data pack's facts and derived sections.

    The data pack has two shapes:
    - facts as a LIST of {metric, session_index, value, ...} objects (Stage 1 format)
    - facts/derived as dicts (older format)

    Tries exact match first, then case-insensitive, then substring.
    Returns the value if found, None otherwise.
    """
    facts = data_pack.get("facts", {})
    derived = data_pack.get("derived", {})

    # Handle facts as a list of fact objects (Stage 1 data pack format)
    if isinstance(facts, list):
        # Build a flat dict: "metric_sessionN" -> value
        flat: dict[str, Any] = {}
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            m = str(fact.get("metric", ""))
            si = fact.get("session_index")
            val = fact.get("value")
            if m and val is not None:
                flat[m] = val  # latest session wins for bare metric name
                if si is not None:
                    flat[f"{m}_session{si}"] = val
                    flat[f"{m}_{si}"] = val
        # Also add derived tables as flat entries
        if isinstance(derived, dict):
            for k, v in derived.items():
                if not isinstance(v, str):  # skip markdown tables
                    flat[k] = v
    else:
        # Original dict-based format
        combined: dict[str, Any] = {}
        if isinstance(facts, dict):
            combined.update(facts)
        if isinstance(derived, dict):
            combined.update(derived)
        flat = {}
        for k, v in combined.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = v2
                    flat[k2] = v2
            else:
                flat[k] = v

    if session_index is not None:
        # A requested session is an evidence boundary. Bare or substring
        # matches can belong to a different session, even with equal values.
        def normalized(key):
            return re.sub(r"_session_", "_session", re.sub(r"[- .]+", "_", key.lower()))
        requested = {normalized(f"{metric}_session{session_index}"), normalized(f"{metric}_{session_index}")}
        for key, value in flat.items():
            if normalized(key) in requested:
                return value
        return None

    # Exact match
    if metric in flat:
        return flat[metric]

    # Case-insensitive
    metric_lower = metric.lower()
    for k, v in flat.items():
        if k.lower() == metric_lower:
            return v

    # Substring match (e.g., "trail_b_session3" matches "trail_making_b.session_3")
    for k, v in flat.items():
        k_norm = k.lower().replace("-", "_").replace(" ", "_")
        m_norm = metric_lower.replace("-", "_").replace(" ", "_")
        if m_norm in k_norm or k_norm in m_norm:
            return v

    return None


def _to_float(value: Any) -> float | None:
    """Coerce a value to float if possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%").replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _validate_numeric_value(
    claim: dict[str, Any],
    data_pack: dict[str, Any],
    tolerance: float = 0.05,
) -> ClaimResult:
    """Validate a numeric_value claim: does the data pack agree with the stated value?"""
    scene_id = claim.get("scene_id", "?")
    metric = str(claim.get("metric", ""))
    claimed = _to_float(claim.get("value"))
    session_index = claim.get("session_index")

    if claimed is None:
        return ClaimResult(scene_id, "numeric_value", metric, False,
                           f"Claim has non-numeric value: {claim.get('value')}")

    actual = _resolve_metric_in_data_pack(data_pack, metric, session_index=session_index)
    if actual is None:
        return ClaimResult(scene_id, "numeric_value", metric, False,
                           f"Metric '{metric}' not found in data pack")

    actual_f = _to_float(actual)
    if actual_f is None:
        return ClaimResult(scene_id, "numeric_value", metric, False,
                           f"Data pack value for '{metric}' is non-numeric: {actual}")

    # Check within tolerance (relative for large numbers, absolute for small)
    if actual_f == 0:
        ok = abs(claimed) < 0.5
    else:
        relative_error = abs(claimed - actual_f) / abs(actual_f)
        ok = relative_error <= tolerance

    if ok:
        return ClaimResult(scene_id, "numeric_value", metric, True,
                           f"Claimed {claimed}, data pack has {actual_f}")
    else:
        return ClaimResult(scene_id, "numeric_value", metric, False,
                           f"MISMATCH: claimed {claimed}, data pack has {actual_f}")


def _validate_directional(
    claim: dict[str, Any],
    data_pack: dict[str, Any],
) -> ClaimResult:
    """Validate a directional claim: did the metric go in the stated direction?"""
    scene_id = claim.get("scene_id", "?")
    metric = str(claim.get("metric", ""))
    direction = str(claim.get("direction", "")).lower()
    from_session = claim.get("from_session")
    to_session = claim.get("to_session")

    # Try to find session-specific values
    from_key = f"{metric}_session{from_session}" if from_session else f"{metric}_session1"
    to_key = f"{metric}_session{to_session}" if to_session else f"{metric}_session3"

    from_val = _to_float(_resolve_metric_in_data_pack(data_pack, from_key))
    to_val = _to_float(_resolve_metric_in_data_pack(data_pack, to_key))

    if from_val is None or to_val is None:
        return ClaimResult(scene_id, "directional", metric, False,
                           f"Cannot resolve session values: {from_key}={from_val}, {to_key}={to_val}")

    actual_direction = "increased" if to_val > from_val else "decreased" if to_val < from_val else "unchanged"
    ok = (direction in ("increased", "up") and actual_direction == "increased") or \
         (direction in ("decreased", "down") and actual_direction == "decreased") or \
         (direction == actual_direction)

    if ok:
        return ClaimResult(scene_id, "directional", metric, True,
                           f"Claimed {direction}: {from_val} -> {to_val} (correct)")
    else:
        return ClaimResult(scene_id, "directional", metric, False,
                           f"MISMATCH: claimed {direction}, actual {actual_direction}: {from_val} -> {to_val}")


def _validate_threshold(
    claim: dict[str, Any],
    data_pack: dict[str, Any],
) -> ClaimResult:
    """Validate a threshold claim: is the value within/outside the stated range?"""
    scene_id = claim.get("scene_id", "?")
    metric = str(claim.get("metric", ""))
    threshold_claim = str(claim.get("claim", "")).strip().lower()
    if threshold_claim not in {"within", "above", "below", "outside"}:
        return ClaimResult(scene_id, "threshold", metric, False,
                           f"Unsupported threshold predicate: {claim.get('claim')!r}")
    range_val = claim.get("range", [])

    actual = _to_float(_resolve_metric_in_data_pack(data_pack, metric))
    if actual is None:
        return ClaimResult(scene_id, "threshold", metric, False,
                           f"Metric '{metric}' not found in data pack")

    if not isinstance(range_val, (list, tuple)) or len(range_val) != 2:
        return ClaimResult(scene_id, "threshold", metric, False,
                           f"Invalid range: {range_val}")

    low, high = _to_float(range_val[0]), _to_float(range_val[1])
    if (low is None or high is None or not math.isfinite(low)
            or not math.isfinite(high) or low > high or not math.isfinite(actual)):
        return ClaimResult(scene_id, "threshold", metric, False,
                           f"Non-numeric range bounds: {range_val}")

    in_range = low <= actual <= high

    if threshold_claim == "within":
        ok = in_range
        detail = f"Claimed within [{low}, {high}], actual={actual}, in_range={in_range}"
    elif threshold_claim == "above":
        ok = actual > high
        detail = f"Claimed above {high}, actual={actual}"
    elif threshold_claim == "below":
        ok = actual < low
        detail = f"Claimed below {low}, actual={actual}"
    else:
        ok = not in_range
        detail = f"Claimed outside [{low}, {high}], actual={actual}, in_range={in_range}"

    return ClaimResult(scene_id, "threshold", metric, ok, detail)


def _validate_comparative(
    claim: dict[str, Any],
    data_pack: dict[str, Any],
    tolerance: float = 2.0,
) -> ClaimResult:
    """Validate a comparative claim: is the percentage/comparison correct?"""
    scene_id = claim.get("scene_id", "?")
    metric = str(claim.get("metric", ""))
    claimed_pct = _to_float(claim.get("value"))

    if claimed_pct is None:
        return ClaimResult(scene_id, "comparative", metric, False,
                           f"Non-numeric comparison value: {claim.get('value')}")

    # Try to derive the comparison from session values
    base_metric = metric.replace("_improvement", "").replace("_change", "").replace("_reduction", "")
    from_val = _to_float(_resolve_metric_in_data_pack(data_pack, f"{base_metric}_session1"))
    to_val = _to_float(_resolve_metric_in_data_pack(data_pack, f"{base_metric}_session3"))

    if from_val is None or to_val is None:
        # Try to find the derived percentage directly
        direct = _to_float(_resolve_metric_in_data_pack(data_pack, metric))
        if direct is not None:
            ok = abs(claimed_pct - direct) <= tolerance
            return ClaimResult(scene_id, "comparative", metric, ok,
                               f"Claimed {claimed_pct}%, data pack has {direct}%")
        return ClaimResult(scene_id, "comparative", metric, False,
                           f"Cannot find base values for comparison: {base_metric}")

    if from_val == 0:
        return ClaimResult(scene_id, "comparative", metric, False,
                           f"Cannot compute percentage: baseline is 0")

    actual_pct = ((to_val - from_val) / abs(from_val)) * 100
    ok = abs(claimed_pct - actual_pct) <= tolerance

    if ok:
        return ClaimResult(scene_id, "comparative", metric, True,
                           f"Claimed {claimed_pct}%, computed {actual_pct:.1f}% ({from_val} -> {to_val})")
    else:
        return ClaimResult(scene_id, "comparative", metric, False,
                           f"MISMATCH: claimed {claimed_pct}%, computed {actual_pct:.1f}% ({from_val} -> {to_val})")


_VALIDATORS = {
    "numeric_value": _validate_numeric_value,
    "directional": _validate_directional,
    "threshold": _validate_threshold,
    "comparative": _validate_comparative,
}


def validate_claims(
    claims: list[dict[str, Any]],
    data_pack: dict[str, Any],
) -> ClaimsQCResult:
    """Validate a list of agent-declared claims against the data pack.

    Args:
        claims: List of claim dicts, each with at least {type, metric, ...}
        data_pack: Stage 1 _data_pack.json contents

    Returns:
        ClaimsQCResult with pass/fail and per-claim details
    """
    results: list[ClaimResult] = []
    errors: list[str] = []
    warnings: list[str] = []

    for claim in claims:
        claim_type = str(claim.get("type", "")).strip().lower()
        validator = _VALIDATORS.get(claim_type)

        if validator is None:
            result = ClaimResult(
                scene_id=claim.get("scene_id", "?"),
                claim_type=claim_type,
                metric=str(claim.get("metric", "")),
                passed=False,
                detail=f"Unsupported claim type: {claim_type!r}",
            )
        else:
            result = validator(claim, data_pack)
        results.append(result)

        if not result.passed:
            errors.append(f"Scene {result.scene_id} [{result.claim_type}] {result.metric}: {result.detail}")

    return ClaimsQCResult(
        passed=len(errors) == 0,
        results=results,
        errors=errors,
        warnings=warnings,
    )


def validate_claims_file(
    claims_path: Path,
    data_pack: dict[str, Any],
) -> ClaimsQCResult:
    """Load scene_claims.json and validate against data pack."""
    def invalid():
        return ClaimsQCResult(passed=False, errors=["Malformed scene claims container"])

    try:
        raw = json.loads(claims_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return invalid()
    all_claims: list[dict[str, Any]] = []
    if isinstance(raw, list):
        if not all(isinstance(claim, dict) for claim in raw):
            return invalid()
        all_claims = raw
    elif isinstance(raw, dict):
        if "scenes" in raw:
            blocks = raw["scenes"]
        elif "claims" in raw:
            blocks = [raw]
        else:
            return invalid()
        if not isinstance(blocks, list):
            return invalid()
        for block in blocks:
            if not isinstance(block, dict) or not isinstance(block.get("claims"), list):
                return invalid()
            for claim in block["claims"]:
                if not isinstance(claim, dict):
                    return invalid()
                all_claims.append({"scene_id": block.get("scene_id", "?"), **claim})
    else:
        return invalid()
    return validate_claims(all_claims, data_pack)


def write_claims_report(result: ClaimsQCResult, output_path: Path) -> None:
    """Write the QC report to disk."""
    report = {
        "passed": result.passed,
        "total_claims": len(result.results),
        "failed_claims": sum(1 for r in result.results if not r.passed),
        "errors": result.errors,
        "warnings": result.warnings,
        "results": [asdict(r) for r in result.results],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
