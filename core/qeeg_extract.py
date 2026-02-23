"""qEEG Stage-1 data extraction helpers for Blender scene rendering."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


ELECTRODE_LABELS: tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "Fpz",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
    "Oz",
    "A1",
    "A2",
    "T7",
    "T8",
    "P7",
    "P8",
)

_LABEL_ALIASES: dict[str, str] = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FPZ": "Fpz",
    "F7": "F7",
    "F3": "F3",
    "FZ": "Fz",
    "F4": "F4",
    "F8": "F8",
    "T3": "T3",
    "C3": "C3",
    "CZ": "Cz",
    "C4": "C4",
    "T4": "T4",
    "T5": "T5",
    "P3": "P3",
    "PZ": "Pz",
    "P4": "P4",
    "T6": "T6",
    "O1": "O1",
    "O2": "O2",
    "OZ": "Oz",
    "A1": "A1",
    "A2": "A2",
    # Common modern aliases.
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

_GENERIC_HINT_TERMS: tuple[str, ...] = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "coherence",
    "connectivity",
    "zscore",
    "z_score",
    "z",
    "power",
    "amplitude",
    "electrode",
    "topography",
    "topomap",
    "band",
    "metric",
    "session",
)

_NUMERIC_VALUE_KEYS: tuple[str, ...] = (
    "value",
    "z",
    "zscore",
    "z_score",
    "score",
    "weight",
    "strength",
    "magnitude",
    "coherence",
    "uv",
    "power",
    "amplitude",
)

_EDGE_A_KEYS: tuple[str, ...] = (
    "a",
    "from",
    "source",
    "src",
    "site_a",
    "electrode_a",
    "channel_a",
    "node_a",
    "node1",
    "ch1",
    "channel1",
)

_EDGE_B_KEYS: tuple[str, ...] = (
    "b",
    "to",
    "target",
    "dst",
    "site_b",
    "electrode_b",
    "channel_b",
    "node_b",
    "node2",
    "ch2",
    "channel2",
)

_EDGE_VALUE_KEYS: tuple[str, ...] = (
    "value",
    "coherence",
    "weight",
    "strength",
    "magnitude",
    "score",
    "z",
    "zscore",
    "z_score",
    "r",
    "corr",
    "connectivity",
)

_PATH_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_PAIR_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]{0,3})\s*[-:/>]+\s*([A-Za-z][A-Za-z0-9]{0,3})\s*$")


@dataclass(frozen=True)
class QEEGExtractConfig:
    """Optional path + scene hints for extraction."""

    electrode_path: str | None = None
    coherence_path: str | None = None
    session_index: int | None = None
    band: str | None = None
    metric: str | None = None


def normalize_electrode_label(raw: Any) -> str | None:
    """Normalize a channel label to canonical 10-20 casing."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    compact = re.sub(r"[\s_]", "", text).upper()
    return _LABEL_ALIASES.get(compact)


def extract_qeeg_visual_data(
    data_pack: Mapping[str, Any] | dict[str, Any],
    *,
    scene: Mapping[str, Any] | None = None,
    config: QEEGExtractConfig | Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """
    Extract electrode values + coherence edges from Stage-1 data pack.

    Strategy:
    1) Optional config paths (exact extraction if available)
    2) Required heuristic scan across nested JSON
    """
    source = data_pack if isinstance(data_pack, Mapping) else {}
    merged_cfg = _merge_configs(_coerce_config(_scene_extract_cfg(scene)), _coerce_config(config))
    hint_terms = _build_hint_terms(scene=scene, config=merged_cfg)

    electrode_values: dict[str, float] = {}
    coherence_edges: list[dict[str, float | str]] = []

    if merged_cfg.electrode_path:
        node = _resolve_path(source, merged_cfg.electrode_path)
        if node is not None:
            electrode_values = _coerce_electrode_map(node)

    if merged_cfg.coherence_path:
        node = _resolve_path(source, merged_cfg.coherence_path)
        if node is not None:
            coherence_edges = _coerce_coherence_edges(node)

    if not electrode_values:
        electrode_values = _extract_electrodes_heuristic(source, hint_terms=hint_terms, session_index=merged_cfg.session_index)
    if not coherence_edges:
        coherence_edges = _extract_coherence_heuristic(source, hint_terms=hint_terms, session_index=merged_cfg.session_index)

    return electrode_values, coherence_edges


def _scene_extract_cfg(scene: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(scene, Mapping):
        return None
    raw = scene.get("qeeg_extract")
    return raw if isinstance(raw, Mapping) else None


def _coerce_config(raw: QEEGExtractConfig | Mapping[str, Any] | None) -> QEEGExtractConfig:
    if isinstance(raw, QEEGExtractConfig):
        return raw
    if not isinstance(raw, Mapping):
        return QEEGExtractConfig()
    session_index: int | None = _parse_int(raw.get("session_index"))
    return QEEGExtractConfig(
        electrode_path=str(raw.get("electrode_path") or "").strip() or None,
        coherence_path=str(raw.get("coherence_path") or "").strip() or None,
        session_index=session_index,
        band=str(raw.get("band") or "").strip() or None,
        metric=str(raw.get("metric") or "").strip() or None,
    )


def _merge_configs(base: QEEGExtractConfig, override: QEEGExtractConfig) -> QEEGExtractConfig:
    return QEEGExtractConfig(
        electrode_path=override.electrode_path or base.electrode_path,
        coherence_path=override.coherence_path or base.coherence_path,
        session_index=override.session_index if override.session_index is not None else base.session_index,
        band=override.band or base.band,
        metric=override.metric or base.metric,
    )


def _build_hint_terms(*, scene: Mapping[str, Any] | None, config: QEEGExtractConfig) -> set[str]:
    terms = set(_GENERIC_HINT_TERMS)
    for raw in (config.band, config.metric):
        if isinstance(raw, str) and raw.strip():
            terms.update(_PATH_TOKEN_RE.findall(raw.lower()))

    if isinstance(scene, Mapping):
        for key in ("title", "visual_prompt", "narration", "band", "metric"):
            value = scene.get(key)
            if isinstance(value, str):
                terms.update(_PATH_TOKEN_RE.findall(value.lower()))
    return {t for t in terms if len(t) >= 2}


def _extract_electrodes_heuristic(
    data_pack: Mapping[str, Any],
    *,
    hint_terms: set[str],
    session_index: int | None,
) -> dict[str, float]:
    best_score = float("-inf")
    best_map: dict[str, float] = {}

    for path, node, node_session_index in _walk_nodes_with_session(data_pack):
        if not _session_scope_matches(requested=session_index, scoped=node_session_index):
            continue
        if not isinstance(node, Mapping):
            continue
        cur_map = _coerce_electrode_map(node)
        if len(cur_map) < 3:
            continue
        score = float(len(cur_map) * 10) + _path_relevance(path, hint_terms)
        if score > best_score:
            best_score = score
            best_map = cur_map

    if best_map:
        return best_map
    return _extract_electrodes_from_facts(data_pack, session_index=session_index, hint_terms=hint_terms)


def _extract_coherence_heuristic(
    data_pack: Mapping[str, Any],
    *,
    hint_terms: set[str],
    session_index: int | None,
) -> list[dict[str, float | str]]:
    best_score = float("-inf")
    best_edges: list[tuple[str, str, float]] = []

    for path, node, node_session_index in _walk_nodes_with_session(data_pack):
        if not _session_scope_matches(requested=session_index, scoped=node_session_index):
            continue
        cur_edges = _coerce_coherence_edge_tuples(node, session_index=session_index)
        if not cur_edges:
            continue
        score = float(len(cur_edges) * 5) + _path_relevance(path, hint_terms)
        if any(tok in {"coherence", "connectivity"} for tok in path):
            score += 3.0
        if score > best_score:
            best_score = score
            best_edges = cur_edges

    if not best_edges:
        best_edges = _extract_coherence_from_facts(data_pack, session_index=session_index, hint_terms=hint_terms)
    return _dedupe_edges(best_edges)


def _extract_electrodes_from_facts(
    data_pack: Mapping[str, Any],
    *,
    session_index: int | None,
    hint_terms: set[str],
) -> dict[str, float]:
    facts = data_pack.get("facts")
    if not isinstance(facts, list):
        return {}

    values: dict[str, tuple[float, float]] = {}
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        if session_index is not None:
            fact_session = _parse_int(fact.get("session_index"))
            if fact_session is None or fact_session != session_index:
                continue
        label = normalize_electrode_label(fact.get("site") or fact.get("electrode") or fact.get("channel"))
        if not label:
            continue
        value = _first_numeric_value(fact, _NUMERIC_VALUE_KEYS)
        if value is None:
            continue
        score = 1.0 + _path_relevance(_fact_tokens(fact), hint_terms)
        prior = values.get(label)
        if prior is None or score >= prior[1]:
            values[label] = (value, score)
    return {label: payload[0] for label, payload in sorted(values.items())}


def _extract_coherence_from_facts(
    data_pack: Mapping[str, Any],
    *,
    session_index: int | None,
    hint_terms: set[str],
) -> list[tuple[str, str, float]]:
    facts = data_pack.get("facts")
    if not isinstance(facts, list):
        return []

    found: list[tuple[str, str, float, float]] = []
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        if session_index is not None:
            fact_session = _parse_int(fact.get("session_index"))
            if fact_session is None or fact_session != session_index:
                continue

        value = _first_numeric_value(fact, _EDGE_VALUE_KEYS)
        if value is None:
            continue

        labels: list[str] = []
        for key, raw in fact.items():
            if not isinstance(key, str):
                continue
            lk = key.lower()
            if not any(t in lk for t in ("site", "channel", "electrode", "node", "from", "to", "pair")):
                continue
            if isinstance(raw, str):
                a, b = _parse_pair_labels(raw)
                if a and b:
                    labels.extend([a, b])
                else:
                    norm = normalize_electrode_label(raw)
                    if norm:
                        labels.append(norm)
        uniq = []
        for label in labels:
            if label not in uniq:
                uniq.append(label)
        if len(uniq) < 2:
            continue
        a, b = uniq[0], uniq[1]
        if a == b:
            continue
        score = 1.0 + _path_relevance(_fact_tokens(fact), hint_terms)
        found.append((a, b, float(value), score))

    best: dict[tuple[str, str], tuple[float, float]] = {}
    for a, b, value, score in found:
        k = tuple(sorted((a, b)))
        cur = best.get(k)
        if cur is None or score >= cur[1]:
            best[k] = (value, score)
    return [(a, b, payload[0]) for (a, b), payload in sorted(best.items())]


def _walk_nodes_with_session(
    node: Any,
    path: tuple[str, ...] = (),
    active_session_index: int | None = None,
) -> Sequence[tuple[tuple[str, ...], Any, int | None]]:
    node_session_index = active_session_index
    if isinstance(node, Mapping):
        parsed = _parse_int(node.get("session_index"))
        if parsed is not None:
            node_session_index = parsed

    out: list[tuple[tuple[str, ...], Any, int | None]] = [(path, node, node_session_index)]
    if isinstance(node, Mapping):
        for key, value in node.items():
            out.extend(
                _walk_nodes_with_session(
                    value,
                    path + (_tokenize_key(key),),
                    node_session_index,
                )
            )
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            out.extend(
                _walk_nodes_with_session(
                    value,
                    path + (f"[{idx}]",),
                    node_session_index,
                )
            )
    return out


def _session_scope_matches(*, requested: int | None, scoped: int | None) -> bool:
    if requested is None or scoped is None:
        return True
    return int(requested) == int(scoped)


def _path_relevance(path_tokens: Sequence[str], hint_terms: set[str]) -> float:
    score = 0.0
    for token in path_tokens:
        for t in _PATH_TOKEN_RE.findall(token.lower()):
            if t in hint_terms:
                score += 1.0
            if t in {"coherence", "connectivity", "electrode", "topography"}:
                score += 0.5
    return score


def _coerce_electrode_map(node: Any) -> dict[str, float]:
    if not isinstance(node, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in node.items():
        label = normalize_electrode_label(key)
        if not label:
            continue
        val = _to_float(value)
        if val is None and isinstance(value, Mapping):
            val = _first_numeric_value(value, _NUMERIC_VALUE_KEYS)
        if val is None:
            continue
        out[label] = float(val)
    return {label: out[label] for label in sorted(out)}


def _coerce_coherence_edges(node: Any) -> list[dict[str, float | str]]:
    return _dedupe_edges(_coerce_coherence_edge_tuples(node))


def _coerce_coherence_edge_tuples(node: Any, *, session_index: int | None = None) -> list[tuple[str, str, float]]:
    if isinstance(node, list):
        out: list[tuple[str, str, float]] = []
        for item in node:
            out.extend(_coerce_coherence_edge_tuples(item, session_index=session_index))
        return out
    if not isinstance(node, Mapping):
        return []
    if session_index is not None and "session_index" in node:
        try:
            if int(node.get("session_index")) != int(session_index):
                return []
        except Exception:
            return []

    edges: list[tuple[str, str, float]] = []

    direct = _edge_from_mapping(node)
    if direct:
        edges.append(direct)

    # Pair key mapping, e.g. {"F3-C3": 0.72}
    for key, value in node.items():
        val = _to_float(value)
        if val is None:
            continue
        a, b = _parse_pair_labels(str(key))
        if a and b:
            edges.append((a, b, float(val)))

    # Adjacency mapping, e.g. {"F3": {"C3": 0.7}}
    for key, value in node.items():
        a = normalize_electrode_label(key)
        if not a or not isinstance(value, Mapping):
            continue
        for sub_key, sub_value in value.items():
            b = normalize_electrode_label(sub_key)
            if not b:
                continue
            val = _to_float(sub_value)
            if val is None and isinstance(sub_value, Mapping):
                val = _first_numeric_value(sub_value, _EDGE_VALUE_KEYS)
            if val is None:
                continue
            edges.append((a, b, float(val)))
    return edges


def _edge_from_mapping(node: Mapping[str, Any]) -> tuple[str, str, float] | None:
    a = _first_label(node, _EDGE_A_KEYS, pair_index=0)
    b = _first_label(node, _EDGE_B_KEYS, pair_index=1)
    if (not a or not b) and isinstance(node.get("pair"), str):
        p_a, p_b = _parse_pair_labels(node.get("pair"))
        a = a or p_a
        b = b or p_b

    if (not a or not b) and isinstance(node.get("edge"), str):
        p_a, p_b = _parse_pair_labels(node.get("edge"))
        a = a or p_a
        b = b or p_b

    value = _first_numeric_value(node, _EDGE_VALUE_KEYS)
    if not a or not b or value is None or a == b:
        return None
    return a, b, float(value)


def _dedupe_edges(edges: Sequence[tuple[str, str, float]]) -> list[dict[str, float | str]]:
    best: dict[tuple[str, str], float] = {}
    for a, b, value in edges:
        a_norm = normalize_electrode_label(a)
        b_norm = normalize_electrode_label(b)
        if not a_norm or not b_norm or a_norm == b_norm:
            continue
        key = tuple(sorted((a_norm, b_norm)))
        current = best.get(key)
        if current is None or abs(value) > abs(current):
            best[key] = float(value)
    out = [
        {"a": a, "b": b, "value": float(val)}
        for (a, b), val in sorted(best.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    return out


def _resolve_path(root: Mapping[str, Any], path_expr: str) -> Any:
    path = str(path_expr or "").strip()
    if not path:
        return None
    if path.startswith("$."):
        path = path[2:]
    path = path.replace("/", ".")
    parts = [p for p in path.split(".") if p]
    cur: Any = root
    for part in parts:
        m = re.match(r"^([^\[]+)(?:\[(\d+)\])?$", part)
        if not m:
            return None
        key = m.group(1)
        idx = m.group(2)
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
        if idx is not None:
            if not isinstance(cur, list):
                return None
            i = int(idx)
            if i < 0 or i >= len(cur):
                return None
            cur = cur[i]
    return cur


def _first_label(node: Mapping[str, Any], keys: Sequence[str], *, pair_index: int = 0) -> str | None:
    for key in keys:
        raw = node.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            norm = normalize_electrode_label(raw)
            if norm:
                return norm
            a, b = _parse_pair_labels(raw)
            if a and b:
                return (a, b)[0 if pair_index <= 0 else 1]
    return None


def _first_numeric_value(node: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    lowered = {str(k).lower(): v for k, v in node.items()}
    for key in keys:
        if key in lowered:
            parsed = _to_float(lowered[key])
            if parsed is not None:
                return parsed
    for value in node.values():
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return None
    m = re.match(r"^[-+]?\d+(?:\.\d+)?$", cleaned)
    if not m:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _parse_pair_labels(raw: str) -> tuple[str | None, str | None]:
    text = str(raw or "").strip()
    if not text:
        return None, None
    m = _PAIR_RE.match(text)
    if not m:
        return None, None
    a = normalize_electrode_label(m.group(1))
    b = normalize_electrode_label(m.group(2))
    return a, b


def _tokenize_key(key: Any) -> str:
    return str(key).strip().lower()


def _fact_tokens(fact: Mapping[str, Any]) -> tuple[str, ...]:
    tokens: list[str] = []
    for key, value in fact.items():
        if isinstance(key, str):
            tokens.append(key.lower())
        if isinstance(value, str):
            tokens.extend(_PATH_TOKEN_RE.findall(value.lower()))
    return tuple(tokens)


def _parse_int(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if re.match(r"^-?\d+$", text):
            try:
                return int(text)
            except Exception:
                return None
    return None
