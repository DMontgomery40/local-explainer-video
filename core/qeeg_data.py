"""Shared helpers for loading Stage-1 qEEG data packs outside QC flow."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any


_PATIENT_ID_RE = re.compile(r"^(?P<mm>\d{2})-(?P<dd>\d{2})-(?P<yyyy>\d{4})-(?P<n>\d+)$")
_PATIENT_ID_PREFIX_RE = re.compile(r"^(?P<pid>\d{2}-\d{2}-\d{4}-\d+)(?:__\d+)?$")


class QEEGDataPackError(RuntimeError):
    """Raised when Stage-1 data-pack resolution fails."""


@dataclass(frozen=True)
class Stage1DataPack:
    patient_id: str
    run_id: str
    data_pack_path: Path
    data_pack: dict[str, Any]


def _repo_root() -> Path:
    # core/qeeg_data.py -> core/ -> repo root
    return Path(__file__).resolve().parents[1]


def default_qeeg_analysis_dir() -> Path:
    env = os.getenv("QEEG_ANALYSIS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root().parent / "qEEG-analysis").resolve()


def infer_patient_id(project_name: str) -> str | None:
    """Infer MM-DD-YYYY-N from a project folder name (supports __02 suffix)."""
    raw = (project_name or "").strip()
    m = _PATIENT_ID_PREFIX_RE.match(raw)
    if not m:
        return None
    pid = m.group("pid")
    return pid if _PATIENT_ID_RE.match(pid) else None


def _resolve_qeeg_path(qeeg_dir: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    return p if p.is_absolute() else (qeeg_dir / p)


def load_latest_stage1_data_pack(
    *,
    patient_label: str,
    qeeg_dir: Path | None = None,
) -> Stage1DataPack:
    """
    Load the most recent Stage-1 `_data_pack.json` for a patient label.

    Unlike QC ground-truth loading, this does not require Stage-4 consolidation.
    """
    qeeg_root = (qeeg_dir or default_qeeg_analysis_dir()).expanduser().resolve()
    db_path = qeeg_root / "data" / "app.db"
    if not db_path.exists():
        raise QEEGDataPackError(f"qEEG Council DB not found: {db_path}")

    label = (patient_label or "").strip()
    if not label:
        raise QEEGDataPackError("patient_label is required")

    candidates: list[str] = [label]
    inferred = infer_patient_id(label)
    if inferred and inferred not in candidates:
        candidates.append(inferred)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        for candidate in candidates:
            cur.execute(
                """
                SELECT
                  runs.id AS run_id,
                  runs.created_at AS created_at
                FROM runs
                JOIN patients ON patients.id = runs.patient_id
                WHERE lower(patients.label) = lower(?)
                ORDER BY runs.created_at DESC
                """,
                (candidate,),
            )
            runs = cur.fetchall() or []
            if not runs:
                continue

            for row in runs:
                run_id = str(row["run_id"])
                cur.execute(
                    """
                    SELECT content_path, created_at
                    FROM artifacts
                    WHERE run_id = ? AND stage_num = 1 AND kind = 'data_pack' AND model_id = '_data_pack'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (run_id,),
                )
                artifact = cur.fetchone()
                if artifact is None:
                    continue
                data_pack_path = _resolve_qeeg_path(qeeg_root, str(artifact["content_path"]))
                if not data_pack_path.exists():
                    continue
                try:
                    data_pack = json.loads(data_pack_path.read_text(encoding="utf-8"))
                except Exception as e:
                    raise QEEGDataPackError(f"Failed reading Stage-1 data pack: {data_pack_path} ({e})") from e
                return Stage1DataPack(
                    patient_id=candidate,
                    run_id=run_id,
                    data_pack_path=data_pack_path,
                    data_pack=data_pack,
                )

        raise QEEGDataPackError(
            "No Stage-1 `_data_pack.json` artifact found.\n"
            f"- patient_label={label}\n"
            f"- qeeg_dir={qeeg_root}\n"
            "Hint: run qEEG-analysis extraction so Stage-1 artifacts exist for this patient."
        )
    finally:
        con.close()


def try_load_latest_stage1_data_pack(
    *,
    patient_label: str,
    qeeg_dir: Path | None = None,
) -> Stage1DataPack | None:
    """Best-effort wrapper: returns None instead of raising."""
    try:
        return load_latest_stage1_data_pack(patient_label=patient_label, qeeg_dir=qeeg_dir)
    except Exception:
        return None
