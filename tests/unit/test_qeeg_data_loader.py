import json
import sqlite3
from pathlib import Path

import pytest

from core.qeeg_data import (
    QEEGDataPackError,
    infer_patient_id,
    load_latest_stage1_data_pack,
    try_load_latest_stage1_data_pack,
)


def _init_qeeg_db(root: Path) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "app.db"
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("CREATE TABLE patients (id TEXT PRIMARY KEY, label TEXT)")
        cur.execute(
            """
            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                patient_id TEXT,
                created_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                stage_num INTEGER,
                kind TEXT,
                model_id TEXT,
                content_path TEXT,
                created_at TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _add_stage1_artifact(root: Path, *, run_id: str, created_at: str, payload: dict) -> str:
    rel = f"data/artifacts/{run_id}/stage-1/_data_pack.json"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return rel


def test_infer_patient_id_accepts_versioned_project_name() -> None:
    assert infer_patient_id("09-23-1982-0__03") == "09-23-1982-0"
    assert infer_patient_id("09-23-1982-0") == "09-23-1982-0"
    assert infer_patient_id("invalid-project") is None


def test_load_latest_stage1_data_pack_returns_newest_run(tmp_path: Path) -> None:
    qeeg_dir = tmp_path / "qEEG-analysis"
    _init_qeeg_db(qeeg_dir)
    db_path = qeeg_dir / "data" / "app.db"

    old_rel = _add_stage1_artifact(
        qeeg_dir,
        run_id="run_old",
        created_at="2026-01-01T00:00:00Z",
        payload={"schema_version": 1, "facts": [{"fact_type": "old"}]},
    )
    new_rel = _add_stage1_artifact(
        qeeg_dir,
        run_id="run_new",
        created_at="2026-02-01T00:00:00Z",
        payload={"schema_version": 1, "facts": [{"fact_type": "new"}]},
    )

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("INSERT INTO patients (id, label) VALUES (?, ?)", ("patient_uuid_1", "09-23-1982-0"))
        cur.execute(
            "INSERT INTO runs (id, patient_id, created_at) VALUES (?, ?, ?)",
            ("run_old", "patient_uuid_1", "2026-01-01T00:00:00Z"),
        )
        cur.execute(
            "INSERT INTO runs (id, patient_id, created_at) VALUES (?, ?, ?)",
            ("run_new", "patient_uuid_1", "2026-02-01T00:00:00Z"),
        )
        cur.execute(
            """
            INSERT INTO artifacts (run_id, stage_num, kind, model_id, content_path, created_at)
            VALUES (?, 1, 'data_pack', '_data_pack', ?, ?)
            """,
            ("run_old", old_rel, "2026-01-01T00:00:00Z"),
        )
        cur.execute(
            """
            INSERT INTO artifacts (run_id, stage_num, kind, model_id, content_path, created_at)
            VALUES (?, 1, 'data_pack', '_data_pack', ?, ?)
            """,
            ("run_new", new_rel, "2026-02-01T00:00:00Z"),
        )
        con.commit()
    finally:
        con.close()

    loaded = load_latest_stage1_data_pack(patient_label="09-23-1982-0", qeeg_dir=qeeg_dir)
    assert loaded.run_id == "run_new"
    assert loaded.data_pack_path.name == "_data_pack.json"
    assert loaded.data_pack["facts"][0]["fact_type"] == "new"


def test_try_load_latest_stage1_data_pack_returns_none_on_missing(tmp_path: Path) -> None:
    qeeg_dir = tmp_path / "qEEG-analysis"
    _init_qeeg_db(qeeg_dir)
    loaded = try_load_latest_stage1_data_pack(patient_label="09-23-1982-0", qeeg_dir=qeeg_dir)
    assert loaded is None


def test_load_latest_stage1_data_pack_raises_on_missing(tmp_path: Path) -> None:
    qeeg_dir = tmp_path / "qEEG-analysis"
    _init_qeeg_db(qeeg_dir)
    with pytest.raises(QEEGDataPackError):
        load_latest_stage1_data_pack(patient_label="09-23-1982-0", qeeg_dir=qeeg_dir)
