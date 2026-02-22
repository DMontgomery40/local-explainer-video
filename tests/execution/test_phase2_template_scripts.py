from __future__ import annotations

import json
from pathlib import Path

from scripts.build_template_anchors import run_anchor_build
from scripts.build_template_approval_packet import build_approval_packet
from scripts.generate_qwen_templates import run_generation


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1664, 928), (20, 30, 42)).save(path, format="PNG")


def _write_manifest(repo: Path, payload: dict) -> Path:
    manifest_path = repo / "templates" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def test_generate_templates_dry_run_writes_report(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    bg = repo / "templates" / "backgrounds" / "bar_volume_chart_3panel_v1.png"
    anchor = repo / "templates" / "anchors" / "bar_volume_chart_3panel_v1.json"
    _write_png(bg)
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text('{"template_id": "bar_volume_chart_3panel_v1", "anchors": []}', encoding="utf-8")

    manifest = _write_manifest(
        repo,
        {
            "manifest_version": "1.1.0",
            "templates": [
                {
                    "template_id": "bar_volume_chart_3panel_v1",
                    "scene_types": ["bar_volume_chart"],
                    "template_path": "templates/backgrounds/bar_volume_chart_3panel_v1.png",
                    "anchors_path": "templates/anchors/bar_volume_chart_3panel_v1.json",
                    "priority": 10,
                    "origin": "scaffold_only",
                    "operational_status": "dev_only",
                    "production_ready": False,
                    "curation_status": "pending",
                    "selector_metadata": {},
                    "selector": {},
                    "tags": [],
                }
            ],
        },
    )

    code, report = run_generation(
        repo_root=repo,
        manifest_path=manifest,
        template_ids=set(),
        all_archetypes=True,
        include_dev_only=False,
        live=False,
        model="qwen/qwen-image-2512",
        output_root=repo / "artifacts" / "template_generation",
        update_manifest=False,
    )
    assert code == 0
    assert report["target_count"] == 1
    assert report["rows"][0]["status"] == "dry_run"
    artifact_dir = Path(str(report["artifact_dir"]))
    assert (artifact_dir / "generation_report.json").exists()


def test_generate_templates_loads_repo_dotenv_for_replicate_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    (repo / ".env").write_text("REPLICATE_API_TOKEN=r8_test_token\\n", encoding="utf-8")
    _write_png(repo / "templates" / "backgrounds" / "bar_volume_chart_3panel_v1.png")
    (repo / "templates" / "anchors").mkdir(parents=True, exist_ok=True)
    (repo / "templates" / "anchors" / "bar_volume_chart_3panel_v1.json").write_text(
        '{"template_id":"bar_volume_chart_3panel_v1","anchors":[]}',
        encoding="utf-8",
    )
    manifest = _write_manifest(
        repo,
        {
            "manifest_version": "1.1.0",
            "templates": [
                {
                    "template_id": "bar_volume_chart_3panel_v1",
                    "scene_types": ["bar_volume_chart"],
                    "template_path": "templates/backgrounds/bar_volume_chart_3panel_v1.png",
                    "anchors_path": "templates/anchors/bar_volume_chart_3panel_v1.json",
                    "priority": 10,
                    "origin": "scaffold_only",
                    "operational_status": "dev_only",
                    "production_ready": False,
                    "curation_status": "pending",
                    "selector_metadata": {},
                    "selector": {},
                    "tags": [],
                }
            ],
        },
    )
    code, report = run_generation(
        repo_root=repo,
        manifest_path=manifest,
        template_ids=set(),
        all_archetypes=True,
        include_dev_only=False,
        live=False,
        model="qwen/qwen-image-2512",
        output_root=repo / "artifacts" / "template_generation",
        update_manifest=False,
    )
    assert code == 0
    assert report["replicate_token_present"] is True
    assert report["env_loaded"] == str((repo / ".env").resolve())


def test_anchor_builder_creates_missing_anchor_from_prototype(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    prototype = repo / "templates" / "anchors" / "generic_data_panel_v1.json"
    prototype.parent.mkdir(parents=True, exist_ok=True)
    prototype.write_text(
        json.dumps({"template_id": "generic_data_panel_v1", "dimensions": [1664, 928], "anchors": []}),
        encoding="utf-8",
    )

    manifest = _write_manifest(
        repo,
        {
            "manifest_version": "1.1.0",
            "templates": [
                {
                    "template_id": "table_dashboard_v2",
                    "scene_types": ["table_dashboard"],
                    "template_path": "templates/backgrounds/table_dashboard_v2.png",
                    "anchors_path": "templates/anchors/table_dashboard_v2.json",
                    "priority": 50,
                    "origin": "qwen_curated",
                    "operational_status": "production",
                    "production_ready": False,
                }
            ],
        },
    )

    code, report = run_anchor_build(
        repo_root=repo,
        manifest_path=manifest,
        output_root=repo / "artifacts" / "template_anchors",
        template_ids={"table_dashboard_v2"},
        include_all=False,
        force=False,
        dry_run=False,
    )
    assert code == 0
    assert report["created_count"] == 1

    generated_anchor = repo / "templates" / "anchors" / "table_dashboard_v2.json"
    assert generated_anchor.exists()
    payload = json.loads(generated_anchor.read_text(encoding="utf-8"))
    assert payload["template_id"] == "table_dashboard_v2"


def test_approval_packet_writes_pending_approval_artifact(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    bg = repo / "templates" / "backgrounds" / "title_card_v1.png"
    anchor = repo / "templates" / "anchors" / "title_card_v1.json"
    _write_png(bg)
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text('{"template_id": "title_card_v1", "anchors": []}', encoding="utf-8")

    manifest = _write_manifest(
        repo,
        {
            "manifest_version": "1.1.0",
            "templates": [
                {
                    "template_id": "title_card_v1",
                    "scene_types": ["atmospheric_title_card"],
                    "template_path": "templates/backgrounds/title_card_v1.png",
                    "anchors_path": "templates/anchors/title_card_v1.json",
                    "priority": 1,
                    "origin": "qwen_curated",
                    "operational_status": "production",
                    "production_ready": False,
                    "curation_status": "pending_approval",
                }
            ],
        },
    )

    approval_path = repo / ".codex" / "template_approval" / "approval.json"
    result = build_approval_packet(
        repo_root=repo,
        manifest_path=manifest,
        output_root=repo / "artifacts" / "template_approval",
        approval_path=approval_path,
        packet_name="packet_a",
    )

    packet_dir = Path(result["packet_dir"])
    assert packet_dir.exists()
    assert (packet_dir / "template_contact_sheet.png").exists()
    assert (packet_dir / "manifest_summary.json").exists()
    assert (packet_dir / "coverage_gaps.json").exists()

    payload = json.loads(approval_path.read_text(encoding="utf-8"))
    assert payload["approved"] is False
    assert payload["packet_dir"] == str(packet_dir.resolve())
