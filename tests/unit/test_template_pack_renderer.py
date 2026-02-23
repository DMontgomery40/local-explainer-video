from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.template_pack import (
    TemplatePackEntry,
    classify_scene_archetype,
    load_template_manifest,
    render_non_brain_scene,
    select_template_for_scene,
    write_template_manifest,
)


def test_classify_scene_archetype_metric_dashboard() -> None:
    scene = {
        "title": "Finding the zone",
        "visual_prompt": "Gauge panel with 9.5 uV and 14.3 uV markers",
    }
    assert classify_scene_archetype(scene) == "metric_dashboard"


def test_load_template_manifest_missing_file_returns_empty(tmp_path: Path) -> None:
    manifest = load_template_manifest(tmp_path / "missing.json")
    assert manifest.entries == tuple()


def test_select_template_for_scene_is_deterministic(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (1664, 928), (20, 30, 40)).save(image_a)
    Image.new("RGB", (1664, 928), (40, 60, 80)).save(image_b)

    entries = [
        TemplatePackEntry("t_a", "trend_panel", image_a, "p1", 1),
        TemplatePackEntry("t_b", "trend_panel", image_b, "p2", 2),
    ]

    manifest_path = tmp_path / "manifest.json"
    write_template_manifest(manifest_path, entries)
    manifest = load_template_manifest(manifest_path)

    scene = {
        "id": 7,
        "uid": "abc12345",
        "title": "Trend",
        "visual_prompt": "trend panel",
    }

    first = select_template_for_scene(scene, manifest)
    second = select_template_for_scene(scene, manifest)
    assert first is not None and second is not None
    assert first.template_id == second.template_id


def test_render_non_brain_scene_writes_output(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    scene = {
        "id": 3,
        "title": "Roadmap",
        "subtitle": "What comes next",
        "visual_prompt": "A timeline panel with five milestones and 30% progress",
    }

    out = render_non_brain_scene(scene, project_dir, manifest_path=tmp_path / "none.json")
    assert out.exists()

    with Image.open(out) as img:
        assert img.size == (1664, 928)


def test_manifest_round_trip_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "one.png"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(image_path)

    entries = [TemplatePackEntry("one", "general_panel", image_path, "p", 9)]
    manifest_path = tmp_path / "manifest.json"
    write_template_manifest(manifest_path, entries)

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["entries"][0]["template_id"] == "one"

    loaded = load_template_manifest(manifest_path)
    assert loaded.entries and loaded.entries[0].path == image_path.resolve()
