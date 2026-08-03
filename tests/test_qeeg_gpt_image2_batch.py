from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "qeeg_gpt_image2_batch.py"
    spec = importlib.util.spec_from_file_location("qeeg_gpt_image2_batch", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_plan(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_discover_latest_prompt_projects_prefers_newest_video_and_skips_promptless_portal_patient(tmp_path: Path):
    mod = _load_module()
    portal_dir = tmp_path / "portal_patients"
    lev_root = tmp_path / "local-explainer-video"
    cathode_root = tmp_path / "cathode"

    for patient_id in ("DM_01-01-1983", "AB_02-02-1984"):
        (portal_dir / patient_id).mkdir(parents=True)

    lev_project = lev_root / "projects" / "DM_01-01-1983"
    lev_video = lev_project / "01-01-1983-0.mp4"
    lev_video.parent.mkdir(parents=True, exist_ok=True)
    lev_video.write_bytes(b"lev")
    _write_plan(
        lev_project,
        {
            "meta": {
                "video_path": str(lev_video),
                "rendered_utc": "2026-04-20T20:00:00Z",
            },
            "scenes": [
                {"id": 0, "visual_prompt": "Prompt A", "image_path": str(lev_project / "images" / "scene_000.png")},
                {"id": 1, "visual_prompt": "", "image_path": str(lev_project / "images" / "scene_001.png")},
            ],
        },
    )
    lev_video.touch()

    cathode_project = cathode_root / "projects" / "DM_01-01-1983__02"
    cathode_video = cathode_project / "01-01-1983-0.mp4"
    cathode_video.parent.mkdir(parents=True, exist_ok=True)
    cathode_video.write_bytes(b"cat")
    _write_plan(
        cathode_project,
        {
            "meta": {
                "video_path": str(cathode_video),
                "rendered_utc": "2026-04-21T20:00:00Z",
            },
            "scenes": [
                {"id": 0, "scene_type": "image", "visual_prompt": "Prompt B", "image_path": str(cathode_project / "images" / "scene_000.png")},
            ],
        },
    )
    cathode_video.touch()

    no_video_newer_project = cathode_root / "projects" / "DM_01-01-1983__03"
    _write_plan(
        no_video_newer_project,
        {
            "meta": {
                "created_utc": "2026-04-21T21:00:00Z",
            },
            "scenes": [
                {"id": 0, "scene_type": "image", "visual_prompt": "Prompt C", "image_path": str(no_video_newer_project / "images" / "scene_000.png")},
            ],
        },
    )

    promptless_project = cathode_root / "projects" / "AB_02-02-1984"
    promptless_video = promptless_project / "02-02-1984-0.mp4"
    promptless_video.parent.mkdir(parents=True, exist_ok=True)
    promptless_video.write_bytes(b"promptless")
    _write_plan(
        promptless_project,
        {
            "meta": {"video_path": str(promptless_video)},
            "scenes": [
                {"id": 0, "scene_type": "image", "visual_prompt": "", "image_path": str(promptless_project / "images" / "scene_000.png")},
            ],
        },
    )

    candidates = mod.discover_latest_prompt_projects(
        portal_patients_dir=portal_dir,
        local_explainer_root=lev_root,
        cathode_root=cathode_root,
    )

    assert [candidate.patient_id for candidate in candidates] == ["DM_01-01-1983"]
    selected = candidates[0]
    assert selected.repo_name == "cathode"
    assert selected.project_name == "DM_01-01-1983__02"
    assert selected.prompt_scene_count == 1
    assert selected.promptless_image_scene_count == 0


def test_build_candidate_counts_promptless_image_scenes_without_marking_video_scenes(tmp_path: Path):
    mod = _load_module()
    repo_root = tmp_path / "repo"
    project_dir = repo_root / "projects" / "CD_03-03-1985"
    project_dir.mkdir(parents=True)
    plan_path = project_dir / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "meta": {"created_utc": "2026-04-21T12:00:00Z"},
                "scenes": [
                    {"id": 0, "scene_type": "image", "visual_prompt": "Prompt 0", "image_path": str(project_dir / "images" / "scene_000.png")},
                    {"id": 1, "scene_type": "image", "visual_prompt": "", "image_path": str(project_dir / "images" / "scene_001.png")},
                    {"id": 2, "scene_type": "video", "visual_prompt": "", "video_path": str(project_dir / "clips" / "scene_002.mp4")},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    candidate = mod.build_candidate("local-explainer-video", repo_root, project_dir)

    assert candidate is not None
    assert candidate.prompt_scene_count == 1
    assert candidate.prompt_scene_ids == [0]
    assert candidate.promptless_image_scene_count == 1
    assert candidate.promptless_image_scene_ids == [1]


def test_discover_latest_prompt_projects_includes_prompt_plan_when_portal_has_video_but_source_mp4_is_missing(tmp_path: Path):
    mod = _load_module()
    portal_dir = tmp_path / "portal_patients"
    lev_root = tmp_path / "local-explainer-video"
    cathode_root = tmp_path / "cathode"

    patient_id = "BB_10-07-1963"
    patient_dir = portal_dir / patient_id
    patient_dir.mkdir(parents=True)
    portal_video = patient_dir / "10-07-1963_v3.mp4"
    portal_video.write_bytes(b"portal")

    source_project = lev_root / "projects" / patient_id
    _write_plan(
        source_project,
        {
            "meta": {"created_utc": "2026-04-20T12:00:00Z"},
            "scenes": [
                {"id": 0, "scene_type": "image", "visual_prompt": "Prompt A", "image_path": str(source_project / "images" / "scene_000.png")},
            ],
        },
    )

    candidates = mod.discover_latest_prompt_projects(
        portal_patients_dir=portal_dir,
        local_explainer_root=lev_root,
        cathode_root=cathode_root,
    )

    assert len(candidates) == 1
    selected = candidates[0]
    assert selected.patient_id == patient_id
    assert selected.repo_name == "local-explainer-video"
    assert selected.video_exists is False
    assert selected.portal_latest_video is not None
    assert selected.portal_latest_video.endswith("10-07-1963_v3.mp4")


def test_build_codex_prompt_forbids_secret_search_and_embeds_scene_jobs(tmp_path: Path):
    mod = _load_module()
    project_dir = tmp_path / "projects" / "PL_10-31-2008"
    image_path = project_dir / "images" / "scene_000.png"
    _write_plan(
        project_dir,
        {
            "meta": {},
            "scenes": [
                {"id": 0, "visual_prompt": "Exact prompt", "image_path": str(image_path)},
            ],
        },
    )
    candidate = mod.build_candidate("local-explainer-video", tmp_path, project_dir)

    prompt = mod.build_codex_prompt(candidate)

    assert "Do not inspect `~/.codex/skills`" in prompt
    assert "search the filesystem for `OPENAI_API_KEY`" in prompt
    assert '"scene_id": 0' in prompt
    assert '"visual_prompt": "Exact prompt"' in prompt
    assert "landscape 16:9" in prompt
    assert "1664x928" in prompt


def test_normalize_project_images_forces_exact_target_size(tmp_path: Path):
    mod = _load_module()
    from PIL import Image

    project_dir = tmp_path / "projects" / "DM_01-01-1983"
    image_dir = project_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "scene_000.png"
    Image.new("RGB", (1024, 1536), (10, 20, 30)).save(image_path)

    _write_plan(
        project_dir,
        {
            "meta": {},
            "scenes": [
                {"id": 0, "visual_prompt": "Portrait input", "image_path": str(image_path)},
            ],
        },
    )

    candidate = mod.build_candidate("local-explainer-video", tmp_path, project_dir)
    assert candidate is not None

    normalized = mod.normalize_project_images(candidate)

    assert normalized == [
        {
            "scene_id": 0,
            "path": str(image_path),
            "before": [1024, 1536],
            "after": [1664, 928],
        }
    ]
    with Image.open(image_path) as img:
        assert img.size == (1664, 928)


def test_sync_patient_to_thrylen_uses_timeout_and_returns_false_on_timeout(monkeypatch):
    mod = _load_module()

    monkeypatch.setenv("QEEG_THRYLEN_SYNC_TIMEOUT_SECONDS", "7")
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/opt/homebrew/bin/uv" if name == "uv" else None)

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    ok = mod.sync_patient_to_thrylen("01-01-1991-0")

    assert ok is False
    assert calls["cmd"] == [
        "/opt/homebrew/bin/uv",
        "run",
        "python",
        "-m",
        "backend.portal_sync",
        "--patient-label",
        "01-01-1991-0",
    ]
    assert calls["kwargs"]["timeout"] == 7
    assert calls["kwargs"]["cwd"] == str(mod.QEEG_ANALYSIS_ROOT)
    assert calls["kwargs"]["capture_output"] is True


def test_project_suffixes_are_told_apart_from_the_patients_own_ordinal():
    """Three things end a project name with an underscore and a number and only
    one is the patient's. `BT_12-11-1963_2` is a different person from
    `BT_12-11-1963`; `__02` and `_v4` are the same person's repeat work.
    """
    module = _load_module()

    assert module.infer_patient_id("BT_12-11-1963_2__02") == "BT_12-11-1963_2"
    assert module.infer_patient_id("BT_12-11-1963__02") == "BT_12-11-1963"
    # Repeat projects stack on disk.
    assert module.infer_patient_id("BT_12-11-1963_2__02__03") == "BT_12-11-1963_2"
    assert module.infer_patient_id("BT_12-11-1963_v4") == "BT_12-11-1963"
    assert module.infer_patient_id("DK_08-10-1989_10") == "DK_08-10-1989_10"
    # The retired date-of-birth key is not an id any runtime accepts now.
    assert module.infer_patient_id("10-31-2008-0") is None
    # `_1` never exists: ordinal one is the unsuffixed form.
    assert module.infer_patient_id("BT_12-11-1963_1") is None


def test_a_portal_folder_it_cannot_read_is_named_not_dropped(tmp_path, capsys):
    """A silently skipped folder is a patient who never gets a video."""
    module = _load_module()
    portal = tmp_path / "portal_patients"
    (portal / "BT_12-11-1963").mkdir(parents=True)
    (portal / "12-11-1963-0").mkdir(parents=True)

    module.discover_latest_prompt_projects(
        portal_patients_dir=portal,
        local_explainer_root=tmp_path / "lev",
        cathode_root=tmp_path / "cathode",
    )

    assert "skipping portal folder 12-11-1963-0" in capsys.readouterr().out
