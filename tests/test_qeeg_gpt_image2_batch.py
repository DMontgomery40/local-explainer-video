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
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"synthetic image")
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


def test_the_engine_directory_is_read_from_the_same_env_var_as_the_publisher(
    tmp_path, monkeypatch
):
    """This script publishes into the engine's portal folder and then asks the
    engine to sync it. If it resolved a different installation than
    core.qc_publish does, videos would land in one clinic's folder and the sync
    would push another's."""
    from core.qeeg_env import default_qeeg_analysis_dir as publisher_default

    elsewhere = tmp_path / "other-qEEG"
    elsewhere.mkdir()
    monkeypatch.setenv("QEEG_ANALYSIS_DIR", str(elsewhere))
    module = _load_module()

    assert module.default_qeeg_analysis_dir() == elsewhere.resolve()
    assert module.default_qeeg_analysis_dir() == publisher_default()

    monkeypatch.delenv("QEEG_ANALYSIS_DIR")
    assert module.default_qeeg_analysis_dir() == publisher_default()

import errno
import pytest
from concurrent.futures import ThreadPoolExecutor


@pytest.mark.parametrize('absolute', [False, True])
def test_project_assets_use_project_root_through_backup_prompt_normalize_and_mirror(tmp_path, monkeypatch, absolute):
    from PIL import Image
    mod = _load_module()
    project = tmp_path / 'projects' / 'ZZ_01-01-1900'
    (project / 'images').mkdir(parents=True)
    source = project / 'images' / 'scene.png'
    Image.new('RGB', (32, 20), 'red').save(source)
    original = source.read_bytes()
    raw = str(source) if absolute else 'images/scene.png'
    _write_plan(project, {'scenes': [{'id': 0, 'visual_prompt': 'Synthetic', 'image_path': raw}, {'id': 1, 'visual_prompt': 'Missing', 'image_path': 'missing.png'}]})
    candidate = mod.build_candidate('local-explainer-video', tmp_path, project)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, 'CATHODE_ROOT', tmp_path / 'mirror')
    backup = mod.ensure_backup_dir(candidate)
    assert (backup / 'scene.png').read_bytes() == original
    prompt = mod.build_codex_prompt(candidate)
    assert str(source) in prompt and 'missing.png' not in prompt
    assert mod.normalize_project_images(candidate)[0]['path'] == str(source)
    mod.ensure_backup_dir(candidate)
    assert (backup / 'scene.png').read_bytes() == original
    video = project / 'video.mp4'
    video.write_bytes(b'video')
    mirrored = mod.mirror_to_cathode(candidate, video)
    assert (mirrored / 'images' / 'scene_000.png').read_bytes() == source.read_bytes()


@pytest.mark.parametrize('escape', ['relative', 'absolute', 'symlink'])
@pytest.mark.parametrize('operation', ['ensure_backup_dir', 'normalize_project_images', 'build_codex_prompt', 'mirror_to_cathode'])
def test_project_asset_escapes_rejected(tmp_path, monkeypatch, escape, operation):
    mod = _load_module()
    project = tmp_path / 'ZZ_01-01-1900'
    project.mkdir()
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'outside')
    (project / 'link.png').symlink_to(outside)
    raw = {'relative': '../outside.png', 'absolute': str(outside), 'symlink': 'link.png'}[escape]
    _write_plan(project, {'scenes': [{'id': 0, 'visual_prompt': 'Synthetic', 'image_path': raw}]})
    candidate = mod.build_candidate('local-explainer-video', tmp_path, project)
    monkeypatch.setattr(mod, 'CATHODE_ROOT', tmp_path / 'mirror')
    with pytest.raises(ValueError, match='project'):
        getattr(mod, operation)(candidate, outside) if operation == 'mirror_to_cathode' else getattr(mod, operation)(candidate)
    assert outside.read_bytes() == b'outside'


@pytest.mark.parametrize('mode', ['link', 'copy', 'failed_copy', 'interrupted'])
def test_portal_replacement_preserves_previous_until_success(tmp_path, monkeypatch, mode):
    mod = _load_module()
    monkeypatch.setattr(mod, 'PORTAL_PATIENTS_DIR', tmp_path / 'portal')
    patient = 'ZZ_01-01-1900'
    dest = tmp_path / 'portal' / patient / f'{patient}.mp4'
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'previous')
    source = tmp_path / 'new.mp4'
    source.write_bytes(b'new complete video')
    if mode != 'link':
        def cross_device(*args):
            assert dest.read_bytes() == b'previous'
            raise OSError(errno.EXDEV, 'synthetic cross device')
        monkeypatch.setattr(mod.os, 'link', cross_device)
    if mode in ('failed_copy', 'interrupted'):
        def failed_copy(src, target):
            Path(target).write_bytes(b'partial')
            raise OSError('disk full') if mode == 'failed_copy' else KeyboardInterrupt()
        monkeypatch.setattr(mod.shutil, 'copy2', failed_copy)
        with pytest.raises((OSError, KeyboardInterrupt)):
            mod.publish_to_portal(patient, source)
        assert dest.read_bytes() == b'previous'
    else:
        assert mod.publish_to_portal(patient, source).read_bytes() == source.read_bytes()
    assert list(dest.parent.glob('.*.partial')) == []


def test_concurrent_portal_publications_use_independent_staging(tmp_path, monkeypatch):
    import threading
    mod = _load_module()
    monkeypatch.setattr(mod, 'PORTAL_PATIENTS_DIR', tmp_path / 'portal')
    barrier = threading.Barrier(2)
    original_copy = mod.shutil.copy2
    def no_link(*args):
        raise OSError(errno.EXDEV, 'synthetic')
    def synchronized_copy(src, target):
        result = original_copy(src, target)
        barrier.wait(timeout=5)
        return result
    monkeypatch.setattr(mod.os, 'link', no_link)
    monkeypatch.setattr(mod.shutil, 'copy2', synchronized_copy)
    sources = [tmp_path / f'{i}.mp4' for i in range(2)]
    for i, source in enumerate(sources):
        source.write_bytes(bytes([i]) * 4096)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda source: mod.publish_to_portal('ZZ_01-01-1900', source), sources))
    assert results[0].read_bytes() in [source.read_bytes() for source in sources]
    assert list(results[0].parent.glob('.*.partial')) == []


@pytest.mark.parametrize('produced', [0, 1, 2, 3])
def test_native_refresh_requires_every_run_bound_image_before_replacing_targets(tmp_path, monkeypatch, produced):
    import re
    from PIL import Image
    from types import SimpleNamespace
    from core import image_gen
    mod=_load_module()
    project=tmp_path/'projects'/'ZZ_01-01-1900';(project/'images').mkdir(parents=True)
    scenes=[];originals={}
    for i in range(2):
        p=project/'images'/f'scene_{i:03d}.png';Image.new('RGB',(32,18),'red').save(p)
        originals[p]=p.read_bytes();scenes.append({'id':i,'visual_prompt':f'Original prompt {i}','image_path':str(p)})
    _write_plan(project,{'meta':{},'scenes':scenes})
    candidate=mod.build_candidate('local-explainer-video',tmp_path,project)
    monkeypatch.setattr(image_gen,'resolve_codex_runtime',lambda:{'path':'synthetic-codex'})
    calls=[]
    original_run = mod.subprocess.run
    def command(*args, **kwargs):
        if args[0][0] != "synthetic-codex": return original_run(*args, **kwargs)
        calls.append(kwargs['input'])
        match=re.search(r'Copy the generated PNG to ([^\n]+)\.',kwargs['input'])
        if match and len(calls)<=produced:
            p=Path(match.group(1));p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(32,18),'blue').save(p)
        if produced == 3 and len(calls) == 2:
            (project/'plan.json').write_text((project/'plan.json').read_text()+'\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(mod.subprocess,'run',command)
    if produced != 2:
        with pytest.raises(RuntimeError):mod.run_codex_refresh(candidate,run_dir=tmp_path/'run',model=None)
        assert all(p.read_bytes()==raw for p,raw in originals.items())
        if produced < 2:
            count=len(calls)
            with pytest.raises(RuntimeError):mod.run_codex_refresh(candidate,run_dir=tmp_path/'run',model=None)
            assert len(calls)==count, 'unknown dispatch must not automatically repeat paid work'
    else:
        result=mod.run_codex_refresh(candidate,run_dir=tmp_path/'run',model=None)
        assert result[0]==0 and len(calls)==2
        assert all(p.read_bytes()!=raw for p,raw in originals.items())
        mod.run_codex_refresh(candidate,run_dir=tmp_path/'run',model=None)
        assert len(calls)==2, 'same run recovers original response receipts without redispatch'

import pytest

@pytest.mark.parametrize('motion', [
    {'scene_type':'motion'}, {'scene_type':' MOTION '},
    {'composition':{'mode':'native'}}, {'composition':{'manifestation':'native_remotion'}},
])
def test_static_batch_rejects_motion_with_poster_and_prompt(tmp_path, monkeypatch, motion):
    mod=_load_module();project=tmp_path/'projects/ZZ_01-01-1900'
    _write_plan(project, {'scenes':[{'visual_prompt':'static','scene_type':'image'}]})
    candidate=mod.build_candidate('local-explainer-video',tmp_path,project)
    assert candidate
    _write_plan(project, {'scenes':[{'visual_prompt':'static','scene_type':'image'},dict(motion,visual_prompt='motion poster',image_path='images/poster.png')]})
    assert mod.build_candidate('local-explainer-video',tmp_path,project) is None
    from core import image_gen
    monkeypatch.setattr(image_gen,'_run_codex_exec_image',lambda **k:pytest.fail('Paid static refresh for motion'))
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**k:pytest.fail('Static render for motion'))
    with pytest.raises(ValueError,match='motion'):
        mod.run_codex_refresh(candidate,run_dir=tmp_path/'run',model=None)
    with pytest.raises(ValueError,match='motion'):
        mod.rerender_local_explainer(candidate)
    assert not (tmp_path/'run').exists()


def test_static_batch_plan_changed_after_discovery_fails_before_backup(tmp_path, monkeypatch):
    from types import SimpleNamespace
    mod=_load_module();project=tmp_path/'projects/ZZ_01-01-1900'
    _write_plan(project, {'scenes':[{'visual_prompt':'static'}]})
    candidate=mod.build_candidate('local-explainer-video',tmp_path,project)
    _write_plan(project, {'scenes':[{'scene_type':'motion','visual_prompt':'poster'}]})
    monkeypatch.setattr(mod,'parse_args',lambda:SimpleNamespace(patients='',run_label='test',dry_run=False,model='',skip_thrylen_sync=True,max_patients=0))
    monkeypatch.setattr(mod,'_patient_results_dir',lambda *a:tmp_path/'results')
    monkeypatch.setattr(mod,'discover_latest_prompt_projects',lambda:[candidate])
    monkeypatch.setattr(mod,'ensure_backup_dir',lambda *a:pytest.fail('Backed up invalid motion candidate'))
    assert mod.main()==1
    assert json.loads((tmp_path/'results/results.json').read_text())['results'][0]['ok'] is False

def test_new_batch_same_second_gets_distinct_action_directory(monkeypatch):
    mod=_load_module();now=mod.utc_now();monkeypatch.setattr(mod,'utc_now',lambda:now)
    assert mod._patient_results_dir('same-label') != mod._patient_results_dir('same-label')

def test_batch_new_action_dispatches_same_prompt_while_retry_reuses_receipt(tmp_path,monkeypatch):
    import re
    from PIL import Image
    from types import SimpleNamespace
    from core import image_gen
    mod=_load_module();project=tmp_path/'projects'/'ZZ_01-01-1900';project.mkdir(parents=True)
    image=project/'image.png';Image.new('RGB',(32,18),'green').save(image)
    _write_plan(project,{'meta':{},'scenes':[{'id':0,'visual_prompt':'Same prompt','image_path':str(image)}]})
    candidate=mod.build_candidate('local-explainer-video',tmp_path,project)
    monkeypatch.setattr(image_gen,'resolve_codex_runtime',lambda:{'path':'synthetic-codex'})
    calls=[]
    original_run=image_gen.subprocess.run
    def command(*args,**kwargs):
        if args[0][0] != "synthetic-codex":return original_run(*args,**kwargs)
        calls.append(kwargs['input']);match=re.search(r'Copy the generated PNG to ([^\n]+)\.',kwargs['input'])
        target=Path(match.group(1));target.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(32,18),'red' if len(calls)==1 else 'blue').save(target)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(image_gen.subprocess,'run',command)
    mod.run_codex_refresh(candidate,run_dir=tmp_path/'run-a',model=None)
    mod.run_codex_refresh(candidate,run_dir=tmp_path/'run-a',model=None);assert len(calls)==1
    mod.run_codex_refresh(candidate,run_dir=tmp_path/'run-b',model=None);assert len(calls)==2
    assert calls[0]!=calls[1], 'Native raw output belongs to its action'


def test_same_instant_batch_backups_preserve_each_immediately_previous_image(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    mod = _load_module()
    monkeypatch.setattr(mod, 'utc_now', lambda: datetime(2026, 9, 5, tzinfo=timezone.utc))
    project=tmp_path/'projects'/'ZZ_01-01-1900'; (project/'images').mkdir(parents=True)
    source=project/'images'/'scene.png'
    _write_plan(project, {'scenes':[{'id':0,'visual_prompt':'Synthetic','image_path':'images/scene.png'}]})
    candidate=mod.build_candidate('local-explainer-video',tmp_path,project)
    backups=[]
    for raw in [b'version-one', b'version-two', b'version-three']:
        source.write_bytes(raw)
        backup=mod.ensure_backup_dir(candidate)
        backups.append((backup,raw))
    assert len({path for path,_ in backups}) == 3
    for path,raw in backups: assert (path/'scene.png').read_bytes() == raw
