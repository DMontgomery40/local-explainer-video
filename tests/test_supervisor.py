"""Detached ownership and exact-output contracts with synthetic providers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from core.generation_receipts import AssetBusy, ReceiptConflict, atomic_json, exclusive_lock

REPO = Path(__file__).resolve().parents[1]


def supervisor():
    from md_video_maker import supervisor as module
    return module


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCAL_EXPLAINER_IMAGE_PROVIDER', 'openai')
    project = tmp_path / 'project'; project.mkdir()
    (project / 'plan.json').write_text(json.dumps({'meta': {'tts_provider': 'openrouter'}, 'scenes': [
        {'id': 0, 'narration': 'Synthetic speech', 'visual_prompt': 'Synthetic frame'},
        {'id': 1, 'narration': 'Second speech', 'visual_prompt': 'Second frame'}]}))
    return project


def prepare(project, operation='op', **kwargs):
    return supervisor().prepare_attempt(project, operation, state_dir=project.parent/'state', **kwargs)


def wait_for(predicate, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.025)
    pytest.fail('Process did not reach expected durable evidence')


def test_admission_is_durable_distinct_and_exact(project, monkeypatch):
    s = supervisor()
    a = prepare(project)
    assert s.attempt_snapshot(a)['state'] == 'pending'
    assert prepare(project) == a
    assert prepare(project, 'another') != a
    admission = json.loads((a/'admission.json').read_text())
    assert admission['operation_id'] == 'op'
    assert admission['project_dir'] == str(project.resolve())
    assert admission['expected_output'] == str(a/'output.mp4')
    assert 'OPENAI_API_KEY' not in json.dumps(admission)
    monkeypatch.setenv('CODEX_BINARY', '/updated/codex')
    assert prepare(project) == a  # mutable runtime is provenance, not immutable intent
    (project/'plan.json').write_text((project/'plan.json').read_text() + '\n')
    with pytest.raises(ReceiptConflict):
        prepare(project)


@pytest.mark.parametrize('change', ['local_source', 'force', 'config', 'source', 'lock'])
def test_changed_immutable_input_conflicts(project, monkeypatch, change):
    s = supervisor()
    source = project/'source.png'; source.write_bytes(b'original source')
    plan = json.loads((project/'plan.json').read_text()); plan['scenes'][0]['image_source_path'] = 'source.png'
    (project/'plan.json').write_text(json.dumps(plan))
    prepare(project)
    kwargs = {}
    if change == 'local_source': source.write_bytes(b'changed')
    elif change == 'force': kwargs['force_images'] = True
    elif change == 'config': monkeypatch.setenv('LOCAL_EXPLAINER_IMAGE_QUALITY', 'low')
    else:
        original = s.release_identity
        def changed():
            result = original(); result[change + '_sha256'] = 'changed'; return result
        monkeypatch.setattr(s, 'release_identity', changed)
    with pytest.raises(ReceiptConflict): prepare(project, **kwargs)


def test_project_lock_serializes_admission_and_editing(project):
    s = supervisor()
    with s.project_write_lock(project):
        with pytest.raises(AssetBusy): prepare(project)
    assert prepare(project).exists()


@pytest.fixture
def worker(project, tmp_path, monkeypatch):
    # The normal detached CLI inherits this synthetic-provider harness. Production
    # has no fake-provider switch or testing branch.
    fixture = tmp_path/'fixture.mp4'
    subprocess.run(['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        'color=c=blue:s=64x36:r=24:d=0.2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(fixture)], check=True)
    hooks = tmp_path/'hooks'; hooks.mkdir()
    (hooks/'sitecustomize.py').write_text((REPO/'tests'/'supervisor_process_hook.py').read_text())
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(hooks), str(REPO)]))
    monkeypatch.setenv('SUPERVISOR_TEST_ROOT', str(tmp_path))
    monkeypatch.setenv('SUPERVISOR_TEST_FIXTURE', str(fixture))
    for key in ('OPENAI_API_KEY','OPENAI_IMAGE_API_KEY','OPENROUTER_API_KEY','ELEVENLABS_API_KEY','REPLICATE_API_TOKEN'):
        monkeypatch.setenv(key, '')
    children = []
    def launch(attempt, phase=''):
        env = {**os.environ, 'SUPERVISOR_TEST_PHASE': phase}
        child = subprocess.Popen([sys.executable, '-m', 'md_video_maker.supervisor', 'run', str(attempt)], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        children.append(child)
        return child
    yield launch
    for child in children:
        if child.poll() is None: child.kill()
        child.communicate(timeout=10)
    # start_attempt launches detached descendants rather than these tracked workers.
    for started in (tmp_path/'state').glob('attempts/*/started.json'):
        pid = json.loads(started.read_text())['pid']
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass


def calls(project):
    path = project.parent/'dispatches'
    return path.read_text().splitlines() if path.exists() else []


def state(attempt):
    return supervisor().attempt_snapshot(attempt)


def terminal(attempt):
    return wait_for(lambda: (snap if (snap:=state(attempt))['state'] in {'complete','local_failure','reconciliation_required'} else None))


@pytest.mark.parametrize('phase', ['before_started', 'after_started'])
def test_parent_death_around_child_acknowledgment(project, worker, monkeypatch, phase):
    a = prepare(project)
    monkeypatch.setenv('SUPERVISOR_TEST_PHASE', phase)
    parent = subprocess.Popen([sys.executable, '-c',
        'import sys,time; from pathlib import Path; from md_video_maker.supervisor import start_attempt; '
        'start_attempt(Path(sys.argv[1])); time.sleep(30)', str(a)])
    wait_for(lambda: (project.parent/('barrier-'+phase)).exists())
    parent.kill(); parent.wait()
    # A replacement launching parent may launch another lightweight supervisor.
    worker(a)
    (project.parent/('release-'+phase)).touch()
    assert terminal(a)['state'] == 'complete'
    assert sorted(calls(project)) == ['audio-0','audio-1','image-0','image-1']


def test_old_live_owner_never_stolen_and_nonowner_never_writes(project, worker):
    a = prepare(project)
    owner = worker(a, 'dispatch-image-0')
    wait_for(lambda: (project.parent/'barrier-dispatch-image-0').exists())
    before = (a/'started.json').read_bytes()
    stale = json.loads(before); stale['started_utc'] = '1900-01-01T00:00:00Z'; atomic_json(a/'started.json', stale)
    other = worker(a); other.wait(timeout=10)
    assert state(a)['state'] == 'running'
    assert json.loads((a/'started.json').read_text())['owner_token'] == stale['owner_token']
    assert not (a/'terminal.json').exists()
    assert calls(project) == ['image-0']
    (project.parent/'release-dispatch-image-0').touch()
    assert terminal(a)['state'] == 'complete'


@pytest.mark.parametrize('phase,expected', [('after_started','complete'), ('dispatch-image-0','reconciliation_required'),
                                          ('converted-image-0','complete'), ('before_terminal','complete')])
def test_owner_death_resumes_exact_receipts(project, worker, phase, expected):
    a = prepare(project, force_images=True, force_audio=True)
    owner = worker(a, phase)
    wait_for(lambda: (project.parent/('barrier-'+phase)).exists())
    owner.kill(); owner.wait()
    worker(a)
    result = terminal(a)
    assert result['state'] == expected, result
    assert len(calls(project)) == len(set(calls(project))) == 4
    if expected == 'reconciliation_required':
        assert result['failures']['image-0']['state'] == 'reconciliation_required'
        assert (a/'assets'/a.name/'audio-1'/'asset.json').exists()


def test_pid_reuse_without_os_ownership_cannot_claim_running(project, worker):
    a = prepare(project)
    atomic_json(a/'started.json', {'pid': os.getpid(), 'process_start': 'unrelated-old-process', 'owner_token': 'old'})
    assert state(a)['state'] == 'pending'
    worker(a)
    result = terminal(a)
    assert result['state'] == 'complete'
    assert result['owner']['owner_token'] != 'old'


def test_preexisting_mp4_and_later_render_cannot_substitute_output(project, worker):
    canonical = project/(project.name+'.mp4')
    canonical.write_bytes((project.parent/'fixture.mp4').read_bytes()+b'old')
    a = prepare(project)
    worker(a)
    first = terminal(a)
    assert first['state'] == 'complete', first
    owned = Path(first['output']['path']); original = owned.read_bytes()
    assert original != (project.parent/'fixture.mp4').read_bytes()+b'old'
    b = prepare(project, 'op-next', force_images=True)
    worker(b)
    assert terminal(b)['state'] == 'complete'
    assert canonical.read_bytes() != original
    assert owned.read_bytes() == original
    assert state(a)['output'] == first['output']


def test_completed_output_survives_death_before_terminal_and_later_canonical(project, worker):
    a = prepare(project)
    child = worker(a, 'before_terminal')
    wait_for(lambda: (project.parent/'barrier-before_terminal').exists())
    output = (a/'output.mp4').read_bytes()
    child.kill(); child.wait()
    (project/(project.name+'.mp4')).write_bytes(b'later canonical')
    worker(a)
    assert terminal(a)['state'] == 'complete'
    assert (a/'output.mp4').read_bytes() == output
    assert len(calls(project)) == 4


def test_output_handoff_holds_project_lock(project, worker):
    a = prepare(project)
    child = worker(a, 'before_output')
    wait_for(lambda: (project.parent/'barrier-before_output').exists())
    with pytest.raises(AssetBusy):
        with supervisor().project_write_lock(project): pass
    (project.parent/'release-before_output').touch()
    assert terminal(a)['state'] == 'complete'


def test_new_launch_checks_changed_input_before_any_dispatch(project, worker):
    a = prepare(project)
    (project/'plan.json').write_text((project/'plan.json').read_text()+'\n')
    worker(a)
    assert terminal(a)['state'] == 'reconciliation_required'
    assert calls(project) == []


def test_process_diagnostics_unavailable_never_blocks_os_owned_execution(monkeypatch):
    s = supervisor()
    def denied(*args, **kwargs): raise PermissionError('ps restricted')
    monkeypatch.setattr(s.subprocess, 'run', denied)
    assert s._process_start(os.getpid()) is None


def test_python_identity_normalizes_directory_aliases_and_preserves_venv(tmp_path, monkeypatch):
    s = supervisor()
    real = tmp_path/'runtime'; (real/'bin').mkdir(parents=True)
    (real/'bin'/'python').symlink_to(sys.executable)
    alias = tmp_path/'alias'; alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(sys, 'executable', str(alias/'bin'/'python'))
    monkeypatch.setattr(sys, 'prefix', str(alias))
    identity = s.release_identity()['python']
    monkeypatch.setattr(sys, 'executable', str(real/'bin'/'python'))
    monkeypatch.setattr(sys, 'prefix', str(real))
    assert s.release_identity()['python'] == identity
    assert identity['executable'] == str(real/'bin'/'python')


def test_same_operation_rejoins_while_render_owns_project(project, worker):
    a = prepare(project)
    worker(a, 'dispatch-image-0')
    wait_for(lambda: (project.parent/'barrier-dispatch-image-0').exists())
    assert prepare(project) == a
    (project.parent/'release-dispatch-image-0').touch()
    assert terminal(a)['state'] == 'complete'


@pytest.mark.parametrize('phase', ['before_output', 'after_output'])
def test_completion_handoff_interruption_uses_same_assets(project, worker, phase):
    a = prepare(project)
    child = worker(a, phase)
    wait_for(lambda: (project.parent/('barrier-'+phase)).exists())
    child.kill(); child.wait()
    worker(a)
    result = terminal(a)
    assert result['state'] == 'complete'
    assert sorted(calls(project)) == ['audio-0','audio-1','image-0','image-1']


def test_missing_renderer_manifest_with_dispatch_evidence_never_reinitializes(project, worker):
    a = prepare(project)
    child = worker(a, 'dispatch-image-0')
    wait_for(lambda: (project.parent/'barrier-dispatch-image-0').exists())
    child.kill(); child.wait()
    (a/'assets'/a.name/'attempt.json').unlink()
    worker(a)
    result = terminal(a)
    assert result['state'] == 'reconciliation_required'
    assert 'attempt' in result['failures']
    assert calls(project) == ['image-0']


def test_distinct_projects_continue_while_one_project_is_owned(project, worker):
    a = prepare(project)
    worker(a, 'dispatch-image-0')
    wait_for(lambda: (project.parent/'barrier-dispatch-image-0').exists())
    second = project.parent/'independent'; second.mkdir()
    (second/'plan.json').write_bytes((project/'plan.json').read_bytes())
    b = prepare(second, 'independent')
    worker(b)
    assert terminal(b)['state'] == 'complete'
    assert state(a)['state'] == 'running'
    (project.parent/'release-dispatch-image-0').touch()
    assert terminal(a)['state'] == 'complete'


def test_other_attempt_same_project_waits_without_dispatch(project, worker):
    a = prepare(project)
    b = prepare(project, 'second', force_images=True)
    worker(a, 'dispatch-image-0')
    wait_for(lambda: (project.parent/'barrier-dispatch-image-0').exists())
    waiting = worker(b); waiting.wait(timeout=10)
    assert state(b)['state'] == 'pending'
    assert calls(project) == ['image-0']
    (project.parent/'release-dispatch-image-0').touch()
    assert terminal(a)['state'] == 'complete'
    worker(b)
    assert terminal(b)['state'] == 'complete'


def test_local_asset_failure_preserves_independent_successes(project, worker):
    a = prepare(project)
    worker(a, 'local-image-0')
    result = terminal(a)
    assert result['state'] == 'local_failure'
    assert result['failures']['image-0']['error_type'] == 'OSError'
    assert sorted(calls(project)) == ['audio-0','audio-1','image-1']


def test_stale_owner_token_cannot_write_terminal(project, worker):
    a = prepare(project)
    child = worker(a, 'converted-image-0')
    wait_for(lambda: (project.parent/'barrier-converted-image-0').exists())
    started = json.loads((a/'started.json').read_text()); started['owner_token'] = 'replacement-token'
    atomic_json(a/'started.json', started)
    (project.parent/'release-converted-image-0').touch()
    child.wait(timeout=10)
    assert not (a/'terminal.json').exists()
    assert not (a/'output.json').exists()
    worker(a)
    assert terminal(a)['state'] == 'complete'
    assert len(calls(project)) == 4


def test_codex_runtime_remains_updateable_and_checked_per_job(project, worker, monkeypatch):
    binary = project.parent/'codex'
    def install(version):
        binary.write_text('#!/bin/sh\nprintf "codex '+version+'\\n"\n')
        binary.chmod(0o755)
    install('1.0')
    monkeypatch.setenv('CODEX_BINARY', str(binary))
    monkeypatch.setenv('LOCAL_EXPLAINER_IMAGE_PROVIDER', 'codex')
    a = prepare(project)
    install('2.0')
    assert prepare(project) == a
    worker(a)
    assert terminal(a)['state'] == 'complete'
    runtime = json.loads((a/'runtime.json').read_text())
    assert runtime['path'] == str(binary)
    assert runtime['version'] == 'codex 2.0'
    assert '1.0' not in (a/'admission.json').read_text()


@pytest.mark.parametrize('field', ['path','sha256','attempt_token','manifest_sha256','media'])
def test_corrupt_completion_evidence_cannot_be_reported_success(project, worker, field):
    a = prepare(project)
    worker(a)
    assert terminal(a)['state'] == 'complete'
    receipt = json.loads((a/'output.json').read_text())
    receipt[field] = 'corrupt'
    atomic_json(a/'output.json', receipt)
    with pytest.raises(ReceiptConflict): state(a)


def test_new_owner_waiting_for_project_does_not_reuse_old_terminal_failure(project, worker):
    a = prepare(project)
    worker(a, 'local-image-0')
    assert terminal(a)['state'] == 'local_failure'
    b = prepare(project, 'project-owner', force_images=True)
    worker(b, 'dispatch-image-0')
    wait_for(lambda: (project.parent/'barrier-dispatch-image-0').exists())
    waiting = worker(a); waiting.wait(timeout=10)
    assert state(a)['state'] == 'pending'
    (project.parent/'release-dispatch-image-0').touch()
    assert terminal(b)['state'] == 'complete'
    worker(a)
    assert terminal(a)['state'] == 'complete'
