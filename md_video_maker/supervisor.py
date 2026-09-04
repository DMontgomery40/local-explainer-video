"""Detached, receipt-bound renderer attempts.

Only the owner of ``owner.lock`` may execute or write execution receipts. The
renderer owns the independent project lock; its callbacks validate admitted
inputs and bind the new output while that lock is still held. PIDs and wall
clock times are diagnostics, never permission to take ownership or redispatch.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
from urllib.parse import urlsplit

from core.generation_receipts import (AssetBusy, AssetFailures, ReceiptConflict,
    UnknownDispatch, atomic_bytes, atomic_json, digest_bytes, exclusive_lock,
    request_digest)
from core.image_gen import resolve_codex_runtime
from md_video_maker import mdvm

RELEASE_ROOT = Path(__file__).resolve().parents[1]
# Credentials and Codex executable/version are deliberately absent. The latter
# is resolved per job, and can be updated independently of application releases.
CONFIG_DEFAULTS = {
    'LOCAL_EXPLAINER_IMAGE_PROVIDER': 'codex',
    'LOCAL_EXPLAINER_IMAGE_MODEL': 'gpt-image-1',
    'LOCAL_EXPLAINER_IMAGE_QUALITY': 'high',
    'OPENAI_IMAGE_BASE_URL': 'https://api.openai.com/v1',
    'OPENAI_BASE_URL': 'https://api.openai.com/v1',
    'REPLICATE_BASE_URL': 'https://api.replicate.com',
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (ValueError, OSError) as exc:
        raise ReceiptConflict(f'Unreadable receipt: {path}') from exc


def release_identity() -> dict:
    """Application bytes plus lock and Python identity, independent of Git/CLI."""
    files = {}
    for directory in ('core', 'md_video_maker', 'prompts'):
        for path in sorted((RELEASE_ROOT/directory).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.md', '.txt', '.json'} and '__pycache__' not in path.parts:
                if 'tests' not in path.relative_to(RELEASE_ROOT).parts:
                    files[str(path.relative_to(RELEASE_ROOT))] = digest_bytes(path.read_bytes())
    lock = RELEASE_ROOT/'renderer-requirements.lock'
    return {'release_root': str(RELEASE_ROOT), 'source_sha256': request_digest(files),
            'source_files': files, 'lock_sha256': digest_bytes(lock.read_bytes()),
            'python': {'executable': str(Path(sys.executable).parent.resolve()/Path(sys.executable).name), 'version': sys.version,
                       'prefix': str(Path(sys.prefix).resolve())}}


def effective_config() -> dict:
    result = {key: os.getenv(key) or default for key, default in CONFIG_DEFAULTS.items()}
    result['LOCAL_EXPLAINER_CODEX_IMAGE_RUNNER_MODEL'] = (
        os.getenv('LOCAL_EXPLAINER_CODEX_IMAGE_RUNNER_MODEL') or os.getenv('CODEX_IMAGE_RUNNER_MODEL') or '')
    for key, value in result.items():
        if key.endswith('BASE_URL'):
            parsed = urlsplit(value)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f'{key} must be a non-secret endpoint URL')
    return result


@contextmanager
def project_write_lock(project_dir: Path):
    """All plan/source edits use this same nonblocking lock as render_project.

    Hold across the entire read/modify/atomic-write operation. Do not wrap
    render_project with it: render_project acquires this lock itself.
    """
    with exclusive_lock(Path(project_dir).resolve()/'.render-operations'/'project.lock'):
        yield


def _input_identity(project, force_images, force_audio):
    plan_bytes = (project/'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    sources = {}
    for scene in plan.get('scenes', []):
        for key in ('image_source_path', 'video_source_path'):
            if scene.get(key):
                source = mdvm.resolve_plan_path(project, str(scene[key]))
                sources[str(source)] = digest_bytes(source.read_bytes())
    return {'project_dir': str(project), 'plan_sha256': digest_bytes(plan_bytes),
            'local_sources': sources, 'force_images': bool(force_images),
            'force_audio': bool(force_audio), 'release': release_identity(),
            'config': effective_config()}


def prepare_attempt(project_dir: Path, operation_id: str, *, state_dir: Path,
                    force_images: bool = False, force_audio: bool = False) -> Path:
    """Durably admit exact intent; identical operation/input rejoins its Path.

    operation_id is the workbench's stable authorization identity, not text or
    settings. State lives outside the release. No process/provider is launched.
    Raises AssetBusy on concurrent admission/project editing, ReceiptConflict
    on reuse of an operation with different immutable inputs.
    """
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError('A stable nonempty workbench operation_id is required')
    project = Path(project_dir).expanduser().resolve()
    state = Path(state_dir).expanduser().resolve()
    if state.is_relative_to(RELEASE_ROOT):
        raise ValueError('Attempt state must live outside the application release')
    # Deterministic addressing closes death between admission and registration.
    # Different authorized operations always get different attempt directories.
    attempt = state/'attempts'/digest_bytes(operation_id.encode())
    with exclusive_lock(attempt/'admission.lock'):
        identity = _input_identity(project, force_images, force_audio)
        path = attempt/'admission.json'
        if path.exists():
            saved = _read(path)
            if saved['operation_id'] != operation_id or saved['input'] != identity:
                raise ReceiptConflict('Operation was admitted with different immutable input')
            return attempt
        with project_write_lock(project):
            identity = _input_identity(project, force_images, force_audio)
            preexisting = {str(path): digest_bytes(path.read_bytes()) for path in project.glob('*.mp4') if path.is_file()}
            atomic_json(path, {'schema': 1, 'operation_id': operation_id,
                'attempt_id': attempt.name, 'attempt_token': uuid4().hex,
                'project_dir': str(project), 'input': identity,
                'expected_output': str(attempt/'output.mp4'),
                'canonical_output': str(project/(project.name+'.mp4')),
                'preexisting_outputs': preexisting, 'admitted_utc': _now()})
    return attempt


def _admission(attempt):
    admission = _read(attempt/'admission.json')
    if (admission.get('attempt_id') != attempt.name or
        admission.get('expected_output') != str(attempt/'output.mp4') or
        digest_bytes(admission['operation_id'].encode()) != attempt.name):
        raise ReceiptConflict('Attempt address differs from immutable admission')
    return admission


def _process_start(pid):
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None  # Diagnostics may be restricted; the OS lock remains authoritative.


def _busy(path):
    try:
        with exclusive_lock(path):
            return False
    except AssetBusy:
        return True


def _media_facts(path):
    result = subprocess.run([mdvm.tool_binary('ffprobe'), '-v', 'error', '-show_streams',
        '-show_format', '-of', 'json', str(path)], capture_output=True, text=True, timeout=30, check=True)
    probe = json.loads(result.stdout)
    duration = float(probe.get('format', {}).get('duration', 0))
    videos = [stream for stream in probe.get('streams', []) if stream.get('codec_type') == 'video']
    if duration <= 0 or not videos or videos[0].get('width', 0) <= 0 or videos[0].get('height', 0) <= 0:
        raise ReceiptConflict('Attempt output is not a valid video')
    return {'duration_seconds': duration, 'width': videos[0]['width'], 'height': videos[0]['height'],
            'video_codec': videos[0].get('codec_name'), 'size_bytes': path.stat().st_size}


def _output(attempt, admission):
    path = attempt/'output.json'
    if not path.exists():
        return None
    output = _read(path)
    artifact = attempt/'output.mp4'
    if (output.get('attempt_id') != admission['attempt_id'] or
        output.get('attempt_token') != admission['attempt_token'] or
        output.get('admission_sha256') != request_digest(admission) or
        output.get('path') != str(artifact) or
        not artifact.is_file() or digest_bytes(artifact.read_bytes()) != output.get('sha256')):
        raise ReceiptConflict('Attempt-owned output evidence is incompatible or corrupt')
    if _media_facts(artifact) != output.get('media'):
        raise ReceiptConflict('Attempt-owned media facts differ from saved evidence')
    manifest = attempt/'assets'/attempt.name/'attempt.json'
    if digest_bytes(manifest.read_bytes()) != output.get('manifest_sha256'):
        raise ReceiptConflict('Renderer manifest differs from the completed output')
    return output


def attempt_snapshot(attempt_dir: Path) -> dict:
    """Read-only status: pending/running/complete/reconciliation_required/local_failure.

    owner_active is an OS lock observation. A saved live/reused PID alone never
    means running. Pending with recovery_required invites start_attempt on the
    same attempt. No timestamp deadline or provider retry is inferred here.
    """
    attempt = Path(attempt_dir).resolve()
    admission = _admission(attempt)
    active = _busy(attempt/'owner.lock')
    owner = _read(attempt/'started.json') if (attempt/'started.json').exists() else None
    result = {'attempt_id': attempt.name, 'operation_id': admission['operation_id'],
        'attempt_dir': str(attempt), 'state': 'running' if active else 'pending',
        'owner_active': active, 'owner': owner, 'recovery_required': bool(owner and not active),
        'stdout_path': str(attempt/'stdout.log'), 'stderr_path': str(attempt/'stderr.log')}
    terminal = _read(attempt/'terminal.json') if (attempt/'terminal.json').exists() else None
    if terminal and owner and terminal.get('owner_token') == owner.get('owner_token'):
        if terminal.get('admission_sha256') != request_digest(admission):
            raise ReceiptConflict('Terminal receipt differs from immutable admission')
        result.update(state=terminal['state'], failures=terminal.get('failures', {}), recovery_required=False)
        if terminal['state'] == 'complete':
            output = _output(attempt, admission)
            if output is None or terminal.get('output') != output:
                raise ReceiptConflict('Terminal success has no exact output evidence')
            result['output'] = output
    return result


def start_attempt(attempt_dir: Path) -> dict:
    """Launch a detached observer/owner and return immediately; safe to repeat.

    Child-only started/terminal writes close both sides of the parent-death
    handshake. Multiple launchers may spawn lightweight observers, but only one
    OS lock owner can execute. No launch/PID acknowledgment authorizes replay.
    """
    attempt = Path(attempt_dir).resolve()
    snapshot = attempt_snapshot(attempt)
    if snapshot['owner_active'] or snapshot['state'] == 'complete':
        return snapshot
    admission = _admission(attempt)
    release = admission['input']['release']
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [release['release_root'], env.get('PYTHONPATH')]))
    with (attempt/'stdout.log').open('ab', buffering=0) as stdout, (attempt/'stderr.log').open('ab', buffering=0) as stderr:
        subprocess.Popen([release['python']['executable'], '-m', 'md_video_maker.supervisor', 'run', str(attempt)],
            cwd=release['release_root'], env=env, stdin=subprocess.DEVNULL, stdout=stdout,
            stderr=stderr, close_fds=True, start_new_session=True)
    return attempt_snapshot(attempt)


def _failure(exc):
    return {'state': 'reconciliation_required' if isinstance(exc, (UnknownDispatch, ReceiptConflict)) else 'local_failure',
            'error_type': type(exc).__name__}


def run_attempt(attempt_dir: Path) -> dict:
    """Own one attempt until render returns; no parent deadline or paid replay loop."""
    attempt = Path(attempt_dir).resolve()
    admission = _admission(attempt)
    try:
        ownership = exclusive_lock(attempt/'owner.lock')
        ownership.__enter__()
    except AssetBusy:
        return attempt_snapshot(attempt)
    try:
        terminal_path = attempt/'terminal.json'
        if terminal_path.exists() and _read(terminal_path).get('state') == 'complete':
            return attempt_snapshot(attempt)
        owner = {'pid': os.getpid(), 'process_start': _process_start(os.getpid()),
                 'owner_token': uuid4().hex, 'attempt_token': admission['attempt_token'], 'started_utc': _now()}
        atomic_json(attempt/'started.json', owner)
        def write_owned(path, value):
            if _read(attempt/'started.json').get('owner_token') != owner['owner_token']:
                raise ReceiptConflict('Supervisor owner token changed')
            atomic_json(path, value)
        def verify_input():
            accepted = admission['input']
            current = _input_identity(Path(admission['project_dir']), accepted['force_images'], accepted['force_audio'])
            if current != accepted:
                raise ReceiptConflict('Admitted application, lock, config or source changed')
        def locked_input():
            verify_input()
            if admission['input']['config']['LOCAL_EXPLAINER_IMAGE_PROVIDER'] == 'codex':
                write_owned(attempt/'runtime.json', {'owner_token': owner['owner_token'], **resolve_codex_runtime()})
        def bind_output(path):
            if _output(attempt, admission) is not None:
                raise ReceiptConflict('Attempt already has immutable output evidence')
            facts = _media_facts(path)
            payload = path.read_bytes()
            artifact = attempt/'output.mp4'
            atomic_bytes(artifact, payload)
            output = {'attempt_id': attempt.name, 'attempt_token': admission['attempt_token'],
                'admission_sha256': request_digest(admission), 'owner_token': owner['owner_token'],
                'path': str(artifact), 'sha256': digest_bytes(payload), 'media': facts,
                'manifest_sha256': digest_bytes((attempt/'assets'/attempt.name/'attempt.json').read_bytes())}
            write_owned(attempt/'output.json', output)
        try:
            accepted = admission['input']
            if release_identity() != accepted['release'] or effective_config() != accepted['config']:
                raise ReceiptConflict('Admitted application, lock or config changed')
            output = _output(attempt, admission)
            if output is None:
                verify_input()
                asset_attempt = attempt/'assets'/attempt.name
                recovery = asset_attempt.exists() and any(asset_attempt.iterdir())
                mdvm.render_project(Path(admission['project_dir']), attempt_id=attempt.name,
                    operation_dir=attempt/'assets', recovery=recovery,
                    force_images=admission['input']['force_images'], force_audio=admission['input']['force_audio'],
                    on_locked=locked_input, on_output=bind_output)
                output = _output(attempt, admission)
                if output is None:
                    raise ReceiptConflict('Renderer returned without exact output handoff')
            outcome = {'state': 'complete', 'output': output}
        except AssetBusy:
            # A different render owns this project. Preserve pending intent;
            # no terminal failure and no automatic loop or timeout takeover.
            return {'state': 'pending', 'reason': 'project_busy', 'attempt_dir': str(attempt)}
        except Exception as exc:
            failures = exc.failures if isinstance(exc, AssetFailures) else {'attempt': exc}
            mapped = {key: _failure(error) for key, error in failures.items()}
            outcome = {'state': 'reconciliation_required' if any(item['state'] == 'reconciliation_required' for item in mapped.values()) else 'local_failure',
                       'failures': mapped}
        write_owned(terminal_path, {**outcome, 'owner_token': owner['owner_token'],
            'admission_sha256': request_digest(admission), 'finished_utc': _now()})
    finally:
        ownership.__exit__(None, None, None)
    return attempt_snapshot(attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('project_dir', type=Path)
    prepare.add_argument('--operation-id', required=True)
    prepare.add_argument('--state-dir', type=Path, required=True)
    prepare.add_argument('--force-images', action='store_true')
    prepare.add_argument('--force-audio', action='store_true')
    for name in ('start', 'snapshot', 'run'):
        sub.add_parser(name).add_argument('attempt_dir', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            attempt = prepare_attempt(args.project_dir, args.operation_id, state_dir=args.state_dir,
                force_images=args.force_images, force_audio=args.force_audio)
            result = attempt_snapshot(attempt)
        else:
            result = {'start': start_attempt, 'snapshot': attempt_snapshot, 'run': run_attempt}[args.command](args.attempt_dir)
        print(json.dumps(result), flush=True)
        return 0
    except (ReceiptConflict, AssetBusy, ValueError, OSError) as exc:
        print(json.dumps({'error_type': type(exc).__name__, 'error': str(exc)}), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
