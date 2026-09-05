import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from core import rate_limiter, image_gen

@pytest.mark.parametrize('error', [TimeoutError('timeout'), ConnectionResetError('reset'), RuntimeError('500')])
def test_ambiguous_paid_failures_are_never_retried(error, monkeypatch):
    monkeypatch.setattr(rate_limiter.time, 'sleep', lambda _: None)
    call = Mock(side_effect=error)
    with pytest.raises(type(error)):
        rate_limiter.RateLimiter(0, 3).call_with_retry(call)
    assert call.call_count == 1


def test_receipt_reuses_acknowledged_bytes_after_conversion_failure(tmp_path):
    from core.generation_receipts import AssetOperation, paid_bytes
    call = Mock(return_value=b'paid raw audio')
    for _ in range(2):
        with AssetOperation(tmp_path, 'attempt-a', 'audio-0'):
            assert paid_bytes({'provider': 'fake', 'text': 'hello'}, call) == b'paid raw audio'
    assert call.call_count == 1


def test_receipt_unknown_dispatch_parks_and_never_replays(tmp_path):
    from core.generation_receipts import AssetOperation, paid_bytes, UnknownDispatch
    call = Mock(side_effect=TimeoutError('unknown'))
    for _ in range(2):
        with AssetOperation(tmp_path, 'attempt-a', 'audio-0'):
            with pytest.raises(UnknownDispatch):
                paid_bytes({'text': 'same'}, call)
    assert call.call_count == 1


def test_receipt_rejects_changed_request_and_corruption(tmp_path):
    from core.generation_receipts import AssetOperation, paid_bytes, ReceiptConflict
    with AssetOperation(tmp_path, 'a', 'image'):
        paid_bytes({'size': 'large'}, lambda: b'raw')
    with AssetOperation(tmp_path, 'a', 'image'):
        with pytest.raises(ReceiptConflict):
            paid_bytes({'size': 'small'}, lambda: b'wrong')
    record = next(tmp_path.rglob('paid.json'))
    data = json.loads(record.read_text()); data['raw_sha256'] = 'invalid'; record.write_text(json.dumps(data))
    with AssetOperation(tmp_path, 'a', 'image'):
        with pytest.raises(ReceiptConflict):
            paid_bytes({'size': 'large'}, lambda: b'wrong')


def test_codex_runtime_failure_cannot_fall_back_to_paid_api(monkeypatch, tmp_path):
    monkeypatch.setattr(image_gen, '_codex_cli_available', lambda: True)
    monkeypatch.setattr(image_gen, '_run_codex_exec_image', Mock(side_effect=RuntimeError('unknown')))
    paid = Mock(); monkeypatch.setattr(image_gen, '_generate_image_openai', paid)
    monkeypatch.setenv('LOCAL_EXPLAINER_IMAGE_PROVIDER', 'codex')
    with pytest.raises(RuntimeError):
        image_gen.generate_image('hello', tmp_path/'old.png')
    paid.assert_not_called()

@pytest.mark.parametrize('code', [400, 429, 500, None])
def test_image_only_explicit_invalid_size_rejection_uses_auto(monkeypatch, tmp_path, code):
    import base64, io, sys
    from PIL import Image
    blob = io.BytesIO(); Image.new('RGB', (8, 8)).save(blob, 'PNG')
    class Failure(Exception):
        status_code = code
        body = {'error': {'param': 'size', 'code': 'invalid_value'}}
    call = Mock(side_effect=[Failure('size rejected'), SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(blob.getvalue()).decode())])])
    factory = Mock(return_value=SimpleNamespace(images=SimpleNamespace(generate=call)))
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=factory))
    monkeypatch.setenv('OPENAI_IMAGE_API_KEY', 'fake')
    if code == 400:
        image_gen._generate_image_openai(prompt='exact', output_path=tmp_path/'a.png', model='gpt-image-2', target_width=8, target_height=8)
        assert call.call_count == 2
        assert call.call_args.kwargs['size'] == 'auto'
        image_gen._generate_image_openai(prompt='exact', output_path=tmp_path/'a.png', model='gpt-image-2', target_width=8, target_height=8)
        assert call.call_count == 2
    else:
        with pytest.raises(Exception):
            image_gen._generate_image_openai(prompt='exact', output_path=tmp_path/'a.png', model='gpt-image-2', target_width=8, target_height=8)
        assert call.call_count == 1
    assert factory.call_args.kwargs['max_retries'] == 0


def test_elevenlabs_conversion_retry_reuses_paid_raw(monkeypatch, tmp_path):
    from core import voice_gen
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'fake')
    paid = Mock(return_value=SimpleNamespace(content=b'raw-mp3', raise_for_status=lambda: None))
    monkeypatch.setattr(voice_gen.requests, 'post', paid)
    monkeypatch.setattr(voice_gen.elevenlabs_limiter, 'wait', lambda: None)
    convert = Mock(side_effect=[RuntimeError('local conversion failed'), None])
    monkeypatch.setattr(voice_gen, '_convert_mp3_to_wav', convert)
    with pytest.raises(RuntimeError):
        voice_gen._generate_with_elevenlabs(text='exact', output_path=tmp_path/'a.wav')
    voice_gen._generate_with_elevenlabs(text='exact', output_path=tmp_path/'a.wav')
    assert paid.call_count == 1


def test_codex_resolver_skips_broken_path_and_accepts_updated_version(monkeypatch, tmp_path):
    from core.image_gen import resolve_codex_runtime
    bad = tmp_path/'bad'; good=tmp_path/'good'; bad.mkdir(); good.mkdir()
    for path, script in [(bad/'codex', '#!/bin/sh\nexit 1\n'), (good/'codex', '#!/bin/sh\necho codex-cli 99.1\n')]:
        path.write_text(script); path.chmod(0o755)
    monkeypatch.delenv('CODEX_BINARY', raising=False)
    monkeypatch.setenv('PATH', f'{bad}:{good}')
    runtime = resolve_codex_runtime()
    assert runtime == {'path': str(good/'codex'), 'version': 'codex-cli 99.1'}
    (good/'codex').write_text('#!/bin/sh\necho codex-cli 99.2\n')
    assert resolve_codex_runtime()['version'] == 'codex-cli 99.2'
    monkeypatch.setenv('CODEX_BINARY', str(bad/'codex'))
    with pytest.raises(RuntimeError):
        resolve_codex_runtime()


def test_explicit_rejection_retries_but_unknown_does_not(tmp_path, monkeypatch):
    from core.generation_receipts import AssetOperation, paid_bytes
    class Rejected(Exception):
        status_code = 429
    monkeypatch.setattr(rate_limiter.time, 'sleep', lambda _: None)
    call = Mock(side_effect=[Rejected(), b'ack'])
    with AssetOperation(tmp_path, 'a', 'voice'):
        result = rate_limiter.RateLimiter(0, 3).call_with_retry(lambda: paid_bytes({'text':'same'}, call))
    assert result == b'ack'
    assert call.call_count == 2


def test_asset_receipt_hash_detects_valid_but_changed_image(tmp_path):
    from md_video_maker import mdvm
    from core.generation_receipts import ReceiptConflict
    from PIL import Image
    asset = tmp_path/'a.png'; Image.new('RGB', (8, 8), 'red').save(asset)
    mdvm._record_asset(asset, 'source')
    Image.new('RGB', (8, 8), 'blue').save(asset)
    with pytest.raises(ReceiptConflict):
        mdvm._asset_current(asset, 'source')


def test_record_write_failure_is_not_swallowed(monkeypatch, tmp_path):
    from md_video_maker import mdvm
    asset=tmp_path/'a.wav'; asset.write_bytes(b'bytes')
    monkeypatch.setattr(mdvm, '_fingerprint_path', lambda _: tmp_path/'absent'/'s')
    # An unwritable path must prevent successful acceptance.
    (tmp_path/'absent').write_text('file')
    with pytest.raises(OSError):
        mdvm._record_asset(asset, 'source')


def test_legacy_fingerprint_reuses_valid_asset(tmp_path):
    from md_video_maker import mdvm
    from PIL import Image
    asset=tmp_path/'a.png'; Image.new('RGB', (8,8)).save(asset)
    mdvm._fingerprint_path(asset).write_text('source\n')
    assert mdvm._asset_current(asset, 'source')
    Image.new('RGB', (8,8), 'red').save(asset)
    from core.generation_receipts import ReceiptConflict
    with pytest.raises(ReceiptConflict):
        mdvm._asset_current(asset, 'source')


def test_asset_operation_lock_excludes_second_owner(tmp_path):
    from core.generation_receipts import AssetOperation, AssetBusy
    with AssetOperation(tmp_path, 'attempt', 'asset'):
        with pytest.raises(AssetBusy):
            with AssetOperation(tmp_path, 'attempt', 'asset'):
                pass


def test_render_does_not_accept_old_mp4_or_mutate_source_plan(monkeypatch, tmp_path):
    from md_video_maker import mdvm
    from PIL import Image
    import wave
    image=tmp_path/'source.png'; Image.new('RGB',(8,8)).save(image)
    audio=tmp_path/'audio'/'scene_000.wav'; audio.parent.mkdir()
    with wave.open(str(audio),'wb') as w:
        w.setparams((1,2,24000,0,'NONE','not compressed')); w.writeframes(b'\0\0'*240)
    plan={'scenes':[{'id':0,'image_source_path':'source.png','narration':'hello'}]}
    source=json.dumps(plan).encode(); (tmp_path/'plan.json').write_bytes(source)
    old=tmp_path/(tmp_path.name+'.mp4'); old.write_bytes(b'old')
    monkeypatch.setattr(mdvm, '_asset_current', lambda *a, **kw: True)
    monkeypatch.setattr(mdvm, 'assemble_video', lambda *a, **kw: old)
    monkeypatch.setattr(mdvm, 'ffprobe_duration', lambda _: 1.0)
    with pytest.raises(RuntimeError):
        mdvm.render_project(tmp_path)
    assert old.read_bytes()==b'old'
    assert (tmp_path/'plan.json').read_bytes()==source


def test_replicate_acknowledges_prediction_before_free_download(monkeypatch, tmp_path):
    from core import voice_gen
    import sys
    prediction=SimpleNamespace(id='prediction-1', status='succeeded', output='https://fake/audio', wait=lambda:None)
    create=Mock(return_value=prediction)
    client=SimpleNamespace(models=SimpleNamespace(predictions=SimpleNamespace(create=create)), predictions=SimpleNamespace(get=Mock(return_value=prediction)))
    monkeypatch.setitem(sys.modules,'replicate',SimpleNamespace(Client=lambda:client))
    get=Mock(side_effect=[TimeoutError('download'),SimpleNamespace(content=b'audio',raise_for_status=lambda:None)])
    monkeypatch.setattr(voice_gen.requests,'get',get)
    monkeypatch.setattr(voice_gen.image_limiter,'wait',lambda:None)
    with pytest.raises(TimeoutError):
        voice_gen._generate_with_chatterbox('text',tmp_path/'a.wav')
    voice_gen._generate_with_chatterbox('text',tmp_path/'a.wav')
    assert create.call_count==1
    assert (tmp_path/'a.wav').read_bytes()==b'audio'


def test_recovery_unknown_asset_does_not_stop_other_assets(monkeypatch,tmp_path):
    from md_video_maker import mdvm
    from core.generation_receipts import UnknownDispatch
    scene={'id':0,'visual_prompt':'image','narration':'words'}
    (tmp_path/'plan.json').write_text(json.dumps({'scenes':[scene]}))
    generated=[]
    def generate(*args,**kwargs):
        generated.append(args[2])
        if args[2]=='image-0':
            raise UnknownDispatch('unknown image')
    monkeypatch.setattr(mdvm,'_generate_asset',generate)
    with pytest.raises(RuntimeError):
        mdvm.render_project(tmp_path,attempt_id='a')
    assert generated == ['image-0','audio-0']


def test_old_canonical_png_cannot_satisfy_failed_codex_dispatch(monkeypatch,tmp_path):
    from core.generation_receipts import UnknownDispatch
    from PIL import Image
    output=tmp_path/'image.png'; Image.new('RGB',(8,8),'red').save(output); old=output.read_bytes()
    monkeypatch.setattr(image_gen,'resolve_codex_runtime',lambda: {'path':'fake-codex','version':'99'})
    dispatch=Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(image_gen.subprocess,'run',dispatch)
    for _ in range(2):
        with pytest.raises(UnknownDispatch):
            image_gen._run_codex_exec_image(prompt=f'Write {output}',output_path=output,target_width=8,target_height=8,action_id='original-action')
    assert output.read_bytes()==old
    assert dispatch.call_count==1


def test_generation_requires_exact_staged_output_and_force_attempts_are_distinct(tmp_path):
    from md_video_maker import mdvm
    from core.generation_receipts import ReceiptConflict,paid_bytes
    from PIL import Image
    import io
    canonical=tmp_path/'images'/'scene_000.png'; canonical.parent.mkdir(); Image.new('RGB',(8,8),'red').save(canonical)
    with pytest.raises(ReceiptConflict):
        mdvm._generate_asset(tmp_path/'ops','old','image-0',canonical,'s',{},lambda _:canonical,recovery=False)
    old = canonical.read_bytes()
    stream=io.BytesIO(); Image.new('RGB',(8,8),'blue').save(stream,'PNG')
    paid=Mock(return_value=stream.getvalue())
    def generate(staging):
        out=staging/'images'/'scene_000.png'; out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(paid_bytes({'prompt':'same'},paid)); return out
    for attempt in ('a','a','b'):
        mdvm._generate_asset(tmp_path/'ops',attempt,'image-0',canonical,'s',{},generate,recovery=False)
    assert paid.call_count==2
    assert any(path.read_bytes() == old for path in (tmp_path/'ops'/'accepted-assets').rglob('*.png'))


def test_recovery_with_unknown_provenance_does_not_generate(tmp_path):
    from md_video_maker import mdvm
    from core.generation_receipts import UnknownDispatch
    call=Mock()
    with pytest.raises(UnknownDispatch):
        mdvm._generate_asset(tmp_path/'ops','a','image-0',tmp_path/'images'/'a.png','s',{},call,recovery=True)
    call.assert_not_called()


def test_codex_completed_owned_raw_recovers_without_dispatch(monkeypatch,tmp_path):
    from core.generation_receipts import paid_bytes,AssetOperation,UnknownDispatch
    from PIL import Image
    # Simulate kill after the file is complete but before acknowledgement commit.
    output=tmp_path/'image.png'
    runtime={'path':'fake-codex','version':'99'}
    monkeypatch.setattr(image_gen,'resolve_codex_runtime',lambda:runtime)
    def dispatch(*args,**kwargs):
        raw=next((tmp_path/'.codex-output').glob('*/raw.png'),None)
        # The output path is mentioned in the rewritten dispatch prompt.
        path=Path(kwargs['input'].split('Write ',1)[1])
        Image.new('RGB',(8,8),'blue').save(path)
        raise KeyboardInterrupt()
    monkeypatch.setattr(image_gen.subprocess,'run',dispatch)
    with pytest.raises(KeyboardInterrupt):
        image_gen._run_codex_exec_image(prompt=f'Write {output}',output_path=output,target_width=8,target_height=8,action_id='original-action')
    call=Mock(side_effect=AssertionError('must not dispatch'))
    monkeypatch.setattr(image_gen.subprocess,'run',call)
    monkeypatch.setattr(image_gen,'_ensure_png',lambda p:p)
    image_gen._run_codex_exec_image(prompt=f'Write {output}',output_path=output,target_width=8,target_height=8,action_id='original-action')
    assert output.is_file()
    call.assert_not_called()


def test_codex_runtime_is_resolved_once_per_render_scope(monkeypatch,tmp_path):
    from core.image_gen import codex_runtime_scope,resolve_codex_runtime
    executable=tmp_path/'codex'; executable.write_text('#!/bin/sh\necho codex-cli 100\n'); executable.chmod(0o755)
    monkeypatch.setenv('CODEX_BINARY',str(executable))
    run=Mock(return_value=SimpleNamespace(stdout='codex-cli 100'))
    monkeypatch.setattr(image_gen.subprocess,'run',run)
    with codex_runtime_scope():
        resolve_codex_runtime(); resolve_codex_runtime()
    assert run.call_count==1


def test_recovery_checks_explicit_local_source_bytes(monkeypatch,tmp_path):
    from md_video_maker import mdvm
    from core.generation_receipts import ReceiptConflict
    source=tmp_path/'input.png'; source.write_bytes(b'original')
    plan={'scenes':[{'id':0,'image_source_path':'input.png','narration':'words'}]}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(mdvm,'_generate_asset',Mock(side_effect=RuntimeError('stop after manifest')))
    with pytest.raises(RuntimeError):
        mdvm.render_project(tmp_path,attempt_id='a')
    source.write_bytes(b'changed')
    with pytest.raises(ReceiptConflict):
        mdvm.render_project(tmp_path,attempt_id='a',recovery=True)


def test_all_assets_have_prepared_intent_before_first_dispatch(monkeypatch,tmp_path):
    from md_video_maker import mdvm
    plan={'scenes':[{'id':i,'visual_prompt':'image','narration':'words'} for i in range(2)]}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    def generate(*args,**kwargs):
        for asset in ('image-0','audio-0','image-1','audio-1'):
            assert (tmp_path/'.render-operations'/'a'/asset/'prepared.json').is_file()
        raise RuntimeError('stop')
    monkeypatch.setattr(mdvm,'_generate_asset',generate)
    with pytest.raises(RuntimeError) as error:
        mdvm.render_project(tmp_path,attempt_id='a')
    assert all(isinstance(exc, RuntimeError) and not isinstance(exc, AssertionError) for exc in error.value.failures.values())


def test_missing_acknowledgement_after_dispatch_never_becomes_unstarted(tmp_path):
    from core.generation_receipts import AssetOperation,paid_bytes,UnknownDispatch
    call=Mock(return_value=b'raw')
    with AssetOperation(tmp_path,'a','audio'):
        paid_bytes({'text':'same'},call)
    (tmp_path/'a'/'audio'/'paid.json').unlink()
    with AssetOperation(tmp_path,'a','audio'):
        with pytest.raises(UnknownDispatch):
            paid_bytes({'text':'same'},call)
    assert call.call_count==1


@pytest.fixture
def synthetic_render(monkeypatch, tmp_path):
    """Exercise real receipt/asset code with synthetic paid bytes and local media."""
    import io
    import wave
    from PIL import Image
    from core.generation_receipts import paid_bytes
    from md_video_maker import mdvm

    plan = {'meta': {'render_profile': {'width': 8, 'height': 8}},
            'scenes': [{'id': n, 'visual_prompt': f'image {n}', 'narration': f'words {n}'}
                       for n in range(2)]}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    paid_calls = []

    def generate_image(scene, staging, **kwargs):
        quality = image_gen._image_quality()
        def dispatch():
            paid_calls.append(('image', scene['id'], quality))
            buffer = io.BytesIO()
            Image.new('RGB', (8, 8), 'red' if quality == 'low' else 'blue').save(buffer, 'PNG')
            return buffer.getvalue()
        output = staging / 'images' / f"scene_{scene['id']:03d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(paid_bytes({'prompt': scene['visual_prompt'], 'quality': quality}, dispatch))
        return output

    def generate_audio(scene, staging, **kwargs):
        def dispatch():
            paid_calls.append(('audio', scene['id']))
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as audio:
                audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\0\0' * 240)
            return buffer.getvalue()
        output = staging / 'audio' / f"scene_{scene['id']:03d}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(paid_bytes({'narration': scene['narration']}, dispatch))
        return output

    def assemble(scenes, project, *, output_filename, **kwargs):
        output = project / output_filename
        output.write_bytes(b'synthetic compositor output')
        return output

    monkeypatch.setattr(mdvm, 'generate_scene_image', generate_image)
    monkeypatch.setattr(mdvm, 'generate_scene_audio', generate_audio)
    monkeypatch.setattr(mdvm, 'assemble_video', assemble)
    monkeypatch.setattr(mdvm, 'ffprobe_duration', lambda _: 1.0)
    monkeypatch.setenv('LOCAL_EXPLAINER_IMAGE_QUALITY', 'low')
    return mdvm, tmp_path, paid_calls


@pytest.mark.parametrize('interruption', ['before_receipt', 'before_fingerprint', 'after_fingerprint'])
def test_same_source_new_settings_recovers_each_promotion_sidecar_gap(synthetic_render, monkeypatch, interruption):
    from core.generation_receipts import AssetFailures, digest_bytes
    mdvm, project, paid_calls = synthetic_render
    mdvm.render_project(project, attempt_id='old')
    canonical = project / 'images' / 'scene_000.png'
    old_source = mdvm._fingerprint_path(canonical).read_bytes()
    old_bytes = canonical.read_bytes()
    monkeypatch.setenv('LOCAL_EXPLAINER_IMAGE_QUALITY', 'high')
    real_json, real_bytes = mdvm.atomic_json, mdvm.atomic_bytes
    receipt = canonical.with_name(canonical.name + '.receipt.json')
    fingerprint = mdvm._fingerprint_path(canonical)

    def interrupt_json(path, value):
        if path == receipt and interruption == 'before_receipt':
            raise OSError('interrupted before receipt')
        return real_json(path, value)

    def interrupt_bytes(path, value):
        if path == fingerprint and interruption == 'before_fingerprint':
            raise OSError('interrupted before fingerprint')
        result = real_bytes(path, value)
        if path == fingerprint and interruption == 'after_fingerprint':
            raise OSError('interrupted after fingerprint')
        return result

    with monkeypatch.context() as fault:
        fault.setattr(mdvm, 'atomic_json', interrupt_json)
        fault.setattr(mdvm, 'atomic_bytes', interrupt_bytes)
        with pytest.raises(AssetFailures) as failure:
            mdvm.render_project(project, attempt_id='new')
        assert set(failure.value.failures) == {'image-0'}

    operation = project / '.render-operations' / 'new' / 'image-0'
    completed = json.loads((operation / 'asset.json').read_text())
    exact = operation / 'staging' / 'images' / canonical.name
    assert canonical.read_bytes() != old_bytes
    assert digest_bytes(exact.read_bytes()) == completed['output_sha256']
    assert mdvm._fingerprint_path(canonical).read_bytes() == old_source
    before_recovery = list(paid_calls)
    mdvm.render_project(project, attempt_id='new', recovery=True)
    assert paid_calls == before_recovery
    assert canonical.read_bytes() == exact.read_bytes()
    assert mdvm._asset_current(canonical, completed['source_digest'], completed['settings'])


@pytest.mark.parametrize('failure_kind', ['hash', 'format', 'receipt_json', 'legacy_registration'])
@pytest.mark.parametrize('asset_kind', ['image', 'audio'])
def test_reuse_validation_failures_are_isolated_without_regeneration(synthetic_render, monkeypatch, failure_kind, asset_kind):
    from core.generation_receipts import AssetFailures
    mdvm, project, paid_calls = synthetic_render
    mdvm.render_project(project, attempt_id='old')
    folder, suffix = ('images', '.png') if asset_kind == 'image' else ('audio', '.wav')
    canonical = project / folder / ('scene_000' + suffix)
    receipt = canonical.with_name(canonical.name + '.receipt.json')
    if failure_kind == 'hash':
        canonical.write_bytes(b'corrupted bytes')
    elif failure_kind == 'format':
        receipt.unlink()
        canonical.write_bytes(b'not valid media')
    elif failure_kind == 'receipt_json':
        receipt.write_text('{invalid json')
    else:
        receipt.unlink()
        real_json = mdvm.atomic_json
        def fail_legacy_registration(path, payload):
            if path == receipt:
                raise OSError('legacy receipt registration failed')
            return real_json(path, payload)
        monkeypatch.setattr(mdvm, 'atomic_json', fail_legacy_registration)
    # Independent scene assets are missing and need work in the new accepted attempt.
    for path in (project / 'images' / 'scene_001.png', project / 'audio' / 'scene_001.wav'):
        path.unlink()
    paid_calls.clear()
    with pytest.raises(AssetFailures) as failure:
        mdvm.render_project(project, attempt_id='new')
    assert set(failure.value.failures) == {f'{asset_kind}-0'}
    assert ('image', 1, 'low') in paid_calls
    assert ('audio', 1) in paid_calls
    assert not any(call[0] == asset_kind and call[1] == 0 for call in paid_calls)
    for asset in ('image-1', 'audio-1'):
        assert json.loads((project / '.render-operations' / 'new' / asset / 'asset.json').read_text())['output_sha256']


@pytest.mark.parametrize('corruption', ['missing_staged', 'staged_hash', 'staged_format', 'operation_identity', 'operation_json'])
def test_recovery_never_uses_valid_canonical_as_substitute_for_corrupt_completed_evidence(synthetic_render, corruption):
    from core.generation_receipts import AssetFailures, digest_bytes
    mdvm, project, paid_calls = synthetic_render
    mdvm.render_project(project, attempt_id='old')
    canonical = project / 'images' / 'scene_000.png'
    original = canonical.read_bytes()
    operation = project / '.render-operations' / 'old' / 'image-0'
    exact = operation / 'staging' / 'images' / canonical.name
    manifest = operation / 'asset.json'
    record = json.loads(manifest.read_text())
    if corruption == 'missing_staged':
        exact.unlink()
    elif corruption == 'staged_hash':
        exact.write_bytes(b'unrelated staged bytes')
    elif corruption == 'staged_format':
        exact.write_bytes(b'not an image')
        record['output_sha256'] = digest_bytes(exact.read_bytes())
        manifest.write_text(json.dumps(record))
    elif corruption == 'operation_identity':
        record['source_digest'] = 'unrelated-source'
        manifest.write_text(json.dumps(record))
    else:
        manifest.write_text('{broken JSON')
    paid_calls.clear()
    with pytest.raises(AssetFailures) as failure:
        mdvm.render_project(project, attempt_id='old', recovery=True)
    assert set(failure.value.failures) == {'image-0'}
    assert paid_calls == []
    assert canonical.read_bytes() == original

@pytest.mark.parametrize('explicit', [False,True])
def test_image_actions_fresh_but_explicit_retries_stable(monkeypatch,tmp_path,explicit):
    from core.generation_receipts import paid_bytes,atomic_bytes
    calls=[]
    monkeypatch.setenv('LOCAL_EXPLAINER_IMAGE_PROVIDER','openai')
    def provider(*,prompt,output_path,**kwargs):
        def dispatch():calls.append(prompt);return str(len(calls)).encode()
        atomic_bytes(output_path,paid_bytes({'prompt':prompt},dispatch,output_path=output_path));return output_path
    monkeypatch.setattr(image_gen,'_generate_image_openai',provider)
    scene={'id':0,'visual_prompt':'unchanged'}
    for _ in range(2):image_gen.generate_scene_image(scene,tmp_path,**({'action_id':'action-one'} if explicit else {}))
    assert len(calls)==(1 if explicit else 2)
    image_gen.generate_scene_image(scene,tmp_path,action_id='action-two')
    assert len(calls)==(2 if explicit else 3)
