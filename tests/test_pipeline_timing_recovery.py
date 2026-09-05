import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest


@pytest.mark.parametrize('change', ['same', 'narration', 'cues', 'code'])
def test_timing_pass_receipts_bind_current_input_and_output(tmp_path, monkeypatch, change):
    import anthropic
    from core import pipeline_remotion as pipeline
    client = MagicMock()
    client.messages.stream.return_value.__enter__.return_value = [SimpleNamespace(type='content_block_delta', delta=SimpleNamespace(text='return (<div>timed</div>);'))]
    monkeypatch.setattr(anthropic, 'Anthropic', lambda: client)
    plan = {'scenes': [{'title':'Original', 'narration':'Original speech', 'audio_duration':2, 'cue_points':[{'frame':20}], 'scene_code':'preliminary'}]}
    path = tmp_path/'plan.json'
    pipeline._generate_scene_code_with_timing(plan['scenes'], plan, path)
    assert client.messages.stream.call_count == 1
    assert plan['scenes'][0]['timing_code_receipt']
    plan = json.loads(path.read_text())
    if change == 'narration': plan['scenes'][0]['narration'] = 'Changed narration'
    if change == 'cues': plan['scenes'][0]['cue_points'] = [{'frame':33}]
    if change == 'code': plan['scenes'][0]['scene_code'] = 'operator-edited code'
    pipeline._generate_scene_code_with_timing(plan['scenes'], plan, path)
    expected = 1 if change == 'same' else 2
    assert client.messages.stream.call_count == expected
    pipeline._generate_scene_code_with_timing(plan['scenes'], plan, path)
    assert client.messages.stream.call_count == expected


def test_fresh_preliminary_storyboard_reaches_timing_pass(tmp_path, monkeypatch):
    from core import pipeline_remotion as pipeline, director
    from core import remotion_bridge
    monkeypatch.setattr(director, 'generate_storyboard_api', lambda *a: [{'scene_code':'preliminary', 'narration':'speech'}])
    class Stop(Exception): pass
    calls=[]
    def timing(*args): calls.append(args); raise Stop()
    monkeypatch.setattr(pipeline, '_generate_scene_code_with_timing', timing)
    monkeypatch.setattr(remotion_bridge, 'render_dynamic_scene', lambda **k: pytest.fail('Rendered preliminary code'))
    with pytest.raises(Stop): pipeline.run_pipeline(tmp_path, input_text='synthetic', skip_tts=True, skip_whisper=True)
    assert len(calls) == 1


@pytest.mark.parametrize('module_name', ['pipeline_v2', 'core.pipeline_remotion'])
@pytest.mark.parametrize('stale', [False, True])
def test_required_scene_failure_blocks_final_and_preserves_prior_bytes(tmp_path, monkeypatch, module_name, stale):
    import importlib
    pipeline = importlib.import_module(module_name)
    from core import remotion_bridge, whisper_timestamps, cue_points, video_assembly
    (tmp_path/'audio').mkdir()
    scenes=[]
    for i in range(2):
        audio=tmp_path/'audio'/f'scene_{i:03d}.wav'; audio.write_bytes(b'synthetic wav')
        scene={'id':i,'narration':'speech','title':str(i),'scene_code':'code','audio_path':str(audio)}
        if stale:
            clip=tmp_path/f'old-{i}.mp4';clip.write_bytes(b'old clip');scene['clip_path']=str(clip)
        scenes.append(scene)
    plan={'scenes':scenes}; (tmp_path/'plan.json').write_text(json.dumps(plan))
    prior=tmp_path/f'{tmp_path.name}.mp4';prior.write_bytes(b'prior completed video')
    monkeypatch.setattr(whisper_timestamps, 'get_word_timestamps', lambda *a:SimpleNamespace(duration=1))
    monkeypatch.setattr(cue_points, 'extract_cue_points', lambda *a:[])
    monkeypatch.setattr(remotion_bridge, 'duration_frames', lambda *a:30)
    if module_name.endswith('pipeline_remotion'):
        monkeypatch.setattr(pipeline, '_generate_scene_code_with_timing', lambda *a:None)
    render_calls=[]
    def render(**kwargs):
        render_calls.append(kwargs)
        if kwargs['output_path'].name == 'scene_001.mp4': raise RuntimeError('synthetic scene failure')
        kwargs['output_path'].write_bytes(b'fresh clip')
    monkeypatch.setattr(remotion_bridge, 'render_scene' if module_name=='pipeline_v2' else 'render_dynamic_scene', render)
    monkeypatch.setattr(video_assembly, 'assemble_v2_video', lambda *a,**k:pytest.fail('Assembled around required failure'))
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a,**k:pytest.fail('Muxed around required failure')) if hasattr(pipeline,'subprocess') else None
    options={'skip_tts':True}
    if module_name=='pipeline_v2': options['skip_qc']=True
    with pytest.raises(RuntimeError): pipeline.run_pipeline(tmp_path,**options)
    assert len(render_calls)==2
    assert prior.read_bytes()==b'prior completed video'
    saved=json.loads((tmp_path/'plan.json').read_text())
    assert not saved['scenes'][1].get('clip_path')
    if stale: assert (tmp_path/'old-1.mp4').read_bytes()==b'old clip'
    if module_name == 'pipeline_v2':
        def retry_render(**kwargs):
            kwargs['output_path'].write_bytes(b'current clip')
        monkeypatch.setattr(remotion_bridge, 'render_scene', retry_render)
        assembled=[]
        def assemble(current, *args, **kwargs):
            assembled.extend(current)
            return prior
        monkeypatch.setattr(video_assembly, 'assemble_v2_video', assemble)
        assert pipeline.run_pipeline(tmp_path, **options) == prior
        assert len(assembled) == 2
        assert all(Path(scene['clip_path']).read_bytes() == b'current clip' for scene in assembled)
        assert prior.read_bytes() == b'prior completed video'


def test_timing_response_survives_plan_save_interruption(tmp_path, monkeypatch):
    import anthropic
    from core import pipeline_remotion as pipeline
    client = MagicMock()
    client.messages.stream.return_value.__enter__.return_value = [SimpleNamespace(
        type='content_block_delta', delta=SimpleNamespace(text='return (<div>timed</div>);'))]
    monkeypatch.setattr(anthropic, 'Anthropic', lambda: client)
    plan = {'scenes': [{'narration': 'speech', 'scene_code': 'preliminary'}]}
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    save = pipeline.atomic_json
    def fail_save(*args): raise OSError('synthetic plan save interruption')
    monkeypatch.setattr(pipeline, 'atomic_json', fail_save)
    with pytest.raises(OSError): pipeline._generate_scene_code_with_timing(plan['scenes'], plan, path)
    monkeypatch.setattr(pipeline, 'atomic_json', save)
    plan = json.loads(path.read_text())
    pipeline._generate_scene_code_with_timing(plan['scenes'], plan, path)
    assert client.messages.stream.call_count == 1
    assert json.loads(path.read_text())['scenes'][0]['timing_code_receipt']
