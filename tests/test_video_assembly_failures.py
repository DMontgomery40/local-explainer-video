import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from core import video_assembly as video


@pytest.mark.parametrize('failure',['exit','timeout','missing','empty','probe-error','bad-duration','no-audio','success'])
def test_v2_concat_failure_preserves_canonical(tmp_path,monkeypatch,failure):
    old=tmp_path/'out.mp4';old.write_bytes(b'original')
    clip=tmp_path/'clip.mp4';clip.write_bytes(b'clip');audio=tmp_path/'audio.wav';audio.write_bytes(b'audio')
    monkeypatch.setattr(video,'_pick_encoder',lambda:(['-c:v','libx264'],'test'))
    monkeypatch.setattr(video,'_get_media_duration',lambda p:1)
    monkeypatch.setattr(video,'_mux_segment',lambda c,a,out,*args:out.write_bytes(b'segment'))
    outputs=[]
    def run(cmd,**kwargs):
        if '-show_streams' in cmd:
            media={'format':{'duration':'nan' if failure=='bad-duration' else '1'},'streams':[{'codec_type':'video','width':64,'height':36}]+([] if failure=='no-audio' else [{'codec_type':'audio'}])}
            return SimpleNamespace(returncode=1 if failure=='probe-error' else 0,stdout=json.dumps(media),stderr='')
        out=Path(cmd[-1]);outputs.append(out)
        if failure=='timeout':raise video.subprocess.TimeoutExpired(cmd,300)
        if failure!='missing':out.write_bytes(b'' if failure=='empty' else b'new')
        return SimpleNamespace(returncode=1 if failure=='exit' else 0,stderr='failed')
    monkeypatch.setattr(video.subprocess,'run',run)
    call=lambda:video.assemble_v2_video([{'clip_path':str(clip),'audio_path':str(audio)}],tmp_path,output_filename='out.mp4')
    if failure=='success':
        assert call()==old;assert old.read_bytes()==b'new'
        assert any(p.read_bytes()==b'original' for p in (tmp_path/'.v1-videos').glob('*'))
    else:
        with pytest.raises((ValueError,RuntimeError,video.subprocess.TimeoutExpired)):call()
        assert old.read_bytes()==b'original';assert not (tmp_path/'.v1-videos').exists()
    assert outputs[0]!=old and not outputs[0].exists()


@pytest.mark.parametrize('position', [0, 1, 2])
@pytest.mark.parametrize('field', ['clip_path', 'audio_path'])
@pytest.mark.parametrize('failure', ['missing', 'directory', 'empty', 'removed-during-mux'])
def test_v2_requires_every_planned_input(tmp_path, monkeypatch, position, field, failure):
    original = tmp_path / 'out.mp4'; original.write_bytes(b'original')
    scenes = []
    for i in range(3):
        scene = {}
        for key in ['clip_path', 'audio_path']:
            path = tmp_path / f'{i}-{key}'; path.write_bytes(b'input')
            scene[key] = str(path)
        scenes.append(scene)
    target = Path(scenes[position][field])
    if failure == 'missing': target.unlink()
    if failure == 'directory': target.unlink(); target.mkdir()
    if failure == 'empty': target.write_bytes(b'')
    monkeypatch.setattr(video, '_pick_encoder', lambda: ([], 'test'))
    monkeypatch.setattr(video, '_get_media_duration', lambda path: 1)
    muxes = []
    def mux(clip, audio, output, *args):
        muxes.append(output)
        if failure == 'removed-during-mux' and len(muxes) == position + 1:
            target.unlink()
        if not clip.is_file() or not audio.is_file():
            raise RuntimeError('Input disappeared while muxing')
        output.write_bytes(b'new-segment')
    monkeypatch.setattr(video, '_mux_segment', mux)
    def no_concat(*args, **kwargs):
        pytest.fail('Incomplete plan reached concat')
    monkeypatch.setattr(video.subprocess, 'run', no_concat)
    with pytest.raises((ValueError, RuntimeError)):
        video.assemble_v2_video(scenes, tmp_path, output_filename='out.mp4')
    assert original.read_bytes() == b'original'
    assert not (tmp_path / '.v1-videos').exists()
    assert all(not output.exists() for output in muxes)


@pytest.mark.parametrize('position', [0, 1, 2])
@pytest.mark.parametrize('failure', ['missing', 'empty', 'exit', 'timeout'])
def test_v2_requires_fresh_complete_mux_output(tmp_path, monkeypatch, position, failure):
    original = tmp_path / 'out.mp4'; original.write_bytes(b'original')
    stale = tmp_path / 'tmp_segments'; stale.mkdir()
    for i in range(3): (stale / f'seg_{i:03d}.mp4').write_bytes(b'stale')
    clip = tmp_path / 'clip.mp4'; clip.write_bytes(b'clip')
    audio = tmp_path / 'audio.wav'; audio.write_bytes(b'audio')
    monkeypatch.setattr(video, '_pick_encoder', lambda: ([], 'test'))
    monkeypatch.setattr(video, '_get_media_duration', lambda path: 1)
    outputs = []
    def run(cmd, **kwargs):
        assert '-f' not in cmd, 'Incomplete plan reached concat'
        output = Path(cmd[-1]); outputs.append(output)
        if len(outputs) == position + 1:
            if failure == 'timeout': raise video.subprocess.TimeoutExpired(cmd, 120)
            if failure == 'empty': output.write_bytes(b'')
            return SimpleNamespace(returncode=1 if failure == 'exit' else 0, stderr='failed')
        output.write_bytes(b'fresh')
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(video.subprocess, 'run', run)
    with pytest.raises((RuntimeError, video.subprocess.TimeoutExpired)):
        video.assemble_v2_video([{'clip_path': str(clip), 'audio_path': str(audio)}] * 3,
                                tmp_path, output_filename='out.mp4')
    assert original.read_bytes() == b'original'
    assert not (tmp_path / '.v1-videos').exists()
    assert all(output.parent != stale and not output.exists() for output in outputs)
    assert all(path.read_bytes() == b'stale' for path in stale.iterdir())
@pytest.mark.parametrize('clip,start,narration,allowed', [
    (10,0,12,True), (10,4,11,True), (10,4,11.1,False), (10,10,1,False),
    (10,-1,1,False), (float('nan'),0,1,False), (10,0,float('inf'),False)])
def test_mixed_clip_freezes_with_bounded_tail(monkeypatch, tmp_path, clip, start, narration, allowed):
    from md_video_maker import mixed_video_assembly as mixed
    (tmp_path/'clip.mp4').write_bytes(b'fixture')
    monkeypatch.setattr(mixed, 'duration', lambda _: clip)
    commands=[]
    monkeypatch.setattr(mixed.subprocess, 'run', lambda cmd, **kw: commands.append(cmd))
    scene={'video_source_path':'clip.mp4', 'video_start_seconds':start}
    if allowed:
        mixed._make_video_segment(scene,tmp_path,tmp_path/'audio.wav',tmp_path/'out.mp4',narration)
        cmd=commands[0]
        assert '-stream_loop' not in cmd
        assert 'tpad=stop_mode=clone' in cmd[cmd.index('-vf')+1]
    else:
        with pytest.raises(ValueError):
            mixed._make_video_segment(scene,tmp_path,tmp_path/'audio.wav',tmp_path/'out.mp4',narration)
        assert not commands
