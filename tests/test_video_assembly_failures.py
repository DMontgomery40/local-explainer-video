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
