import json
from pathlib import Path
import pytest
from core.generation_receipts import paid_bytes, atomic_bytes

@pytest.mark.parametrize('change', ['narration','provider','voice','speed','identical','skip'])
def test_pipeline_audio_uses_original_request_receipts(tmp_path,monkeypatch,change):
    import pipeline_v2
    from core import voice_gen, whisper_timestamps
    calls=[]
    def generator(text,output_path,*args,**kwargs):
        request={'text':text,**kwargs}
        def dispatch():calls.append(request);return json.dumps(request,sort_keys=True).encode()
        raw=paid_bytes(request,dispatch,output_path=Path(output_path));atomic_bytes(Path(output_path),raw);return output_path
    monkeypatch.setattr(voice_gen,'_generate_with_openrouter',generator)
    monkeypatch.setattr(voice_gen,'_generate_with_openai',generator)
    class Stop(Exception):pass
    monkeypatch.setattr(whisper_timestamps,'get_word_timestamps',lambda *a: (_ for _ in ()).throw(Stop()))
    plan={'scenes':[{'id':0,'narration':'original narration'}]}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));(tmp_path/'audio').mkdir();audio=tmp_path/'audio/scene_000.wav';audio.write_bytes(b'unknown stale wav')
    options={'tts_provider':'openrouter','voice':'Charon','speed':1.0,'skip_qc':True}
    with pytest.raises(Stop):pipeline_v2.run_pipeline(tmp_path,**options)
    assert len(calls)==1 and audio.read_bytes()!=b'unknown stale wav'
    before=audio.read_bytes()
    if change=='narration':plan['scenes'][0]['narration']='changed narration';path.write_text(json.dumps(plan))
    if change=='provider':options['tts_provider']='openai'
    if change=='voice':options['voice']='other voice'
    if change=='speed':options['speed']=1.2
    if change=='skip':options['skip_tts']=True
    with pytest.raises(Stop):pipeline_v2.run_pipeline(tmp_path,**options)
    assert len(calls)==(1 if change in ('identical','skip') else 2)
    assert (audio.read_bytes()==before) is (change in ('identical','skip'))
    with pytest.raises(Stop):pipeline_v2.run_pipeline(tmp_path,**options)
    assert len(calls)==(1 if change in ('identical','skip') else 2)
