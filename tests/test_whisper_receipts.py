from pathlib import Path
from types import SimpleNamespace
import json
import sys
import pytest
from core import whisper_timestamps as whisper
from core.generation_receipts import UnknownDispatch


@pytest.mark.parametrize('failure', ['none', 'poll', 'accepted-response-lost'])
def test_whisper_recovers_original_prediction_without_paid_replay(tmp_path, monkeypatch, failure):
    audio = tmp_path / 'audio.wav'; audio.write_bytes(b'synthetic audio')
    created = []; retrieved = []
    result = {'text': 'four', 'chunks': [{'text': 'four', 'timestamp': [0, 1]}]}
    class Prediction:
        id = 'original-prediction'
        status = 'starting'
        output = None
        def wait(self):
            if failure == 'poll' and not retrieved: raise TimeoutError('poll lost')
            self.status = 'succeeded'; self.output = result
    prediction = Prediction()
    def create(**kwargs):
        created.append(kwargs)
        if failure == 'accepted-response-lost': raise TimeoutError('accepted but lost response')
        return prediction
    def get(identifier):
        retrieved.append(identifier)
        return prediction
    monkeypatch.setitem(sys.modules, 'replicate', SimpleNamespace(predictions=SimpleNamespace(create=create, get=get)))
    if failure != 'none':
        with pytest.raises(UnknownDispatch): whisper.get_word_timestamps(audio)
    if failure == 'accepted-response-lost':
        with pytest.raises(UnknownDispatch): whisper.get_word_timestamps(audio)
        assert not retrieved
    else:
        assert whisper.get_word_timestamps(audio).words[0].word == 'four'
        assert whisper.get_word_timestamps(audio).duration == 1
        if failure == 'poll': assert retrieved == ['original-prediction']
    assert len(created) == 1


def test_whisper_receipt_keys_complete_input_and_settings(tmp_path, monkeypatch):
    audio = tmp_path / 'audio.wav'; audio.write_bytes(b'first')
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id=f'p{len(calls)}', status='succeeded', output={'text': 'ok', 'chunks': []})
    monkeypatch.setitem(sys.modules, 'replicate', SimpleNamespace(predictions=SimpleNamespace(create=create)))
    whisper.get_word_timestamps(audio)
    whisper.get_word_timestamps(audio)
    whisper.get_word_timestamps(audio, task='translate')
    whisper.get_word_timestamps(audio, batch_size=32)
    audio.write_bytes(b'second')
    whisper.get_word_timestamps(audio)
    assert len(calls) == 4
