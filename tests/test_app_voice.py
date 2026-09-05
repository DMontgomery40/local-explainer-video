import importlib,sys
from unittest.mock import MagicMock
import pytest

class State(dict):
    __getattr__=dict.__getitem__
    __setattr__=dict.__setitem__

@pytest.mark.parametrize('initial',[None,'kokoro','openrouter','openai','elevenlabs'])
def test_actual_app_import_sidebar_and_provider_kwargs(monkeypatch,initial,tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    import dotenv
    monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**k:None)
    st=MagicMock();st.session_state=State();selections=[]
    if initial:st.session_state.update(tts_provider=initial,tts_provider_selector=initial,tts_voice='af_bella')
    def select(label,options,index=0,**kwargs):selections.append((label,list(options)));return options[index]
    st.selectbox.side_effect=select;st.slider.side_effect=lambda *a,**k:k['value'];st.checkbox.side_effect=lambda *a,**k:k.get('value',False)
    monkeypatch.setitem(sys.modules,'streamlit',st);sys.modules.pop('app',None)
    app=importlib.import_module('app');monkeypatch.setattr(app,'PROJECTS_DIR',tmp_path)
    app.init_session_state();app.render_sidebar();kwargs=app._tts_kwargs_from_state()
    provider='openrouter' if initial in (None,'kokoro') else initial
    assert kwargs['tts_provider']==provider
    assert dict(selections)['TTS Provider']==['openrouter','elevenlabs','openai']
    if provider=='openrouter':assert kwargs['voice']=='Charon' and kwargs['speed']==1.0
    if provider=='openai':assert kwargs['voice']=='onyx'
    if provider=='elevenlabs':assert kwargs['voice']=='Antoni' and kwargs['elevenlabs_model_id']==app.DEFAULT_ELEVENLABS_MODEL
    assert st.session_state.tts_provider_selector!='kokoro' if 'tts_provider_selector' in st.session_state else True
    sys.modules.pop('app',None)

@pytest.mark.parametrize("key", [None, "", "   ", "synthetic-key"])
def test_openrouter_setup_matches_generation_gate(monkeypatch, key, tmp_path):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    if key is None: monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else: monkeypatch.setenv("OPENROUTER_API_KEY", key)
    st = MagicMock(); st.session_state = State(tts_provider="openrouter")
    monkeypatch.setitem(sys.modules, "streamlit", st); sys.modules.pop("app", None)
    app = importlib.import_module("app")
    assert app.check_api_keys()["openrouter"] == bool(key and key.strip())
    if key and key.strip(): assert app._tts_kwargs_from_state()["voice"] == "Charon"
    else:
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            app._tts_kwargs_from_state()
    sys.modules.pop("app", None)


@pytest.mark.parametrize('provider,voice', [('openai','onyx'), ('openrouter','Charon'), ('elevenlabs','Antoni')])
def test_actual_qc_configuration_uses_selected_provider_voice(monkeypatch, tmp_path, provider, voice):
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *a, **k: None)
    monkeypatch.setenv('OPENROUTER_API_KEY','synthetic-key')
    st = MagicMock(); st.session_state = State()
    st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    st.selectbox.side_effect = lambda label, options, index=0, **k: options[index]
    st.text_input.side_effect = lambda label, value='', **k: value
    st.number_input.side_effect = lambda *a, **k: k['value']
    st.checkbox.return_value = False
    st.button.side_effect = lambda label, **k: label == 'Run QC + Publish'
    monkeypatch.setitem(sys.modules, 'streamlit', st); sys.modules.pop('app', None)
    app = importlib.import_module('app'); app.init_session_state()
    project = tmp_path/'ZZ_01-01-1900'; project.mkdir()
    plan = {'scenes': []}
    st.session_state.update(plan=plan, project_dir=project, tts_provider=provider, tts_voice='Charon')
    monkeypatch.setattr(app, 'get_video_duration', lambda *a: 0.0)
    captured=[]
    monkeypatch.setattr(app, 'qc_and_publish_project', lambda **k: (captured.append(k['config']) or plan, {}))
    monkeypatch.setattr(app, 'save_plan', lambda *a: None)
    app.render_step_3()
    assert captured and captured[0].tts_voice == voice
    sys.modules.pop('app', None)
