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
