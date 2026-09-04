"""Real local encode acceptance, runnable in the freshly locked runtime."""
import base64
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from core import image_gen, voice_gen
from md_video_maker import mdvm


class RendererReceiptEncodeTest(unittest.TestCase):
    def test_real_encode_recovery_preserves_sources_and_old_video(self):
        with tempfile.TemporaryDirectory(prefix='renderer-receipt-') as tmp:
            project=Path(tmp)
            buffer=io.BytesIO(); Image.new('RGB',(64,36),'navy').save(buffer,'PNG')
            image_call=Mock(return_value=SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(buffer.getvalue()).decode())]))
            audio_call=Mock(return_value=SimpleNamespace(content=b'\0\0'*12000,raise_for_status=lambda:None))
            factory=Mock(return_value=SimpleNamespace(images=SimpleNamespace(generate=image_call)))
            plan={'meta':{'tts_provider':'openrouter','tts_model':'google/gemini-3.1-flash-tts-preview','voice':'Charon','render_profile':{'width':64,'height':36}},
                  'scenes':[{'id':0,'title':'Synthetic','visual_prompt':'Blue test frame','narration':'Synthetic narration.'}]}
            source=json.dumps(plan).encode(); (project/'plan.json').write_bytes(source)
            env={'OPENAI_API_KEY':'','OPENAI_IMAGE_API_KEY':'synthetic','OPENROUTER_API_KEY':'synthetic','ELEVENLABS_API_KEY':'','REPLICATE_API_TOKEN':'',
                 'LOCAL_EXPLAINER_IMAGE_PROVIDER':'openai'}
            with patch.dict(os.environ,env), patch('openai.OpenAI',factory), patch.object(voice_gen.requests,'post',audio_call):
                first=mdvm.render_project(project,attempt_id='a')
                original=first.read_bytes()
                self.assertGreater(mdvm.ffprobe_duration(first),0)
                self.assertEqual((project/'plan.json').read_bytes(),source)
                # Recovery reuses exact accepted assets and encodes only locally.
                mdvm.render_project(project,attempt_id='a',recovery=True)
                self.assertEqual(image_call.call_count,1)
                self.assertEqual(audio_call.call_count,1)
                archives=list((project/'.v1-videos').glob('*.mp4'))
                self.assertTrue(any(path.read_bytes()==original for path in archives))
                self.assertEqual((project/'plan.json').read_bytes(),source)
                self.assertTrue((project/'rendered-plan.json').is_file())
                self.assertEqual(factory.call_args.kwargs['max_retries'],0)


if __name__=='__main__':
    unittest.main()
