"""Real locked-runtime compositor acceptance through the attempt supervisor."""
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
from core import voice_gen
from md_video_maker import mdvm, supervisor as s


class SupervisorEncodeTest(unittest.TestCase):
    def test_real_encode_handoff_recovery_and_later_attempt_preserve_output(self):
        with tempfile.TemporaryDirectory(prefix='supervisor-encode-') as tmp:
            root = Path(tmp)
            project = root/'project'; project.mkdir()
            def image_response(color):
                buffer = io.BytesIO(); Image.new('RGB',(64,36),color).save(buffer,'PNG')
                return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(buffer.getvalue()).decode())])
            image_call = Mock(side_effect=[image_response('navy'),image_response('red')])
            audio_call = Mock(return_value=SimpleNamespace(content=b'\0\0'*12000,raise_for_status=lambda:None))
            factory = Mock(return_value=SimpleNamespace(images=SimpleNamespace(generate=image_call)))
            plan = {'meta':{'tts_provider':'openrouter','tts_model':'google/gemini-3.1-flash-tts-preview','voice':'Charon',
                            'render_profile':{'width':64,'height':36}},
                    'scenes':[{'id':0,'title':'Synthetic','visual_prompt':'Synthetic frame','narration':'Synthetic narration.'}]}
            source = json.dumps(plan).encode(); (project/'plan.json').write_bytes(source)
            env = {'OPENAI_API_KEY':'','OPENAI_IMAGE_API_KEY':'synthetic','OPENROUTER_API_KEY':'synthetic',
                   'ELEVENLABS_API_KEY':'','REPLICATE_API_TOKEN':'','LOCAL_EXPLAINER_IMAGE_PROVIDER':'openai'}
            with patch.dict(os.environ,env), patch('openai.OpenAI',factory), patch.object(voice_gen.requests,'post',audio_call):
                first = s.prepare_attempt(project,'first',state_dir=root/'state',force_images=True,force_audio=True)
                atomic = s.atomic_json
                def interrupt_terminal(path, value):
                    if path.name == 'terminal.json': raise KeyboardInterrupt('synthetic supervisor death')
                    atomic(path,value)
                with patch.object(s,'atomic_json',interrupt_terminal):
                    with self.assertRaises(KeyboardInterrupt): s.run_attempt(first)
                original = (first/'output.mp4').read_bytes()
                self.assertFalse((first/'terminal.json').exists())
                # A complete handoff does not even re-encode on replacement.
                with patch.object(mdvm,'assemble_video',side_effect=AssertionError('unexpected re-encode')):
                    result = s.run_attempt(first)
                self.assertEqual(result['state'],'complete')
                self.assertGreater(result['output']['media']['duration_seconds'],0)
                self.assertEqual((result['output']['media']['width'],result['output']['media']['height']),(64,36))
                self.assertEqual(image_call.call_count,1)
                self.assertEqual(audio_call.call_count,1)
                second = s.prepare_attempt(project,'second',state_dir=root/'state',force_images=True)
                self.assertEqual(s.run_attempt(second)['state'],'complete')
                self.assertNotEqual((second/'output.mp4').read_bytes(),original)
                self.assertEqual((first/'output.mp4').read_bytes(),original)
                self.assertEqual((project/'plan.json').read_bytes(),source)
                self.assertEqual(s.attempt_snapshot(first)['output'],result['output'])
                self.assertEqual(image_call.call_count,2)
                self.assertEqual(audio_call.call_count,1)


if __name__ == '__main__':
    unittest.main()
