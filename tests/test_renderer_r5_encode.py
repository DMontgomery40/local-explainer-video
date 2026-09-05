"""Actual locked-runtime media gates with synthetic local inputs only."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch
from PIL import Image
from md_video_maker import mdvm
from core import pipeline_remotion, remotion_bridge


class RendererR5EncodeTest(unittest.TestCase):
    def test_mixed_short_clip_holds_final_frame_without_restarting(self):
        from md_video_maker import mixed_video_assembly as mixed
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
                'color=red:s=32x32:r=24:d=0.5','-f','lavfi','-i','color=blue:s=32x32:r=24:d=0.5',
                '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0','-c:v','libx264',str(root/'source.mp4')],check=True)
            with wave.open(str(root/'audio.wav'),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\0\0'*48000)
            mixed._make_video_segment({'video_source_path':'source.mp4'}, root, root/'audio.wav', root/'out.mp4', 2.0,32,32)
            raw=subprocess.check_output(['ffmpeg','-v','error','-ss','1.2','-i',str(root/'out.mp4'),
                '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
            self.assertGreater(sum(raw[2::3]),sum(raw[0::3])*3)
            self.assertGreater(mixed.duration(root/'out.mp4'),1.9)

    def test_mixed_profile_dimensions_reach_every_segment(self):
        for width,height in [(36,64),(64,36),(48,48)]:
            with self.subTest(size=(width,height)), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()
                Image.new('RGB',(64,36),'navy').save(root/'still.png')
                subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x36:r=24:d=0.5','-c:v','libx264','-pix_fmt','yuv420p',str(root/'source.mp4')],check=True)
                plan={'meta':{'tts_provider':'openrouter','render_profile':{'width':width,'height':height}},'scenes':[
                    {'id':0,'narration':'Synthetic clip','video_source_path':'source.mp4'},
                    {'id':1,'narration':'Synthetic still','image_source_path':'still.png'}]}
                (root/'plan.json').write_text(json.dumps(plan))
                def audio(scene,path,**kwargs):
                    path=path/"audio"/f"scene_{scene['id']:03d}.wav"
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with wave.open(str(path),'wb') as wav:
                        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\0\0'*12000)
                    return path
                with patch.object(mdvm,'generate_scene_audio',audio):output=mdvm.render_project(root,attempt_id='mixed')
                media=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output)]))
                video=next(s for s in media['streams'] if s['codec_type']=='video')
                self.assertEqual((video['width'],video['height']),(width,height))
                self.assertTrue(any(s['codec_type']=='audio' for s in media['streams']))
                self.assertGreater(float(media['format']['duration']),.8)

    def test_remotion_actual_staged_concat_replaces_only_after_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'audio').mkdir()
            with wave.open(str(root/'audio/scene_000.wav'),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\0\0'*12000)
            source=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x36:r=30:d=0.5','-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True)
            (root/'plan.json').write_text(json.dumps({'scenes':[{'scene_code':'synthetic','narration':'Synthetic'}]}))
            output=root/f'{root.name}.mp4';output.write_bytes(b'previous')
            with patch.object(pipeline_remotion,'_generate_scene_code_with_timing',lambda *a:None), patch.object(remotion_bridge,'duration_frames',return_value=15), patch.object(remotion_bridge,'render_dynamic_scene',side_effect=lambda **kw:shutil.copyfile(source,kw['output_path'])):
                self.assertEqual(pipeline_remotion.run_pipeline(root,skip_tts=True,skip_whisper=True),output)
            self.assertGreater(mdvm.ffprobe_duration(output),0)
            self.assertFalse(list(root.glob('.concat-*')))
