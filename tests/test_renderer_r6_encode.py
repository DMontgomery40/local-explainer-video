"""Actual local V2 encode, no provider dispatch."""
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch
from core import video_assembly as video


class V2EncodeTest(unittest.TestCase):
    def test_actual_v2_staged_encode_preserves_original_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();clip=root/'clip.mp4';audio=root/'audio.wav'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x36:r=30:d=0.5','-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True)
            with wave.open(str(audio),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\0\0'*12000)
            output=root/'output.mp4';original=clip.read_bytes();output.write_bytes(original)
            with patch.object(video,'_pick_encoder',return_value=(['-c:v','libx264','-pix_fmt','yuv420p'],'libx264')):
                self.assertEqual(video.assemble_v2_video([{'clip_path':str(clip),'audio_path':str(audio)}] * 3,root,output_filename='output.mp4'),output)
            self.assertGreater(video._get_media_duration(output),1.4)
            self.assertTrue(any(p.read_bytes()==original for p in (root/'.v1-videos').glob('*')))
            self.assertFalse(list(root.glob('.v2-concat-*')))
            self.assertFalse(list(root.glob('.v2-segments-*')))
