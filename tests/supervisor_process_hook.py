"""Imported as sitecustomize by synthetic subprocess tests only."""
import io
import json
import os
from pathlib import Path
import time
import wave

if os.getenv('SUPERVISOR_TEST_ROOT'):
    from PIL import Image
    from core.generation_receipts import paid_bytes
    from md_video_maker import supervisor as s, mdvm
    root = Path(os.environ['SUPERVISOR_TEST_ROOT'])
    phase = os.getenv('SUPERVISOR_TEST_PHASE', '')
    original_json = s.atomic_json

    def barrier(name):
        if phase == name:
            (root/('barrier-'+name)).touch()
            while not (root/('release-'+name)).exists(): time.sleep(.02)

    def atomic(path, value):
        name = Path(path).name
        if name == 'started.json': barrier('before_started')
        if name == 'output.json': barrier('before_output')
        if name == 'terminal.json': barrier('before_terminal')
        original_json(path, value)
        if name == 'started.json': barrier('after_started')
        if name == 'output.json': barrier('after_output')
    s.atomic_json = atomic

    def generate(kind, scene, directory, **kwargs):
        asset = kind+'-'+str(scene['id'])
        if phase == 'local-'+asset: raise OSError('Synthetic local conversion error')
        def dispatch():
            with (root/'dispatches').open('a') as handle:
                handle.write(asset+'\n'); handle.flush(); os.fsync(handle.fileno())
            barrier('dispatch-'+asset)
            return asset.encode()
        raw = paid_bytes({'asset': asset, 'text': scene.get('narration')}, dispatch)
        directory = directory/('images' if kind == 'image' else 'audio')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/('scene_%03d.'%scene['id']+('png' if kind=='image' else 'wav'))
        if kind == 'image': Image.new('RGB', (64,36), 'blue').save(path)
        else:
            with wave.open(str(path),'wb') as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(24000); output.writeframes(b'\0\0'*4800)
        barrier('converted-'+asset)
        return path
    mdvm.generate_scene_image = lambda scene, directory, **kwargs: generate('image', scene, directory, **kwargs)
    mdvm.generate_scene_audio = lambda scene, directory, **kwargs: generate('audio', scene, directory, **kwargs)
    def assemble(scenes, project, *, output_filename, **kwargs):
        path = project/output_filename
        path.write_bytes(Path(os.environ['SUPERVISOR_TEST_FIXTURE']).read_bytes()+output_filename.encode())
        return path
    mdvm.assemble_video = assemble
    # runpy's -m execution otherwise creates a second supervisor module whose
    # globals would bypass these test-only I/O interruption hooks.
    import runpy
    run_module = runpy._run_module_as_main
    def run_as_main(name, alter_argv=True):
        if name == 'md_video_maker.supervisor':
            raise SystemExit(s.main())
        return run_module(name, alter_argv)
    runpy._run_module_as_main = run_as_main
