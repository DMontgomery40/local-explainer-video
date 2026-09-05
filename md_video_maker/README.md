# Clinic markdown renderer

This is the versioned runner previously kept in `markdown-video-experiment/md-video-maker`. It calls this repository's existing `core` image, narration and assembly helpers. `SOURCE_SNAPSHOT.json` records the imported source hashes. The initial move changes only repository/import resolution; VIDEO_BRAIN.md and GOOD_EXAMPLES.md retain their original bytes.

## Install a pinned renderer environment

Use Python 3.10 and uv from the repository root:

```sh
uv venv --python python3.10 .renderer-venv
uv pip sync --python .renderer-venv/bin/python --require-hashes renderer-requirements.lock
.renderer-venv/bin/python md_video_maker/mdvm.py --help
```

`renderer-requirements.in` lists direct dependencies; `renderer-constraints.txt` preserves the working environment's resolved versions. To reproduce the lock:

```sh
uv pip compile --python-version 3.10 --generate-hashes --constraint renderer-constraints.txt renderer-requirements.in --output-file renderer-requirements.lock
```

The runner works both as a script and with `python -m md_video_maker.mdvm`. Pass an explicit existing project directory. Projects and credentials live outside this source release. Install ffmpeg and ffprobe for assembly; the workbench also uses HandBrakeCLI for its delivery encoding. The configured image provider can require the authenticated Codex CLI. Record those executable versions with each deployed release; Python's lock does not manage them.

The existing plan selects narration and image behavior. Rendering may spend money; run it only for an authorized generation request. CLI help and the synthetic tests below do not call providers.

## Local validation

```sh
python -m pytest -q tests/test_renderer_release.py md_video_maker/tests/test_mixed_video_assembly.py
.renderer-venv/bin/python -m unittest discover -s md_video_maker/tests -p 'test*.py'
```

These checks exercise relocated imports and real ffmpeg audio/video assembly using synthetic media. They do not evaluate clinical content or generated image/voice quality. The separate legacy Streamlit director uses its own project reference files and broader dependencies; it is not part of this minimal renderer environment.
