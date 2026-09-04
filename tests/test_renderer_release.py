"""The clinic renderer runs from a clean checkout, independent of its old home."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def test_renderer_entrypoints_from_unrelated_directory(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for arguments in ([str(REPO / "md_video_maker" / "mdvm.py"), "--help"],
                      ["-m", "md_video_maker.mdvm", "--help"]):
        result = subprocess.run([sys.executable, *arguments], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert "render" in result.stdout


def test_relocated_renderer_imports_only_its_own_helpers(tmp_path):
    relocated = tmp_path / "independent-release"
    relocated.mkdir()
    for directory in ("core", "md_video_maker"):
        shutil.copytree(REPO / directory, relocated / directory,
                        ignore=shutil.ignore_patterns("__pycache__"))
    code = (
        "import json; from md_video_maker import mdvm; "
        "print(json.dumps([str(mdvm.LOCAL_EXPLAINER_ROOT), "
        "mdvm.generate_scene_image.__code__.co_filename, "
        "mdvm.generate_scene_audio.__code__.co_filename, "
        "mdvm.assemble_video.__code__.co_filename]))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(relocated)},
                            capture_output=True, text=True, timeout=30, check=True)
    for location in json.loads(result.stdout):
        assert Path(location).resolve().is_relative_to(relocated.resolve()), location
