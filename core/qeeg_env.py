"""Where the qEEG engine and its services live.

Standard library only, deliberately: the batch scripts publish into the
engine's portal folder and then ask that engine to sync it, so they have to
agree with the publisher about which installation that is — without importing
the render pipeline to find out.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    # core/qeeg_env.py -> core/ -> repo root
    return Path(__file__).resolve().parents[1]


def default_qeeg_analysis_dir() -> Path:
    env = os.getenv("QEEG_ANALYSIS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root().parent / "qEEG-analysis").resolve()


def default_qeeg_backend_url() -> str:
    return os.getenv("QEEG_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def default_cliproxy_url() -> str:
    return os.getenv("CLIPROXY_BASE_URL", "http://127.0.0.1:8317").rstrip("/")
