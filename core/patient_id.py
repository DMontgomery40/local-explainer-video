"""Reading the clinic's patient ID off a project folder name.

The one place this is defined. It lives on its own, with no imports beyond the
standard library, so the standalone batch scripts can use it without dragging in
the render pipeline — the reason the second copy existed in the first place, and
the reason the two had already drifted apart from each other.

The ID is two initials, a date of birth, and a collision ordinal that starts at
two: ``BT_12-11-1963``, ``BT_12-11-1963_10``. Ordinal one is the unsuffixed
form, so ``_1`` never exists.
"""

from __future__ import annotations

import re

PATIENT_ID_RE = re.compile(r"^[A-Z]{2}_\d{2}-\d{2}-\d{4}(?:_(?:[2-9]|[1-9]\d+))?$")

# Three different things end a project folder name with an underscore and a
# number, and only one of them belongs to the patient:
#   `_2`    the patient's collision ordinal — part of the ID
#   `__02`  a repeat project for the same patient, zero-padded, may stack
#   `_v4`   a video revision, sometimes capitalised, sometimes decimal
# Widening the ID pattern to swallow `__NN` would make `BT_12-11-1963_2` and
# `BT_12-11-1963__02` indistinguishable — a different patient versus a second
# project for the same one. So strip the suffixes that are not the patient's,
# outermost first, and only then ask whether what remains is an ID.
PROJECT_VERSION_SUFFIX_RE = re.compile(r"__(?P<version>\d+)$")
VIDEO_VERSION_SUFFIX_RE = re.compile(
    r"[_ ]v(?P<version>\d+(?:\.\d+)?)$", re.IGNORECASE
)


def is_patient_id(value: str) -> bool:
    return bool(PATIENT_ID_RE.fullmatch(str(value or "").strip()))


def split_project_name(project_name: str) -> tuple[str | None, int | None, float | None]:
    """Separate a project folder name into patient, project version, video version.

    ``BT_12-11-1963_2__02`` is the second project for patient
    ``BT_12-11-1963_2`` — not patient ``BT_12-11-1963`` and not project 2 of
    anyone else. Returns ``(None, …)`` when what is left is not a clinic ID.
    """
    remaining = str(project_name or "").strip()

    project_version: int | None = None
    # Repeat projects stack (`__02__03`), so peel every one of them.
    while (match := PROJECT_VERSION_SUFFIX_RE.search(remaining)) is not None:
        if project_version is None:
            project_version = int(match.group("version"))
        remaining = remaining[: match.start()]

    video_version: float | None = None
    if (match := VIDEO_VERSION_SUFFIX_RE.search(remaining)) is not None:
        raw = match.group("version")
        video_version = float(raw) if "." in raw else int(raw)
        remaining = remaining[: match.start()]

    if not is_patient_id(remaining):
        return None, project_version, video_version
    return remaining, project_version, video_version


def infer_patient_id(project_name: str) -> str | None:
    """Read the clinic patient ID off a project folder name."""
    return split_project_name(project_name)[0]
