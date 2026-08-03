from __future__ import annotations

import pytest

from core.qc_publish import QCPublishConfig, QCPublishError, qc_and_publish_project


def test_qc_and_publish_project_rejects_motion_scenes_before_loading_ground_truth(
    monkeypatch,
    tmp_path,
):
    def should_not_run(**kwargs):
        raise AssertionError("ground truth lookup should not run for motion scenes")

    monkeypatch.setattr("core.qc_publish.load_qeeg_ground_truth", should_not_run)

    config = QCPublishConfig(
        qeeg_dir=tmp_path,
        backend_url="http://127.0.0.1:8000",
        cliproxy_url="http://127.0.0.1:8317",
        cliproxy_api_key="",
    )

    with pytest.raises(QCPublishError, match="no longer supports Cathode native motion"):
        qc_and_publish_project(
            project_dir=tmp_path,
            plan={"scenes": [{"scene_type": "motion"}]},
            patient_id="01-01-1983-0",
            config=config,
        )


def test_infer_patient_id_reads_the_canonical_clinic_id_off_a_project_folder():
    """The renderer publishes into portal_patients/<clinic id>/, so it has to
    recognise the id the clinic actually uses — initials, date of birth, and a
    collision ordinal that starts at 2."""
    from core.qc_publish import infer_patient_id

    assert infer_patient_id("BT_12-11-1963") == "BT_12-11-1963"
    # A tenth collision is two digits and must not be truncated or refused.
    assert infer_patient_id("DK_08-10-1989_10") == "DK_08-10-1989_10"
    # The renderer's own repeat-project suffix still comes off cleanly.
    assert infer_patient_id("BT_12-11-1963__02") == "BT_12-11-1963"
    assert infer_patient_id("DK_08-10-1989_10__02") == "DK_08-10-1989_10"


def test_infer_patient_id_refuses_what_is_not_a_clinic_id():
    from core.qc_publish import infer_patient_id

    # The retired date-of-birth key is not an id any runtime accepts now.
    assert infer_patient_id("12-11-1963-0") is None
    # `_1` never exists: ordinal one is the unsuffixed form.
    assert infer_patient_id("BT_12-11-1963_1") is None
    assert infer_patient_id("scratch-project") is None
    assert infer_patient_id("") is None


def test_batch_regenerate_selects_projects_through_the_shared_reader(tmp_path, capsys):
    """Both entry points have to agree on what a patient project looks like, or
    a render reachable from one is invisible to the other — so batch_regenerate
    uses the same reader rather than its own copy of the pattern."""
    import batch_regenerate

    projects = tmp_path / "projects"
    for name in ("BT_12-11-1963", "DK_08-10-1989_10", "12-11-1963-0"):
        (projects / name).mkdir(parents=True)
        (projects / name / "plan.json").write_text("{}")
    batch_regenerate.PROJECTS_DIR = projects

    found = [p.name for p in batch_regenerate.get_valid_patient_projects()]

    assert found == ["BT_12-11-1963", "DK_08-10-1989_10"]
    # The legacy-named project is named, not silently dropped.
    assert "skipping 12-11-1963-0" in capsys.readouterr().out


def test_split_project_name_separates_patient_from_repeat_project_and_video():
    """`_2` belongs to the patient. `__02` is a repeat project for that same
    patient, and `_v4` a video revision — neither changes who this is."""
    from core.qc_publish import split_project_name

    assert split_project_name("BT_12-11-1963_2__02") == ("BT_12-11-1963_2", 2, None)
    assert split_project_name("BT_12-11-1963__02") == ("BT_12-11-1963", 2, None)
    assert split_project_name("BT_12-11-1963_2__02__03") == ("BT_12-11-1963_2", 3, None)
    assert split_project_name("DK_08-10-1989_10_v4") == ("DK_08-10-1989_10", None, 4)
    assert split_project_name("BT_12-11-1963") == ("BT_12-11-1963", None, None)
    # A legacy name yields no patient, but still reports what it could strip.
    assert split_project_name("02-25-1988-0__02")[0] is None
