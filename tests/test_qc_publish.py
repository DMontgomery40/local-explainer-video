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
