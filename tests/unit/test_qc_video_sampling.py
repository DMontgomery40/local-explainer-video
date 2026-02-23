from core.qc_video import video_qc_sample_times


def test_video_qc_sample_times_unknown_duration() -> None:
    assert video_qc_sample_times(None) == [("start", 0.0)]


def test_video_qc_sample_times_short_duration() -> None:
    samples = video_qc_sample_times(0.2)
    assert samples[0] == ("start", 0.0)
    assert samples[1][0] == "mid"
    assert samples[1][1] > 0.0


def test_video_qc_sample_times_normal_duration() -> None:
    samples = video_qc_sample_times(10.0)
    assert samples == [("start", 0.0), ("mid", 5.0), ("end", 9.9)]
