import pytest

from scripts.efficiency_timing import classify_pipeline_stage, summarize_durations


def test_summarize_durations_reports_query_latency_and_throughput():
    summary = summarize_durations([0.10, 0.12, 0.11], queries=100)
    assert summary["repetitions"] == 3
    assert summary["seconds_mean"] == pytest.approx(0.11)
    assert summary["seconds_std"] == pytest.approx(0.01)
    assert summary["milliseconds_per_query_mean"] == pytest.approx(1.1)
    assert summary["queries_per_second"] == pytest.approx(100 / 0.11)


def test_summarize_durations_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        summarize_durations([], queries=10)
    with pytest.raises(ValueError):
        summarize_durations([0.1], queries=0)


def test_pipeline_stage_classification_excludes_control_branch():
    assert classify_pipeline_stage("run/candidates/uci/pretrained.pth") == "source_selection"
    assert classify_pipeline_stage("run/final/rasp/pretrained.pth") == "source_pretraining"
    assert classify_pipeline_stage("run/final/forward_pair/pretrained.pth") == "source_pretraining"
    assert classify_pipeline_stage("run/final/all_sources/pretrained.pth") == "control_branch"
