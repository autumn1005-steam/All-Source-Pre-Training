import json

import pytest

from forward_pair.artifacts import validate_candidate_scores, write_selection_artifacts
from forward_pair.selector import CandidateScore, select_forward_pair


def _valid_scores() -> list[CandidateScore]:
    common = {
        "seed": 42,
        "target": "mooc",
        "config_digest": "abc123",
    }
    return [
        CandidateScore("a", ("a",), 0.8, 0.78, **common),
        CandidateScore("b", ("b",), 0.7, 0.68, **common),
        CandidateScore("pair_a_b", ("a", "b"), 0.81, 0.79, **common),
    ]


def test_stale_digest_is_rejected() -> None:
    scores = _valid_scores()
    scores[-1] = CandidateScore(
        "pair_a_b", ("a", "b"), 0.81, 0.79, 42, "mooc", "stale"
    )

    with pytest.raises(ValueError, match="config_digest"):
        validate_candidate_scores(scores)


def test_non_finite_score_is_rejected() -> None:
    scores = _valid_scores()
    scores[0] = CandidateScore("a", ("a",), float("nan"), 0.7, 42, "mooc", "abc123")

    with pytest.raises(ValueError, match="finite"):
        validate_candidate_scores(scores)


def test_written_selection_is_auditable(tmp_path) -> None:
    scores = _valid_scores()
    result = select_forward_pair(scores, ("a", "b"), margin=0.002)

    paths = write_selection_artifacts(
        tmp_path,
        scores,
        result,
        source_commits={"TempoFound": "tempo-sha", "Forward-Pair": "pair-sha"},
    )
    payload = json.loads(paths.selection_json.read_text(encoding="utf-8"))

    assert payload["information_boundary"] == "target_training_period_only"
    assert payload["candidate_count"] == 3
    assert payload["selected_sources"] == ["a", "b"]
    assert payload["source_commits"]["TempoFound"] == "tempo-sha"
    assert paths.candidate_scores_csv.is_file()
