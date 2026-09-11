import json

from forward_pair.pipeline import (
    build_forward_pair_plan,
    config_digest,
    result_artifact_matches,
    score_artifact_matches,
)
from tempofound.config import ExperimentConfig


def _k4_config() -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "name": "k4",
            "datasets": ["a", "b", "c", "d", "target"],
            "targets": ["target"],
            "seeds": [42],
        }
    )


def test_k4_plan_has_seven_candidates_before_final_retraining() -> None:
    plan = build_forward_pair_plan(_k4_config(), target="target", seed=42)

    assert [job.kind for job in plan.singleton_jobs] == ["singleton"] * 4
    assert [job.sources for job in plan.singleton_jobs] == [
        ("a",),
        ("b",),
        ("c",),
        ("d",),
    ]
    assert plan.dynamic_anchor_pair_count == 3
    assert plan.expected_candidate_count == 7
    assert plan.final_stages == ("selected_pretrain", "all_source_pretrain", "transfer")


def test_default_plan_never_schedules_obsolete_selectors() -> None:
    plan = build_forward_pair_plan(_k4_config(), target="target", seed=42)

    names = [job.name for job in plan.singleton_jobs] + list(plan.final_stages)
    assert not any("proposed" in name or "leave_one" in name for name in names)


def test_configuration_digest_is_stable_and_sensitive() -> None:
    first = _k4_config()
    same = _k4_config()
    changed = ExperimentConfig.from_mapping(
        {
            "name": "k4",
            "datasets": ["a", "b", "c", "d", "target"],
            "targets": ["target"],
            "seeds": [42],
            "proxy": {"margin": 0.01},
        }
    )

    assert config_digest(first) == config_digest(same)
    assert config_digest(first) != config_digest(changed)


def test_stale_score_artifact_is_not_resumed(tmp_path) -> None:
    artifact = tmp_path / "score.json"
    artifact.write_text(
        json.dumps(
            {
                "target": "target",
                "seed": 42,
                "sources": ["a"],
                "config_digest": "old",
                "proxy_ap": 0.8,
                "proxy_auc": 0.7,
            }
        ),
        encoding="utf-8",
    )

    assert not score_artifact_matches(
        artifact,
        target="target",
        seed=42,
        sources=("a",),
        digest="new",
    )


def test_non_finite_score_artifact_is_not_resumed(tmp_path) -> None:
    artifact = tmp_path / "score.json"
    artifact.write_text(
        json.dumps(
            {
                "target": "target",
                "seed": 42,
                "sources": ["a"],
                "config_digest": "digest",
                "proxy_ap": float("nan"),
                "proxy_auc": 0.7,
            }
        ),
        encoding="utf-8",
    )

    assert not score_artifact_matches(
        artifact,
        target="target",
        seed=42,
        sources=("a",),
        digest="digest",
    )


def test_valid_transfer_result_is_resumable(tmp_path) -> None:
    artifact = tmp_path / "transfer.json"
    artifact.write_text(
        json.dumps(
            {
                "target": "target",
                "seed": 42,
                "sources": ["a", "b"],
                "config_digest": "digest",
                "mode": "frozen",
                "test_ap": 0.8,
                "test_auc": 0.7,
            }
        ),
        encoding="utf-8",
    )

    assert result_artifact_matches(
        artifact,
        target="target",
        seed=42,
        sources=("a", "b"),
        digest="digest",
        mode="frozen",
    )


def test_transfer_result_with_wrong_sources_is_not_resumable(tmp_path) -> None:
    artifact = tmp_path / "transfer.json"
    artifact.write_text(
        json.dumps(
            {
                "target": "target",
                "seed": 42,
                "sources": ["a"],
                "config_digest": "digest",
                "mode": "frozen",
                "test_ap": 0.8,
                "test_auc": 0.7,
            }
        ),
        encoding="utf-8",
    )

    assert not result_artifact_matches(
        artifact,
        target="target",
        seed=42,
        sources=("a", "b"),
        digest="digest",
        mode="frozen",
    )
