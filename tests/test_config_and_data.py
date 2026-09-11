from pathlib import Path

import numpy as np
import pytest

from tempofound.config import ExperimentConfig
from tempofound.data import split_target_train_period


def test_target_is_rejected_from_explicit_source_pool() -> None:
    with pytest.raises(ValueError, match="target cannot appear in source pool"):
        ExperimentConfig.from_mapping(
            {
                "name": "invalid",
                "datasets": ["mooc", "reddit"],
                "targets": ["mooc"],
                "source_overrides": {"mooc": ["mooc", "reddit"]},
            }
        )


def test_training_period_split_is_stable_and_chronological() -> None:
    data = {
        "train_ts": np.array([4.0, 1.0, 3.0, 2.0]),
        "train_src": np.array([40, 10, 30, 20]),
        "train_dst": np.array([41, 11, 31, 21]),
    }

    calibration, probe = split_target_train_period(data, 0.5)

    assert calibration["train_ts"].tolist() == [1.0, 2.0]
    assert calibration["train_src"].tolist() == [10, 20]
    assert probe["train_ts"].tolist() == [3.0, 4.0]
    assert calibration["train_ts"].max() <= probe["train_ts"].min()


def test_primary_and_scalability_presets_share_one_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    k4 = ExperimentConfig.from_yaml(root / "configs/k4_primary.yaml")
    k6 = ExperimentConfig.from_yaml(root / "configs/k6_scalability.yaml")

    assert k4.time_encoding_mode == "paper_sincos"
    assert k4.sources_for("mooc") == ("wikipedia", "reddit", "uci", "enron")
    assert len(k4.sources_for("wikipedia")) == 4
    assert k6.sources_for("mooc") == (
        "wikipedia",
        "reddit",
        "uci",
        "enron",
        "lastfm",
        "socialevo",
    )
    assert k4.total_batches_per_epoch == k6.total_batches_per_epoch == 160


def test_split_rejects_empty_partition() -> None:
    one_event = {
        "train_ts": np.array([1.0]),
        "train_src": np.array([1]),
        "train_dst": np.array([2]),
    }
    with pytest.raises(ValueError, match="too small"):
        split_target_train_period(one_event, 0.8)


def test_graphmixer_control_is_a_configuration_not_a_duplicate_project() -> None:
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_yaml(root / "configs/graphmixer_control.yaml")

    assert config.backbone == "graphmixer"
    assert len(config.sources_for("mooc")) == 4


def test_checkpoint_selection_mode_is_validated() -> None:
    raw = {
        "name": "checkpoint-audit",
        "datasets": ["a", "b", "c"],
        "targets": ["a"],
        "source_overrides": {"a": ["b", "c"]},
        "training": {"checkpoint_selection_mode": "fixed_epoch"},
    }
    config = ExperimentConfig.from_mapping(raw)
    assert config.checkpoint_selection_mode == "fixed_epoch"

    invalid = dict(raw)
    invalid["training"] = {"checkpoint_selection_mode": "test_selected"}
    with pytest.raises(ValueError, match="checkpoint_selection_mode"):
        ExperimentConfig.from_mapping(invalid)


def test_rasp_configuration_loads_selection_and_evaluation_sections() -> None:
    config = ExperimentConfig.from_mapping(
        {
            "name": "rasp",
            "datasets": ["a", "b", "target"],
            "targets": ["target"],
            "selection": {
                "selection_method": "rasp",
                "proxy_windows": 2,
                "bootstrap_replicates": 32,
                "beam_width": 2,
                "max_sources": 2,
                "candidate_budget": 4,
            },
            "training": {
                "pretraining_strategy": "conflict_aware",
                "affinity_temperature": 0.2,
                "minimum_source_batches": 1,
                "domain_adversarial_scale": 0.5,
                "domain_loss_coefficient": 0.1,
            },
            "evaluation": {
                "evaluation_protocol": "ranking",
                "negative_policy": "historical",
                "negatives_per_positive": 20,
                "evaluation_batch_size": 8,
            },
        }
    )

    assert config.selection_method == "rasp"
    assert config.proxy_windows == 2
    assert config.bootstrap_replicates == 32
    assert config.candidate_budget == 4
    assert config.evaluation_protocol == "ranking"
    assert config.negatives_per_positive == 20
    assert config.pretraining_strategy == "conflict_aware"
    assert config.affinity_temperature == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"selection_method": "unknown"}, "selection_method"),
        ({"selection_method": "rasp", "proxy_windows": 1}, "proxy_windows"),
        (
            {"selection_method": "rasp", "bootstrap_replicates": 0},
            "bootstrap_replicates",
        ),
        ({"selection_method": "rasp", "beam_width": 0}, "beam_width"),
        ({"selection_method": "rasp", "max_sources": 3}, "max_sources"),
        (
            {
                "selection_method": "rasp",
                "max_sources": 2,
                "candidate_budget": 1,
            },
            "candidate_budget",
        ),
    ],
)
def test_rasp_configuration_rejects_invalid_search_limits(
    selection: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_mapping(
            {
                "name": "invalid",
                "datasets": ["a", "b", "target"],
                "targets": ["target"],
                "selection": selection,
            }
        )
