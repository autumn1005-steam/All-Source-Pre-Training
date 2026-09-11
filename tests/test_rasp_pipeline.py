import json

import numpy as np
import pandas as pd
import pytest

from forward_pair.pipeline import (
    build_rasp_plan,
    rasp_score_artifact_matches,
    run_pipeline,
)
from forward_pair.proxy import build_proxy_windows, summarize_proxy_windows
from tempofound.config import ExperimentConfig


def _config() -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "name": "rasp",
            "datasets": ["a", "b", "c", "target"],
            "targets": ["target"],
            "seeds": [42],
            "selection": {
                "selection_method": "rasp",
                "proxy_windows": 3,
                "bootstrap_replicates": 64,
                "beam_width": 2,
                "max_sources": 2,
                "candidate_budget": 6,
            },
        }
    )


def _write_dataset(root, name: str, phase: int) -> None:
    order = np.arange(24)
    pandas_frame = pd.DataFrame(
        {
            "u": ((order + phase) % 5) + 1,
            "i": ((order + phase + 1) % 5) + 1,
            "idx": order + 1,
            "ts": order.astype(float) + 1.0,
            "label": np.zeros(len(order), dtype=int),
        }
    )
    pandas_frame.to_csv(root / f"ml_{name}.csv", index=False)
    np.save(
        root / f"ml_{name}_node.npy",
        np.arange(18, dtype=np.float32).reshape(6, 3) / (20.0 + phase),
    )
    np.save(
        root / f"ml_{name}.npy",
        np.arange(50, dtype=np.float32).reshape(25, 2) / (30.0 + phase),
    )


def test_proxy_windows_are_non_overlapping_and_chronological() -> None:
    data = {
        "train_ts": np.arange(1.0, 11.0),
        "train_src": np.arange(1, 11),
        "train_dst": np.arange(11, 21),
        "neighbor_finder": object(),
        "node_features": np.zeros((2, 2)),
        "edge_features": np.zeros((2, 2)),
    }

    calibration, windows = build_proxy_windows(
        data, calibration_fraction=0.4, window_count=3
    )

    assert calibration["train_ts"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert [window["train_ts"].tolist() for window in windows] == [
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ]
    assert calibration["train_ts"].max() < windows[0]["train_ts"].min()


def test_proxy_window_summary_is_deterministic_and_risk_adjusted() -> None:
    first = summarize_proxy_windows(
        [0.7, 0.8, 0.9],
        seed=5,
        bootstrap_replicates=64,
        confidence=0.9,
        risk_beta=1.0,
        cardinality=2,
        size_penalty=0.01,
    )
    second = summarize_proxy_windows(
        [0.7, 0.8, 0.9],
        seed=5,
        bootstrap_replicates=64,
        confidence=0.9,
        risk_beta=1.0,
        cardinality=2,
        size_penalty=0.01,
    )

    assert first == second
    assert first["mean"] == pytest.approx(0.8)
    assert first["risk_adjusted_utility"] < first["mean"]


def test_rasp_plan_starts_with_every_singleton_under_global_budget() -> None:
    plan = build_rasp_plan(_config(), target="target", seed=42)

    assert plan.initial_candidates == (("a",), ("b",), ("c",))
    assert plan.beam_width == 2
    assert plan.max_sources == 2
    assert plan.candidate_budget == 6


def test_rasp_score_resume_requires_window_and_risk_fields(tmp_path) -> None:
    artifact = tmp_path / "score.json"
    payload = {
        "target": "target",
        "seed": 42,
        "sources": ["a", "b"],
        "config_digest": "digest",
        "window_proxy_ap": [0.7, 0.8, 0.9],
        "risk_adjusted_utility": 0.75,
        "bootstrap": {
            "mean": 0.8,
            "standard_error": 0.04,
            "lower": 0.7,
            "upper": 0.9,
            "replicates": [0.7, 0.8, 0.9],
        },
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    assert rasp_score_artifact_matches(
        artifact,
        target="target",
        seed=42,
        sources=("a", "b"),
        digest="digest",
        proxy_windows=3,
    )
    payload["window_proxy_ap"] = [0.7, 0.8]
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert not rasp_score_artifact_matches(
        artifact,
        target="target",
        seed=42,
        sources=("a", "b"),
        digest="digest",
        proxy_windows=3,
    )


@pytest.mark.parametrize('paired', [False, True])
def test_rasp_pipeline_runs_end_to_end_with_training_only_selection(tmp_path, paired) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    for phase, name in enumerate(("a", "b", "target")):
        _write_dataset(processed, name, phase)
    config = ExperimentConfig.from_mapping(
        {
            "name": "rasp_tiny",
            "datasets": ["a", "b", "target"],
            "targets": ["target"],
            "seeds": [42],
            "model": {
                "shared_dim": 8,
                "time_dim": 4,
                "num_layers": 1,
                "num_heads": 2,
                "num_neighbors": 2,
                "dropout": 0.0,
                "temporal_experts": 4 if paired else 0,
                "expert_identity_init": paired,
            },
            "training": {
                "pretrain_epochs": 1,
                "transfer_epochs": 1,
                "batch_size": 8,
                "total_batches_per_epoch": 2,
            },
            "proxy": {
                "calibration_fraction": 0.7,
                "calibration_steps": 1,
                "proxy_batch_size": 4,
                "proxy_batches": 1,
                "proxy_metric": 'mrr' if paired else 'ap',
            },
            "selection": {
                "selection_method": "rasp",
                "proxy_windows": 2,
                "bootstrap_replicates": 8,
                "bootstrap_confidence": 0.8,
                "beam_width": 2 if paired else 1,
                "max_sources": 2 if paired else 1,
                "candidate_budget": 3 if paired else 2,
                "consistent_candidate_weights": paired,
                "temporal_paired_gate": paired,
            },
            "evaluation": {
                "evaluation_protocol": "ranking",
                "negative_policy": "random" if paired else "historical",
                "negatives_per_positive": 3,
                "evaluation_batch_size": 2,
            },
        }
    )

    summary = run_pipeline(
        config,
        target="target",
        seed=42,
        processed_dir=processed,
        output_root=tmp_path / "output",
        device="cpu",
    )

    assert summary["method"] == "rasp"
    assert summary["candidate_count"] == (3 if paired else 2)
    assert 1 <= len(summary["selected_sources"]) <= (2 if paired else 1)
    assert 0.0 <= summary["rasp_test_mrr"] <= 1.0
    selection = json.loads(
        (tmp_path / "output/target/seed42/selection/selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["information_boundary"] == "target_training_period_only"
    if paired:
        import torch
        run = tmp_path / 'output/target/seed42'
        candidate = torch.load(run / 'candidates/portfolio_a_b/pretrained.pth', weights_only=False)
        final = torch.load(run / 'final/all_sources/pretrained.pth', weights_only=False)
        assert final['model_config']['temporal_experts'] == 4
        assert final['metadata']['pretraining']['temporal_expert_parameters'] > 0
        assert candidate['metadata']['pretraining']['source_weights'] == final['metadata']['pretraining']['source_weights']
        assert selection['temporal_paired_gate']
        if len(selection['selected_sources']) == 2:
            assert summary['test_mrr_delta'] == 0.0
