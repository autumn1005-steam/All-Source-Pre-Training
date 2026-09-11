import json
from pathlib import Path

import pytest

from scripts.audit_socialevo_5seed import (
    collect_dense_gate_report,
    collect_socialevo_records,
    enrich_with_remote_manifest,
    summarize,
    summarize_gate_panels,
)


SEEDS = (0, 1, 2, 3, 42)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(raw_root: Path) -> None:
    branch = raw_root / "hrcsr-lastfm-socialevo-5seed-v1"
    for index, seed in enumerate(SEEDS):
        strict = 0.70 + index / 100
        identity = 0.71 + index / 100
        route = identity + (0.02 if seed != 42 else 0.0)
        gate = 0.0 if seed == 42 else 1.0
        hard = route if gate else identity
        common = {
            "target": "socialevo",
            "seed": seed,
            "evaluation_protocol": "ranking",
            "negative_policy": "historical",
            "negatives_per_positive": 20,
            "checkpoint_selection_scope": "target_validation_only_after_source_selection",
            "target_optimization_events": 100,
            "target_calibration_events_reserved": 25,
            "training_negative_policy": "historical_with_collision_free_random_cold_start_fallback",
            "training_negative_fallback_rate": 0.001,
        }
        _write(
            branch / "calib_suite" / "socialevo" / f"seed{seed}" / "transfer" / "rasp.json",
            {
                **common,
                "config_digest": "suite-digest",
                "sources": ["mooc"],
                "target_gate": gate,
                "target_gate_lcb": 0.01 if gate else 0.0,
                "target_gate_calibration_events": 23,
                "raw_routed_test_mrr": route,
                "router_off_reference_test_mrr": identity,
                "test_mrr": hard,
                "safety_gate_sweep": {
                    "identity": {"gate": 0.0, "test_mrr": identity},
                    "route": {"gate": 1.0, "test_mrr": route},
                    "hard": {"gate": gate, "test_mrr": hard},
                    "soft": {"gate": gate / 2, "test_mrr": hard - 0.001},
                },
            },
        )
        _write(
            branch / "calib_suite" / "socialevo" / f"seed{seed}" / "transfer" / "all_sources.json",
            {**common, "config_digest": "suite-digest", "test_mrr": 0.65 + index / 100},
        )
        _write(
            branch / "calib_strict_ranked" / "socialevo" / f"seed{seed}" / "transfer" / "rasp.json",
            {**common, "config_digest": "strict-digest", "test_mrr": strict},
        )
        _write(
            raw_root / "socialevo-target-only-tgat" / "socialevo" / f"seed{seed}" / "summary.json",
            {**common, "mode": "scratch", "target_calibration_events_reserved": 0, "test_mrr": 0.5 + index / 100},
        )
        _write(
            raw_root / "socialevo-target-only-graphmixer" / "socialevo" / f"seed{seed}" / "summary.json",
            {**common, "mode": "scratch", "target_calibration_events_reserved": 0, "test_mrr": 0.6 + index / 100},
        )
    _write(
        branch / "calib_suite" / "socialevo" / "paired_results.json",
        [{"seed": 3, "rasp_test_mrr": -1.0}, {"seed": 42, "rasp_test_mrr": -1.0}],
    )


def test_direct_seed_artifacts_override_stale_two_row_aggregate(tmp_path: Path) -> None:
    _fixture(tmp_path)

    records = collect_socialevo_records(tmp_path)

    assert [row["Seed"] for row in records] == [0, 1, 2, 3, 42]
    assert records[0]["HRCSRHardMRR"] == pytest.approx(0.73)
    assert records[-1]["HRCSRHardMRR"] == pytest.approx(0.75)
    assert all(row["AggregateSource"] == "direct_per_seed" for row in records)


def test_summary_reports_in_scope_coverage_and_pairing(tmp_path: Path) -> None:
    _fixture(tmp_path)

    summary = summarize(collect_socialevo_records(tmp_path))

    assert summary["coverage"]["HRCSR-Hard"]["seeds"] == [0, 1, 2, 3, 42]
    assert set(summary["coverage"]) == {
        "HRCSR-Hard",
        "Strict RASP",
        "All-source",
        "Target-only T-GAT",
        "Target-only GraphMixer",
    }
    assert summary["hrcsr_minus_strict"]["positive_seeds"] == 5
    assert summary["hrcsr_minus_all_source"]["positive_seeds"] == 5
    assert summary["hrcsr_minus_all_source"]["mean_delta"] == pytest.approx(0.076)
    assert summary["gate_diagnostics"]["gate_open_count"] == 4
    assert summary["gate_diagnostics"]["safety_violations"] == 0
    assert summary["gate_diagnostics"]["missed_opportunities"] == 0


def test_remote_manifest_enrichment_keeps_reconstructed_counters_explicit(tmp_path: Path) -> None:
    _fixture(tmp_path)
    records = collect_socialevo_records(tmp_path)
    suite_remote = records[0]["HRCSRRemoteFile"]
    checkpoint_remote = suite_remote.replace("transfer/rasp.json", "final/rasp/pretrained.pth")
    manifest = {
        "dataset_preprocessing_composite_sha256": "composite-dataset-hash",
        "query_population_reconstruction": {
            "test": {
                "requested": 90,
                "retained": 87,
                "skipped": 3,
                "skip_rate": 3 / 90,
                "fallback_count": 0,
                "collision_rejection_count": 123,
            }
        },
        "artifacts": [
            {"path": suite_remote, "exists": True, "mtime_utc": "2026-09-11T00:00:00+00:00", "sha256": "result-hash"},
            {"path": checkpoint_remote, "exists": True, "mtime_utc": "2026-09-11T00:00:01+00:00", "sha256": "checkpoint-hash"},
            {"path": "/root/autodl-tmp/processed-westb-verified/processed/ml_socialevo.csv", "exists": True, "sha256": "dataset-hash"},
        ],
    }

    enriched = enrich_with_remote_manifest(records, manifest)

    assert enriched[0]["HRCSRResultHash"] == "result-hash"
    assert enriched[0]["HRCSRCheckpointHash"] == "checkpoint-hash"
    assert enriched[0]["DatasetPreprocessingHash"] == "composite-dataset-hash"
    assert enriched[0]["TestRequested"] == 90
    assert enriched[0]["TestFallbackCount"] == 0
    assert enriched[0]["TestCounterSource"] == "RECONSTRUCTED_FROM_DATASET_AND_FINAL_SAMPLER"


def test_gate_panel_summary_keeps_dense_and_sparse_denominators_separate(tmp_path: Path) -> None:
    _fixture(tmp_path)
    sparse = summarize(collect_socialevo_records(tmp_path))
    dense = {
        "targets": [
            {
                "target": "alpha",
                "rows": [
                    {"gate_open": True, "decision_correct": True, "safety_violation": False,
                     "missed_opportunity": False, "selector_regret": 0.0},
                    {"gate_open": False, "decision_correct": False, "safety_violation": False,
                     "missed_opportunity": True, "selector_regret": 0.01},
                ],
            }
        ]
    }

    panels = summarize_gate_panels(dense, sparse)

    assert panels["dense"]["n_decisions"] == 2
    assert panels["dense"]["gate_open_count"] == 1
    assert panels["dense"]["missed_opportunities"] == 1
    assert panels["sparse"]["n_decisions"] == 5
    assert panels["sparse"]["gate_open_count"] == 4
    assert panels["all_audited"]["n_decisions"] == 7
    assert panels["all_audited"]["gate_open_count"] == 5


def test_dense_gate_report_is_rebuilt_from_all_five_direct_target_branches(tmp_path: Path) -> None:
    roots = {
        "wikipedia": "hrcsr-ablation-matched-v1",
        "reddit": "hrcsr-ablation-matched-v1",
        "mooc": "hrcsr-ablation-matched-v1",
        "uci": "hrcsr-crossdomain-5seed-v1",
        "enron": "hrcsr-crossdomain-5seed-v1",
    }
    for target, run in roots.items():
        for seed in SEEDS:
            identity = 0.5
            route = 0.51 if seed != 42 else 0.49
            gate = 1.0 if seed != 42 else 0.0
            _write(
                tmp_path / run / "calib_suite" / target / f"seed{seed}" / "transfer" / "rasp.json",
                {
                    "target": target,
                    "seed": seed,
                    "target_gate": gate,
                    "target_gate_lcb": 0.01 if gate else -0.01,
                    "safety_gate_sweep": {
                        "identity": {"test_mrr": identity},
                        "route": {"test_mrr": route},
                        "hard": {"test_mrr": route if gate else identity},
                    },
                },
            )

    report = collect_dense_gate_report(tmp_path)

    assert [target["target"] for target in report["targets"]] == list(roots)
    assert sum(len(target["rows"]) for target in report["targets"]) == 25
