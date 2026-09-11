import json

import pytest

from scripts.audit_unified_table import (
    check_protocol_consistency,
    exact_sign_flip_p,
    paired_bootstrap_ci,
    read_transfer_record,
)


def test_exact_sign_flip_p_is_exact_for_five_positive_differences():
    # Only the all-negative sign assignment is at least as extreme as the
    # observed positive mean in the opposite direction.
    assert exact_sign_flip_p([1.0, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(2 / 32)


def test_paired_bootstrap_ci_is_deterministic_and_contains_constant_delta():
    low, high = paired_bootstrap_ci([0.25] * 5, replicates=2000, seed=17)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)


def test_transfer_record_uses_explicit_protocol_metadata(tmp_path):
    path = tmp_path / "transfer.json"
    path.write_text(
        json.dumps(
            {
                "target": "reddit",
                "seed": 42,
                "test_mrr": 0.43,
                "evaluation_protocol": "ranking",
                "negative_policy": "historical",
                "negatives_per_positive": 20,
                "checkpoint_selection_scope": "prefix_only",
                "config_digest": "abc",
            }
        ),
        encoding="utf-8",
    )
    record = read_transfer_record(path, method="Strict RASP")
    assert record["Metric"] == "MRR"
    assert record["NegativeSampler"] == "historical;20_per_positive"
    assert record["Checkpoint"] == "prefix_only"
    assert record["CodeVersion"] == "config_digest:abc"
    assert record["MRR"] == pytest.approx(0.43)


def test_transfer_record_marks_unrecorded_fields_instead_of_inferring(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps({"target": "mooc", "seed": 0, "test_mrr": 0.4}),
        encoding="utf-8",
    )
    record = read_transfer_record(path, method="Target-only T-GAT")
    assert record["SplitID"] == "UNRECORDED"
    assert record["NegativeSampler"] == "UNRECORDED"
    assert record["Checkpoint"] == "UNRECORDED"
    assert record["CodeVersion"] == "UNRECORDED"


def test_protocol_consistency_requires_complete_seed_grid_and_matched_protocol():
    rows = []
    for method in ("A", "B"):
        for seed in (0, 1):
            rows.append(
                {
                    "Target": "reddit",
                    "Method": method,
                    "Seed": seed,
                    "Metric": "MRR",
                    "SplitID": "split-x",
                    "NegativeSampler": "historical;20_per_positive",
                    "Checkpoint": "validation",
                }
            )
    report = check_protocol_consistency(
        rows, targets=("reddit",), methods=("A", "B"), seeds=(0, 1)
    )
    assert report["status"] == "PASS"

    rows[-1]["NegativeSampler"] = "random;20_per_positive"
    report = check_protocol_consistency(
        rows, targets=("reddit",), methods=("A", "B"), seeds=(0, 1)
    )
    assert report["status"] == "FAIL"
    assert any("NegativeSampler" in error for error in report["errors"])
