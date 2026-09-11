import json

from scripts.aggregate_checkpoint_boundary import aggregate


def _write(root, mode, target, seed, mrr, gain, gate):
    path = root / mode / target / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "target": target,
                "seed": seed,
                "mode": "frozen",
                "test_mrr": mrr,
                "raw_route_minus_reference_test_mrr": gain,
                "target_gate": gate,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_preserves_audit_mode_when_payload_has_runtime_mode(tmp_path):
    _write(tmp_path, "fixed_epoch", "reddit", 0, 0.4, -0.01, 1.0)
    _write(tmp_path, "fixed_epoch", "reddit", 1, 0.5, 0.01, 0.0)
    rows = aggregate(tmp_path)
    row = next(item for item in rows if item["target"] == "reddit")
    assert row["mode"] == "fixed_epoch"
    assert row["n_seeds"] == 2
    assert row["safety_violation_count"] == 1
    assert row["missed_opportunity_count"] == 1
