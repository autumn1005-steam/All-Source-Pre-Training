import pytest

from scripts.recompute_matched_gate_sensitivity import sweep_transfer_payload


def test_sweep_reuses_predictions_and_reproduces_authoritative_c095():
    payload = {
        "target": "reddit",
        "seed": 0,
        "target_gate_window_means": [-0.02, -0.01, 0.0, 0.01, 0.02],
        "target_gate_lcb": -0.012,
        "target_gate": 0.0,
        "raw_routed_test_mrr": 0.41,
        "router_off_reference_test_mrr": 0.42,
        "config_digest": "abc",
    }
    rows = sweep_transfer_payload(payload, seed=0, confidences=(0.90, 0.95, 0.99), replicates=2000)
    assert [row["confidence"] for row in rows] == [0.90, 0.95, 0.99]
    assert all(row["identity_mrr"] == 0.42 for row in rows)
    assert all(row["route_mrr"] == 0.41 for row in rows)
    assert all(row["future_route_minus_identity"] == pytest.approx(-0.01) for row in rows)
    assert all(row["safety_violation"] is False for row in rows)
    assert all(row["missed_opportunity"] is False for row in rows)
    assert rows[1]["authoritative_c095_gate"] == 0.0


def test_sweep_rejects_missing_window_means():
    with pytest.raises(ValueError, match="window means"):
        sweep_transfer_payload(
            {"target": "reddit", "seed": 0},
            seed=0,
            confidences=(0.95,),
            replicates=2000,
        )
