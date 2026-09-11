import json

import pytest

from scripts.aggregate_checkpoint_boundary import aggregate_roots


def _write_audit(path, *, seed, mode, identity, route, hard, gate):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "target": "reddit",
                "seed": seed,
                "checkpoint_selection_mode": mode,
                "router_off_reference_test_mrr": identity,
                "raw_routed_test_mrr": route,
                "test_mrr": hard,
                "target_gate": gate,
                "target_gate_lcb": -0.01,
            }
        ),
        encoding="utf-8",
    )


def _write_strict(path, *, seed, mrr):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"target": "reddit", "seed": seed, "test_mrr": mrr}),
        encoding="utf-8",
    )


def test_boundary_aggregation_reports_seed_metrics_and_selector_errors(tmp_path):
    audit = tmp_path / "audit"
    strict = tmp_path / "strict"
    _write_audit(
        audit / "validation" / "reddit" / "seed0.json",
        seed=0,
        mode="validation",
        identity=0.40,
        route=0.39,
        hard=0.39,
        gate=1.0,
    )
    _write_audit(
        audit / "validation" / "reddit" / "seed1.json",
        seed=1,
        mode="validation",
        identity=0.40,
        route=0.42,
        hard=0.40,
        gate=0.0,
    )
    _write_strict(strict / "reddit" / "seed0" / "transfer" / "rasp.json", seed=0, mrr=0.38)
    _write_strict(strict / "reddit" / "seed1" / "transfer" / "rasp.json", seed=1, mrr=0.39)

    payload = aggregate_roots([audit], [strict])
    rows = payload["seed_rows"]
    assert len(rows) == 2
    assert rows[0]["safety_violation"] is True
    assert rows[1]["missed_opportunity"] is True
    assert rows[0]["hard_minus_strict"] == pytest.approx(0.01)
    assert rows[1]["selector_regret"] == pytest.approx(0.02)
    summary = payload["summaries"][0]
    assert summary["gate_opens"] == 1
    assert summary["safety_violations"] == 1
    assert summary["missed_opportunities"] == 1
