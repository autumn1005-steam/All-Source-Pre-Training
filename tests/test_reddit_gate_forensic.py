import json

from scripts.audit_reddit_gate import audit_gate_results, check_c095_consistency


def _write(path, *, digest, sources, gate, lcb, identity, route, params):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "target": "reddit",
                    "seed": 0,
                    "selected_sources": sources,
                    "config_digest": digest,
                    "safety_diagnostics": {
                        "rasp": {
                            "target_gate": gate,
                            "target_gate_lcb": lcb,
                            "router_off_reference_test_mrr": identity,
                            "raw_routed_test_mrr": route,
                        }
                    },
                    "training_diagnostics": {
                        "rasp": {"trainable_parameters": params}
                    },
                }
            ]
        ),
        encoding="utf-8",
    )


def test_forensic_audit_rejects_independently_retrained_sensitivity(tmp_path):
    main = tmp_path / "main.json"
    c95 = tmp_path / "c95.json"
    _write(
        main,
        digest="main",
        sources=["uci"],
        gate=0.0,
        lcb=-0.01,
        identity=0.42,
        route=0.421,
        params=269112,
    )
    _write(
        c95,
        digest="other",
        sources=["wikipedia"],
        gate=1.0,
        lcb=0.01,
        identity=0.40,
        route=0.39,
        params=76200,
    )

    report = audit_gate_results(main, {0.95: c95})
    comparison = report["comparisons"][0]
    assert comparison["matched_operating_point"] is False
    assert set(comparison["mismatch_reasons"]) >= {
        "config_digest",
        "selected_sources",
        "trainable_parameters",
        "identity_prediction",
        "route_prediction",
    }

    consistency = check_c095_consistency(report)
    assert consistency["status"] == "FAIL"
    assert consistency["reason"] == "c095_is_not_a_matched_operating_point"


def test_consistency_passes_only_for_same_fitted_model(tmp_path):
    main = tmp_path / "main.json"
    c95 = tmp_path / "c95.json"
    kwargs = dict(
        digest="same",
        sources=["uci"],
        gate=0.0,
        lcb=-0.01,
        identity=0.42,
        route=0.421,
        params=269112,
    )
    _write(main, **kwargs)
    _write(c95, **kwargs)
    report = audit_gate_results(main, {0.95: c95})
    consistency = check_c095_consistency(report)
    assert consistency["status"] == "PASS"
    assert consistency["checked_seeds"] == [0]
