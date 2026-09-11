from scripts.profile_expert_scaling import profile_expert_counts


def test_expert_scaling_parameters_increase_with_source_count():
    payload = profile_expert_counts(
        source_counts=(1, 2, 4),
        dim=8,
        batch_size=4,
        device="cpu",
        warmups=0,
        repetitions=2,
    )
    rows = payload["rows"]
    assert [row["sources"] for row in rows] == [1, 2, 4]
    assert rows[0]["total_module_parameters"] < rows[1]["total_module_parameters"]
    assert rows[1]["total_module_parameters"] < rows[2]["total_module_parameters"]
    assert all(row["profile_scope"] == "synthetic_expert_routing_only" for row in rows)
    assert all(row["performance_metric"] == "not_applicable" for row in rows)
