from scripts.aggregate_component_ablation import aggregate


def test_component_ablation_aggregation_reports_delta_and_wins():
    rows = []
    for variant, value in (("a0_strict_rasp", 0.4), ("a1_expert_identity", 0.42)):
        for seed in (0, 1):
            rows.append({"target": "mooc", "seed": seed, "variant": variant, "mrr": value + seed * 0.01, "delta_vs_a0": (0.02 if variant != "a0_strict_rasp" else 0.0)})
    payload = aggregate(rows)
    summary = {row["variant"]: row for row in payload["summaries"]}
    assert summary["a1_expert_identity"]["delta_vs_a0_mean"] == 0.02
    assert summary["a1_expert_identity"]["positive_seeds_vs_a0"] == 2
