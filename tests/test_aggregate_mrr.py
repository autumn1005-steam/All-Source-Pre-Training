from scripts.aggregate_mrr import extract_method_values


def test_extracts_strict_and_hard_from_paired_row() -> None:
    row = {
        "target": "mooc",
        "seed": 0,
        "rasp_test_mrr": 0.4,
        "all_sources_test_mrr": 0.3,
        "safety_diagnostics": {
            "rasp": {"safety_gate_sweep": {"hard": {"test_mrr": 0.45}}}
        },
    }
    values = extract_method_values(row)
    assert values["strict_rasp"] == 0.4
    assert values["all_source"] == 0.3
    assert values["hrcsr_hard"] == 0.45
