"""Summarize preserved JSON evidence without graph inputs, PyTorch, or GPU work."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n_encoder_level_records": len(values)}


def summarize(records):
    output = {"keys": [r["key"] for r in records], "classification": {}, "clustering": {}}
    budgets = sorted({str(k) for r in records for k in r.get("classification", {})}, key=int)
    for budget in budgets:
        available = [r["classification"][budget]["mean"] for r in records if budget in r.get("classification", {})]
        output["classification"][budget] = {metric: mean_sd([r[metric] for r in available])
                                                    for metric in available[0]}
    available = [r["clustering"]["mean"] for r in records if "clustering" in r]
    if available:
        output["clustering"] = {metric: mean_sd([r[metric] for r in available]) for metric in available[0]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "reproduction")
    parser.add_argument("--output", type=Path, default=ROOT / "summaries/study_summary.json")
    args = parser.parse_args()
    root = args.root
    plan = read(root / "study_plan.json")
    assert plan["status"] == "complete" and plan["attention_control_status"] == "complete"
    selected = set(plan["tuning"])
    groups = {}
    for dataset, methods in plan["primary"].items():
        for method, spec in methods.items():
            selected.update(spec["keys"])
            groups[f"primary/{dataset}/{method}"] = spec["keys"]
    for family in ("ablation", "sensitivity"):
        for spec in plan[family]:
            selected.update(spec["keys"])
            label = spec.get("variant", "lambda=" + str(spec.get("lam")))
            groups[f"{family}/{spec['dataset']}/{label}"] = spec["keys"]
    assert len(selected) == 153
    summary = {"units": "Fractions, not percentages; sample SD is across encoder-level means, never pooled probes.",
               "counts": {"base_trained_encoder_records_including_tuning": len(selected)},
               "base": {name: summarize([read(root / "results" / (key + ".json")) for key in keys])
                        for name, keys in groups.items()}}
    extension_specs = [
        ("positive_extension", 36, lambda r: f"{r['args']['dataset']}/{r['args']['positive'].removeprefix('generated_')}", "results"),
        ("transfer_results", 48, lambda r: f"{r['dataset']}/{r['variant']}/{r['mode']}", ""),
        ("adaptation_results", 18, lambda r: f"{r['dataset']}/{r['variant']}/{r['initial_graph']}", ""),
        ("budget_control_results", 3, lambda r: f"{r['dataset']}/{r['variant']}/fixed600", ""),
    ]
    for directory, expected, group_name, nested in extension_specs:
        folder = root / directory
        subplan = read(folder / "study_plan.json")
        assert subplan["status"] == "complete" and len(subplan["completed_keys"]) == expected
        by_group = defaultdict(list)
        for key in subplan["completed_keys"]:
            record = read(folder / nested / (key + ".json"))
            by_group[group_name(record)].append(record)
        summary[directory] = {name: summarize(records) for name, records in sorted(by_group.items())}
        summary["counts"][directory + "_records"] = expected
    summary["counts"].update(transfer_trained_encoder_records=36, transfer_untrained_encoder_groups=6,
                             transfer_raw_feature_probe_groups=6, single_stage_trained_encoder_records=225,
                             final_trained_encoder_trial_records=246, adaptation_ssl_phases=36,
                             fixed600_new_ssl_phases=3, total_recorded_ssl_training_phases=264)
    summary["limitations"] = [
        "Counts include tuning runs and nested follow-ups; they are not independent graph samples.",
        "One fixed graph partition per dataset; no zero-shot or cross-dataset transfer claim.",
        "Two-stage initialization comparisons can differ in actual epochs and graph costs.",
        "Fixed600 replay matches target updates, not equal total two-stage compute.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
