"""Audit and aggregate direct five-seed SocialEvo result artifacts.

The stale ``paired_results.json`` files are intentionally ignored.  Every
reported observation is loaded from a seed-specific raw JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from scripts.audit_unified_table import exact_sign_flip_p, paired_bootstrap_ci


SEEDS = (0, 1, 2, 3, 42)
REMOTE_RUN_ROOT = "/root/autodl-tmp/TempoFound-TKDE-runs"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _mrr_stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "per_seed": {str(seed): value for seed, value in zip(SEEDS, values, strict=True)},
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            ranks[index] = average
        start = stop
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _average_ranks(left)
    y = _average_ranks(right)
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _remote(path: Path, raw_root: Path) -> str:
    relative = path.relative_to(raw_root).as_posix()
    return f"{REMOTE_RUN_ROOT}/{relative}"


def collect_socialevo_records(raw_root: Path) -> list[dict[str, Any]]:
    raw_root = Path(raw_root)
    run = raw_root / "hrcsr-lastfm-socialevo-5seed-v1"
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        suite_path = run / "calib_suite" / "socialevo" / f"seed{seed}" / "transfer" / "rasp.json"
        strict_path = run / "calib_strict_ranked" / "socialevo" / f"seed{seed}" / "transfer" / "rasp.json"
        all_path = run / "calib_suite" / "socialevo" / f"seed{seed}" / "transfer" / "all_sources.json"
        tgat_path = raw_root / "socialevo-target-only-tgat" / "socialevo" / f"seed{seed}" / "summary.json"
        mixer_path = raw_root / "socialevo-target-only-graphmixer" / "socialevo" / f"seed{seed}" / "summary.json"

        suite = _read(suite_path)
        strict = _read(strict_path)
        all_source = _read(all_path)
        tgat = _read(tgat_path)
        mixer = _read(mixer_path)
        sweep = suite["safety_gate_sweep"]
        identity = float(sweep["identity"]["test_mrr"])
        route = float(sweep["route"]["test_mrr"])
        hard = float(sweep["hard"]["test_mrr"])
        soft = float(sweep["soft"]["test_mrr"])
        future_gain = route - identity
        gate_open = float(suite["target_gate"]) > 0.5
        decision_correct = (gate_open and future_gain > 0.0) or (
            (not gate_open) and future_gain <= 0.0
        )
        safety_violation = gate_open and future_gain < 0.0
        missed_opportunity = (not gate_open) and future_gain > 0.0
        calibration_requested = int(suite["target_calibration_events_reserved"])
        calibration_retained = int(suite["target_gate_calibration_events"])
        row: dict[str, Any] = {
            "Target": "socialevo",
            "Seed": seed,
            "AggregateSource": "direct_per_seed",
            "HRCSRRawFile": suite_path.as_posix(),
            "HRCSRRemoteFile": _remote(suite_path, raw_root),
            "HRCSRRawMTimeUTC": suite_path.stat().st_mtime,
            "StrictRawFile": strict_path.as_posix(),
            "StrictRemoteFile": _remote(strict_path, raw_root),
            "StrictRawMTimeUTC": strict_path.stat().st_mtime,
            "AllSourceRawFile": all_path.as_posix(),
            "AllSourceRemoteFile": _remote(all_path, raw_root),
            "AllSourceRawMTimeUTC": all_path.stat().st_mtime,
            "TGATRawFile": tgat_path.as_posix(),
            "TGATRemoteFile": _remote(tgat_path, raw_root),
            "TGATRawMTimeUTC": tgat_path.stat().st_mtime,
            "GraphMixerRawFile": mixer_path.as_posix(),
            "GraphMixerRemoteFile": _remote(mixer_path, raw_root),
            "GraphMixerRawMTimeUTC": mixer_path.stat().st_mtime,
            "HRCSRConfigDigest": suite.get("config_digest", "UNRECORDED"),
            "StrictConfigDigest": strict.get("config_digest", "UNRECORDED"),
            "TGATConfigDigest": tgat.get("config_digest", "UNRECORDED"),
            "GraphMixerConfigDigest": mixer.get("config_digest", "UNRECORDED"),
            "SelectedSources": "+".join(suite.get("sources", [])),
            "SplitID": suite.get("split_id", suite.get("test_split_id", "UNRECORDED")),
            "NegativeSampler": f"{suite.get('negative_policy', 'UNRECORDED')};{suite.get('negatives_per_positive', 'UNRECORDED')}_per_positive",
            "SparseQueryPolicy": suite.get("fallback_policy", "skip_unsampleable_if_no_historical_candidate"),
            "CheckpointSelection": suite.get("checkpoint_selection_scope", "UNRECORDED"),
            "HRCSRResultHash": "UNRECORDED",
            "StrictResultHash": "UNRECORDED",
            "AllSourceResultHash": "UNRECORDED",
            "TGATResultHash": "UNRECORDED",
            "GraphMixerResultHash": "UNRECORDED",
            "HRCSRCheckpointHash": "UNRECORDED",
            "StrictCheckpointHash": "UNRECORDED",
            "AllSourceCheckpointHash": "UNRECORDED",
            "TGATCheckpointHash": "UNRECORDED",
            "GraphMixerCheckpointHash": "UNRECORDED",
            "DatasetPreprocessingHash": "UNRECORDED",
            "CalibrationRequested": calibration_requested,
            "CalibrationRetained": calibration_retained,
            "CalibrationSkipped": calibration_requested - calibration_retained,
            "CalibrationSkipRate": (calibration_requested - calibration_retained) / calibration_requested,
            "TestRequested": suite.get("routed_test_total_queries", suite.get("test_total_queries", "UNRECORDED")),
            "TestRetained": suite.get("routed_test_valid_queries", suite.get("test_valid_queries", "UNRECORDED")),
            "TestSkipped": suite.get("routed_test_skipped_queries", suite.get("test_skipped_queries", "UNRECORDED")),
            "TestFallbackCount": suite.get("routed_test_fallback_count", suite.get("test_fallback_count", "UNRECORDED")),
            "CollisionRejectionCount": suite.get("routed_test_collision_rejection_count", suite.get("test_collision_rejection_count", "UNRECORDED")),
            "TestCounterSource": "RECORDED_IN_RESULT" if suite.get("routed_test_total_queries", suite.get("test_total_queries")) is not None else "UNRECORDED",
            "StrictRASPMRR": float(strict["test_mrr"]),
            "IdentityMRR": identity,
            "RouteMRR": route,
            "HRCSRHardMRR": hard,
            "SoftMRR": soft,
            "AllSourceMRR": float(all_source["test_mrr"]),
            "TGATMRR": float(tgat["test_mrr"]),
            "GraphMixerMRR": float(mixer["test_mrr"]),
            "GateLCB": float(suite["target_gate_lcb"]),
            "GateOpen": int(gate_open),
            "FutureRouteMinusIdentity": future_gain,
            "DecisionCorrect": int(decision_correct),
            "SafetyViolation": int(safety_violation),
            "MissedOpportunity": int(missed_opportunity),
            "SelectorRegret": max(route, identity) - hard,
            "HardMinusStrict": hard - float(strict["test_mrr"]),
            "IdentityMinusStrict": identity - float(strict["test_mrr"]),
        }
        records.append(row)
    return records


def enrich_with_remote_manifest(
    records: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach audit-time hashes and explicitly reconstructed sparse counters."""

    artifacts = {
        str(row["path"]): row
        for row in manifest.get("artifacts", [])
        if row.get("exists")
    }
    dataset = next(
        (
            row
            for path, row in artifacts.items()
            if path.endswith("/processed/ml_socialevo.csv")
        ),
        None,
    )
    test = manifest.get("query_population_reconstruction", {}).get("test", {})
    enriched: list[dict[str, Any]] = []
    for original in records:
        row = dict(original)
        remote_fields = {
            "HRCSR": "HRCSRRemoteFile",
            "Strict": "StrictRemoteFile",
            "AllSource": "AllSourceRemoteFile",
            "TGAT": "TGATRemoteFile",
            "GraphMixer": "GraphMixerRemoteFile",
        }
        for label, field in remote_fields.items():
            result_path = str(row[field])
            result = artifacts.get(result_path)
            if result is not None:
                row[f"{label}ResultHash"] = result.get("sha256", "UNRECORDED")
                row[f"{label}RawMTimeUTC"] = result.get("mtime_utc", "UNRECORDED")
            if label == "HRCSR":
                checkpoint_path = result_path.replace("/transfer/rasp.json", "/final/rasp/pretrained.pth")
            elif label == "Strict":
                checkpoint_path = result_path.replace("/transfer/rasp.json", "/final/rasp/pretrained.pth")
            elif label == "AllSource":
                checkpoint_path = result_path.replace("/transfer/all_sources.json", "/final/all_sources/pretrained.pth")
            else:
                checkpoint_path = ""
            checkpoint = artifacts.get(checkpoint_path)
            if checkpoint is not None:
                row[f"{label}CheckpointHash"] = checkpoint.get("sha256", "UNRECORDED")
        composite = manifest.get("dataset_preprocessing_composite_sha256")
        if composite:
            row["DatasetPreprocessingHash"] = composite
        elif dataset is not None:
            row["DatasetPreprocessingHash"] = dataset.get("sha256", "UNRECORDED")
        if all(key in test for key in ("requested", "retained", "skipped")):
            row["TestRequested"] = int(test["requested"])
            row["TestRetained"] = int(test["retained"])
            row["TestSkipped"] = int(test["skipped"])
            row["TestFallbackCount"] = int(test.get("fallback_count", 0))
            row["CollisionRejectionCount"] = int(test.get("collision_rejection_count", 0))
            row["TestCounterSource"] = "RECONSTRUCTED_FROM_DATASET_AND_FINAL_SAMPLER"
        enriched.append(row)
    return enriched


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if [int(row["Seed"]) for row in records] != list(SEEDS):
        raise ValueError(f"expected direct SocialEvo seeds {list(SEEDS)}")
    methods = {
        "HRCSR-Hard": "HRCSRHardMRR",
        "Strict RASP": "StrictRASPMRR",
        "All-source": "AllSourceMRR",
        "Target-only T-GAT": "TGATMRR",
        "Target-only GraphMixer": "GraphMixerMRR",
    }
    coverage: dict[str, Any] = {}
    statistics: dict[str, Any] = {}
    for method, key in methods.items():
        present = [row for row in records if row[key] != "MISSING"]
        seeds = [int(row["Seed"]) for row in present]
        coverage[method] = {
            "n": len(present),
            "seeds": seeds,
            "status": "COMPLETE FIVE-SEED COVERAGE" if seeds == list(SEEDS) else "MISSING FIVE-SEED COVERAGE",
        }
        if len(present) == len(SEEDS):
            statistics[method] = _mrr_stats([float(row[key]) for row in present])

    differences = [float(row["HardMinusStrict"]) for row in records]
    low, high = paired_bootstrap_ci(differences, seed=20260911)
    all_source_differences = [
        float(row["HRCSRHardMRR"]) - float(row["AllSourceMRR"])
        for row in records
    ]
    gains = [float(row["FutureRouteMinusIdentity"]) for row in records]
    lcbs = [float(row["GateLCB"]) for row in records]
    selector_regrets = [float(row["SelectorRegret"]) for row in records]
    return {
        "target": "socialevo",
        "seeds": list(SEEDS),
        "authoritative_source": "direct per-seed JSON; paired_results.json intentionally ignored",
        "coverage": coverage,
        "method_statistics": statistics,
        "hrcsr_minus_strict": {
            "mean_delta": mean(differences),
            "median_delta": median(differences),
            "sample_std_delta": stdev(differences),
            "paired_bootstrap_95_ci": [low, high],
            "positive_seeds": sum(value > 0.0 for value in differences),
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
            "per_seed_delta": {str(seed): value for seed, value in zip(SEEDS, differences, strict=True)},
            "interpretation": "descriptive because n=5",
        },
        "hrcsr_minus_all_source": {
            "mean_delta": mean(all_source_differences),
            "positive_seeds": sum(value > 0.0 for value in all_source_differences),
            "per_seed_delta": {
                str(seed): value
                for seed, value in zip(SEEDS, all_source_differences, strict=True)
            },
            "interpretation": "descriptive because n=5",
        },
        "gate_diagnostics": {
            "gate_open_count": sum(int(row["GateOpen"]) for row in records),
            "decision_accuracy": mean(float(row["DecisionCorrect"]) for row in records),
            "safety_violations": sum(int(row["SafetyViolation"]) for row in records),
            "missed_opportunities": sum(int(row["MissedOpportunity"]) for row in records),
            "mean_selector_regret": mean(selector_regrets),
            "max_selector_regret": max(selector_regrets),
            "spearman_lcb_vs_future_gain": _spearman(lcbs, gains),
            "mean_future_route_minus_identity": mean(gains),
        },
        "protocol": {
            "calibration_requested": sorted({int(row["CalibrationRequested"]) for row in records}),
            "calibration_retained": sorted({int(row["CalibrationRetained"]) for row in records}),
            "calibration_skipped": sorted({int(row["CalibrationSkipped"]) for row in records}),
            "calibration_skip_rate": sorted({float(row["CalibrationSkipRate"]) for row in records}),
            "test_query_counters": sorted({str(row["TestCounterSource"]) for row in records}),
            "config_digests": sorted({str(row["HRCSRConfigDigest"]) for row in records}),
        },
    }


def summarize_gate_panels(
    dense_report: dict[str, Any], sparse_summary: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate audited gate decisions without hiding panel denominators."""

    dense_rows = [
        row
        for target in dense_report.get("targets", [])
        for row in target.get("rows", [])
    ]
    sparse_gate = sparse_summary["gate_diagnostics"]

    def panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("dense gate report has no seed-level rows")
        return {
            "n_decisions": len(rows),
            "gate_open_count": sum(bool(row["gate_open"]) for row in rows),
            "decision_correct_count": sum(bool(row["decision_correct"]) for row in rows),
            "safety_violations": sum(bool(row["safety_violation"]) for row in rows),
            "missed_opportunities": sum(bool(row["missed_opportunity"]) for row in rows),
            "mean_selector_regret": mean(float(row["selector_regret"]) for row in rows),
        }

    dense = panel(dense_rows)
    sparse = {
        "n_decisions": len(sparse_summary["seeds"]),
        "gate_open_count": int(sparse_gate["gate_open_count"]),
        "decision_correct_count": round(
            float(sparse_gate["decision_accuracy"]) * len(sparse_summary["seeds"])
        ),
        "safety_violations": int(sparse_gate["safety_violations"]),
        "missed_opportunities": int(sparse_gate["missed_opportunities"]),
        "mean_selector_regret": float(sparse_gate["mean_selector_regret"]),
    }
    total_n = dense["n_decisions"] + sparse["n_decisions"]
    combined_regret = (
        dense["mean_selector_regret"] * dense["n_decisions"]
        + sparse["mean_selector_regret"] * sparse["n_decisions"]
    ) / total_n
    combined = {
        "n_decisions": total_n,
        "gate_open_count": dense["gate_open_count"] + sparse["gate_open_count"],
        "decision_correct_count": dense["decision_correct_count"] + sparse["decision_correct_count"],
        "safety_violations": dense["safety_violations"] + sparse["safety_violations"],
        "missed_opportunities": dense["missed_opportunities"] + sparse["missed_opportunities"],
        "mean_selector_regret": combined_regret,
    }
    return {
        "dense": dense,
        "sparse": sparse,
        "all_audited": combined,
        "reporting_rule": "dense and sparse panels remain separate; the all-audited panel is descriptive only",
    }


def collect_dense_gate_report(raw_root: Path) -> dict[str, Any]:
    """Rebuild the 25 dense decisions from authoritative per-seed artifacts."""

    branches = {
        "wikipedia": "hrcsr-ablation-matched-v1",
        "reddit": "hrcsr-ablation-matched-v1",
        "mooc": "hrcsr-ablation-matched-v1",
        "uci": "hrcsr-crossdomain-5seed-v1",
        "enron": "hrcsr-crossdomain-5seed-v1",
    }
    targets: list[dict[str, Any]] = []
    for target, branch in branches.items():
        rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            path = (
                Path(raw_root)
                / branch
                / "calib_suite"
                / target
                / f"seed{seed}"
                / "transfer"
                / "rasp.json"
            )
            result = _read(path)
            sweep = result["safety_gate_sweep"]
            identity = float(sweep["identity"]["test_mrr"])
            route = float(sweep["route"]["test_mrr"])
            hard = float(sweep["hard"]["test_mrr"])
            gate_open = float(result["target_gate"]) > 0.5
            gain = route - identity
            rows.append({
                "target": target,
                "seed": seed,
                "raw_file": path.as_posix(),
                "calibration_lcb": float(result["target_gate_lcb"]),
                "gate_open": gate_open,
                "decision_correct": (gate_open and gain > 0.0) or ((not gate_open) and gain <= 0.0),
                "safety_violation": gate_open and gain < 0.0,
                "missed_opportunity": (not gate_open) and gain > 0.0,
                "selector_regret": max(route, identity) - hard,
            })
        targets.append({"target": target, "rows": rows})
    return {
        "targets": targets,
        "authoritative_source": "direct per-seed transfer/rasp.json artifacts",
    }


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["Method", "N", "MeanMRR", "SampleStd", "Seeds", "Status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, coverage in summary["coverage"].items():
            stats = summary["method_statistics"].get(method, {})
            writer.writerow({
                "Method": method,
                "N": coverage["n"],
                "MeanMRR": stats.get("mean", "NA"),
                "SampleStd": stats.get("sample_std", "NA"),
                "Seeds": "+".join(str(seed) for seed in coverage["seeds"]),
                "Status": coverage["status"],
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("results/raw/remote"))
    parser.add_argument("--audit-csv", type=Path, default=Path("results/socialevo_5seed_audit.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("results/aggregated/socialevo_5seed_summary.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/aggregated/socialevo_5seed_summary.csv"))
    parser.add_argument(
        "--remote-manifest",
        type=Path,
        default=Path("results/raw/remote/socialevo_remote_artifact_manifest.json"),
    )
    parser.add_argument(
        "--gate-panels-json",
        type=Path,
        default=Path("results/aggregated/gate_summary_dense_sparse.json"),
    )
    args = parser.parse_args()

    records = collect_socialevo_records(args.raw_root)
    if args.remote_manifest.is_file():
        records = enrich_with_remote_manifest(
            records, json.loads(args.remote_manifest.read_text(encoding="utf-8"))
        )
    summary = summarize(records)
    _write_records(args.audit_csv, records)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_csv(args.summary_csv, summary)
    dense_report = collect_dense_gate_report(args.raw_root)
    panels = summarize_gate_panels(dense_report, summary)
    args.gate_panels_json.parent.mkdir(parents=True, exist_ok=True)
    args.gate_panels_json.write_text(json.dumps(panels, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
