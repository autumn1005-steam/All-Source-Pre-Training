"""Rebuild the unified MRR table from seed-level raw result files.

The script never fills protocol metadata from manuscript prose. Missing fields
remain ``UNRECORDED`` so that an unsupported fairness claim cannot silently
enter the paper.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable


SEEDS = (0, 1, 2, 3, 42)
TARGETS = ("wikipedia", "reddit", "mooc", "uci", "enron")


def exact_sign_flip_p(differences: Iterable[float]) -> float:
    """Return the two-sided exact paired sign-flip p-value."""

    values = [float(value) for value in differences]
    if not values:
        return float("nan")
    observed = abs(mean(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(mean(sign * value for sign, value in zip(signs, values)))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def paired_bootstrap_ci(
    differences: Iterable[float],
    *,
    replicates: int = 20_000,
    seed: int = 20260911,
) -> tuple[float, float]:
    """Percentile CI for the paired mean difference."""

    values = [float(value) for value in differences]
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    boot = sorted(
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    )
    lo_index = max(0, math.floor(0.025 * (replicates - 1)))
    hi_index = min(replicates - 1, math.ceil(0.975 * (replicates - 1)))
    return boot[lo_index], boot[hi_index]


def _protocol_value(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return str(value) if value not in (None, "") else "UNRECORDED"


def _negative_sampler(payload: dict[str, Any]) -> str:
    policy = payload.get("negative_policy")
    count = payload.get("negatives_per_positive")
    if policy in (None, ""):
        return "UNRECORDED"
    if count in (None, ""):
        return str(policy)
    return f"{policy};{count}_per_positive"


def read_transfer_record(
    path: Path,
    *,
    method: str,
    value_key: str = "test_mrr",
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if value_key not in payload:
        raise KeyError(f"{path}: missing {value_key}")
    digest = payload.get("config_digest")
    code_version = f"config_digest:{digest}" if digest else "UNRECORDED"
    split_id = payload.get("split_id") or payload.get("test_split_id") or "UNRECORDED"
    protocol = payload.get("evaluation_protocol")
    metric = "MRR" if protocol in (None, "ranking") else f"MRR({protocol})"
    return {
        "Target": str(payload.get("target", "UNRECORDED")),
        "Method": method,
        "Seed": int(payload["seed"]),
        "RawFile": path.as_posix(),
        "Metric": metric,
        "SplitID": str(split_id),
        "NegativeSampler": _negative_sampler(payload),
        "Checkpoint": _protocol_value(payload, "checkpoint_selection_scope"),
        "MRR": float(payload[value_key]),
        "CodeVersion": code_version,
    }


def _expect(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def collect_records(raw_root: Path, split_manifest: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dense_roots = {
        "wikipedia": "hrcsr-ablation-matched-v1",
        "reddit": "hrcsr-ablation-matched-v1",
        "mooc": "hrcsr-ablation-matched-v1",
        "uci": "hrcsr-crossdomain-5seed-v1",
        "enron": "hrcsr-crossdomain-5seed-v1",
    }
    for target in TARGETS:
        branch = dense_roots[target]
        for seed in SEEDS:
            records.append(
                read_transfer_record(
                    _expect(raw_root / "target-only-matched-v1" / target / f"seed{seed}" / "summary.json"),
                    method="Target-only T-GAT",
                )
            )
            records.append(
                read_transfer_record(
                    _expect(raw_root / "graphmixer-target-only-matched-v1" / target / f"seed{seed}" / "summary.json"),
                    method="Target-only GraphMixer",
                )
            )
            records.append(
                read_transfer_record(
                    _expect(raw_root / branch / "calib_suite" / target / f"seed{seed}" / "transfer" / "all_sources.json"),
                    method="All-source",
                )
            )
            records.append(
                read_transfer_record(
                    _expect(raw_root / branch / "calib_strict_ranked" / target / f"seed{seed}" / "transfer" / "rasp.json"),
                    method="Strict RASP",
                )
            )
            records.append(
                read_transfer_record(
                    _expect(raw_root / branch / "calib_suite" / target / f"seed{seed}" / "transfer" / "rasp.json"),
                    method="HRCSR-Hard",
                )
            )
    if split_manifest is not None:
        split_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
        split_code = split_payload["split_code_sha256"]
        datasets = split_payload["datasets"]
        for record in records:
            target = record["Target"]
            dataset = datasets[target]
            record["SplitID"] = (
                f"csv_sha256:{dataset['csv_sha256']};"
                f"split_code_sha256:{split_code};q70_q85"
            )
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for target in TARGETS:
        table[target] = {}
        for method in sorted({str(row["Method"]) for row in records}):
            rows = [row for row in records if row["Target"] == target and row["Method"] == method]
            values = [float(row["MRR"]) for row in rows]
            if not values:
                continue
            table[target][method] = {
                "n": len(values),
                "mean": mean(values),
                "sample_std": stdev(values) if len(values) > 1 else float("nan"),
                "seed_mrr": {str(row["Seed"]): row["MRR"] for row in rows},
            }

        strict = {
            int(row["Seed"]): float(row["MRR"])
            for row in records
            if row["Target"] == target and row["Method"] == "Strict RASP"
        }
        hrcsr = {
            int(row["Seed"]): float(row["MRR"])
            for row in records
            if row["Target"] == target and row["Method"] == "HRCSR-Hard"
        }
        paired_seeds = sorted(set(strict) & set(hrcsr))
        differences = [hrcsr[seed] - strict[seed] for seed in paired_seeds]
        low, high = paired_bootstrap_ci(differences, seed=20260911 + TARGETS.index(target))
        table[target]["HRCSR_minus_Strict_statistics"] = {
            "n": len(differences),
            "seeds": paired_seeds,
            "mean_delta": mean(differences),
            "median_delta": median(differences),
            "sample_std_delta": stdev(differences) if len(differences) > 1 else float("nan"),
            "bootstrap_95_ci": [low, high],
            "positive_seeds": sum(value > 0 for value in differences),
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
            "interpretation": "descriptive because n=5",
        }
    return table


def check_protocol_consistency(
    records: list[dict[str, Any]],
    *,
    targets: Iterable[str] = TARGETS,
    methods: Iterable[str] | None = None,
    seeds: Iterable[int] = SEEDS,
) -> dict[str, Any]:
    """Check the seed grid and fields that must match within each target."""

    expected_methods = tuple(methods or sorted({str(row["Method"]) for row in records}))
    expected_seeds = tuple(int(seed) for seed in seeds)
    errors: list[str] = []
    for target in targets:
        target_rows = [row for row in records if row["Target"] == target]
        for method in expected_methods:
            actual = sorted(int(row["Seed"]) for row in target_rows if row["Method"] == method)
            if actual != sorted(expected_seeds):
                errors.append(f"{target}/{method}: seeds={actual}, expected={sorted(expected_seeds)}")
        for field in ("Metric", "SplitID", "NegativeSampler", "Checkpoint"):
            values = sorted({str(row[field]) for row in target_rows})
            if len(values) != 1 or values == ["UNRECORDED"]:
                errors.append(f"{target}: {field}={values}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "record_count": len(records),
        "targets": list(targets),
        "methods": list(expected_methods),
        "seeds": list(expected_seeds),
        "errors": errors,
        "note": "CodeVersion is audited separately and is not assumed from manuscript prose.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Target",
        "Method",
        "Seed",
        "RawFile",
        "Metric",
        "SplitID",
        "NegativeSampler",
        "Checkpoint",
        "MRR",
        "CodeVersion",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    parser.add_argument("--aggregate-json", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--consistency-check", type=Path)
    args = parser.parse_args()
    records = collect_records(args.raw_root, args.split_manifest)
    write_csv(args.provenance_csv, records)
    result = aggregate(records)
    args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    check = check_protocol_consistency(records)
    if args.consistency_check:
        args.consistency_check.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"TABLE VII PROTOCOL CONSISTENCY: {check['status']}",
            f"records={check['record_count']}",
            f"targets={','.join(check['targets'])}",
            f"methods={','.join(check['methods'])}",
            f"seeds={','.join(str(seed) for seed in check['seeds'])}",
        ]
        lines.extend(f"ERROR: {error}" for error in check["errors"])
        lines.append(check["note"])
        args.consistency_check.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "targets": len(result), "status": check["status"]}))
    if check["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
