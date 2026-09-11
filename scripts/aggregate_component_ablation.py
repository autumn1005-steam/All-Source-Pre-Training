#!/usr/bin/env python3
"""Aggregate the matched-width HRCSR component ablation from raw JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any


VARIANTS = (
    "a0_strict_rasp",
    "a1_expert_identity",
    "a2_response_router",
    "a3_local_acceptance",
    "a4_pairwise_regret",
    "a5_oos_gate",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in args.targets:
        for seed in args.seeds:
            def resolve(templates: list[str]) -> Path:
                candidates = [Path(template.format(target=target, seed=seed)) for template in templates]
                return next((path for path in candidates if path.is_file()), candidates[0])

            roots = {
                "a0_strict_rasp": resolve(args.strict_template),
                "a5_oos_gate": resolve(args.full_template),
            }
            for variant in ("a1_expert_identity", "a2_response_router", "a3_local_acceptance", "a4_pairwise_regret"):
                ablation_base = Path(str(args.ablation_root).format(target=target))
                roots[variant] = ablation_base / variant / target / f"seed{seed}.json"
            for variant in VARIANTS:
                path = roots[variant]
                if not path.is_file():
                    raise FileNotFoundError(f"missing raw result for {target} seed={seed} {variant}: {path}")
                payload = _read(path)
                rows.append({
                    "target": target,
                    "seed": int(seed),
                    "variant": variant,
                    "mrr": float(payload["test_mrr"]),
                    "raw_file": path.as_posix(),
                    "checkpoint_selection_mode": payload.get("checkpoint_selection_mode"),
                    "negative_policy": payload.get("negative_policy"),
                    "gate_mode": payload.get("target_gate_mode"),
                    "config_digest": payload.get("config_digest", payload.get("ablation_config_digest", "UNRECORDED")),
                })
    baseline = {(row["target"], row["seed"]): row["mrr"] for row in rows if row["variant"] == "a0_strict_rasp"}
    for row in rows:
        base = baseline[(row["target"], row["seed"])]
        row["delta_vs_a0"] = row["mrr"] - base
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    for target in sorted({row["target"] for row in rows}):
        for variant in VARIANTS:
            subset = [row for row in rows if row["target"] == target and row["variant"] == variant]
            if not subset:
                continue
            deltas = [float(row["delta_vs_a0"]) for row in subset]
            values = [float(row["mrr"]) for row in subset]
            summaries.append({
                "target": target,
                "variant": variant,
                "n_seeds": len(subset),
                "mrr_mean": statistics.mean(values),
                "mrr_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "delta_vs_a0_mean": statistics.mean(deltas),
                "delta_vs_a0_median": statistics.median(deltas),
                "delta_vs_a0_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "positive_seeds_vs_a0": sum(delta > 0 for delta in deltas),
            })
    return {"variants": list(VARIANTS), "seed_rows": rows, "summaries": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", required=True, type=Path)
    parser.add_argument("--strict-template", action="append", required=True)
    parser.add_argument("--full-template", action="append", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3, 42))
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    payload = aggregate(_rows(args))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["seed_rows"][0]))
        writer.writeheader()
        writer.writerows(payload["seed_rows"])
    print(json.dumps({"seed_rows": len(payload["seed_rows"]), "summaries": len(payload["summaries"])}))


if __name__ == "__main__":
    main()
