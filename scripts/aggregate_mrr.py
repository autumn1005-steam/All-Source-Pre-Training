"""Aggregate matched MRR rows without inventing missing values."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def extract_method_values(row: dict[str, Any]) -> dict[str, Any]:
    """Extract values present in a paired result row.

    ``NA`` is returned when a branch is not present; this keeps legacy and
    modern protocols distinguishable during aggregation.
    """

    diagnostics = row.get("safety_diagnostics", {})
    rasp_diag = diagnostics.get("rasp", {}) if isinstance(diagnostics, dict) else {}
    sweep = rasp_diag.get("safety_gate_sweep", {}) if isinstance(rasp_diag, dict) else {}
    hard = sweep.get("hard", {}) if isinstance(sweep, dict) else {}
    return {
        "strict_rasp": row.get("rasp_test_mrr", "NA"),
        "all_source": row.get("all_sources_test_mrr", "NA"),
        "hrcsr_hard": hard.get("test_mrr", "NA") if isinstance(hard, dict) else "NA",
        "target": row.get("target", "NA"),
        "seed": row.get("seed", "NA"),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _read_rows(roots: list[Path], variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.rglob("paired_results.json")):
            if variant not in path.parts:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            for item in payload:
                if isinstance(item, dict):
                    row = extract_method_values(item)
                    row["source"] = str(path)
                    rows.append(row)
    return rows


def aggregate_roots(strict_roots: list[Path], hrcsr_roots: list[Path]) -> dict[str, Any]:
    strict_rows = _read_rows(strict_roots, "calib_strict_ranked")
    hrcsr_rows = _read_rows(hrcsr_roots, "calib_suite")
    targets = sorted({row["target"] for row in strict_rows + hrcsr_rows})
    grouped: dict[str, dict[str, Any]] = {}
    for target in targets:
        strict_target = [row for row in strict_rows if row["target"] == target]
        hrcsr_target = [row for row in hrcsr_rows if row["target"] == target]
        out: dict[str, Any] = {"strict_rows": strict_target, "hrcsr_rows": hrcsr_target}
        for method, source_rows in (
            ("strict_rasp", strict_target),
            ("all_source", hrcsr_target),
            ("hrcsr_hard", hrcsr_target),
        ):
            values = [float(row[method]) for row in source_rows if isinstance(row[method], (int, float))]
            if values:
                mean, std = _mean_std(values)
                out[method] = {"n": len(values), "mean": mean, "sample_std": std}
            else:
                out[method] = "NA"
        grouped[target] = out
    return {
        "strict_roots": [str(root) for root in strict_roots],
        "hrcsr_roots": [str(root) for root in hrcsr_roots],
        "targets": grouped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", action="append", type=Path, default=[])
    parser.add_argument("--hrcsr-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.strict_root and not args.hrcsr_root:
        parser.error("at least one --strict-root or --hrcsr-root is required")
    result = aggregate_roots(args.strict_root, args.hrcsr_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"targets": len(result["targets"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
