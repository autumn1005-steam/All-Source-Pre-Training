"""Aggregate matched-width checkpoint-boundary outputs without imputation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_unified_table import exact_sign_flip_p, paired_bootstrap_ci


MODES = ("validation", "fixed_epoch", "prefix_only")


def aggregate(root: Path) -> list[dict[str, Any]]:
    """Compatibility helper for the lightweight unit-test fixture.

    The production path is :func:`aggregate_roots`, which joins audit outputs
    to strict-RASP controls and emits seed-level plus summary records.  This
    small helper intentionally accepts the historical fixture schema used by
    older tests and keeps its mode-level diagnostics explicit.
    """
    rows: list[dict[str, Any]] = []
    for mode_dir in sorted(Path(root).iterdir()):
        if not mode_dir.is_dir() or mode_dir.name not in MODES:
            continue
        for path in sorted(mode_dir.glob("*/seed*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            gain = float(payload.get("raw_route_minus_reference_test_mrr", 0.0))
            gate = float(payload.get("target_gate", 0.0)) > 0.5
            rows.append(
                {
                    "target": str(payload.get("target", path.parent.name)),
                    "mode": mode_dir.name,
                    "seed": int(payload.get("seed", 0)),
                    "test_mrr": float(payload.get("test_mrr", 0.0)),
                    "route_minus_identity": gain,
                    "gate_open": gate,
                    "safety_violation_count": int(gate and gain < 0),
                    "missed_opportunity_count": int((not gate) and gain > 0),
                }
            )
    summaries: list[dict[str, Any]] = []
    for target in sorted({row["target"] for row in rows}):
        for mode in MODES:
            subset = [row for row in rows if row["target"] == target and row["mode"] == mode]
            if not subset:
                continue
            summaries.append(
                {
                    "target": target,
                    "mode": mode,
                    "n_seeds": len(subset),
                    "test_mrr_mean": statistics.mean(row["test_mrr"] for row in subset),
                    "route_minus_identity_mean": statistics.mean(row["route_minus_identity"] for row in subset),
                    "safety_violation_count": sum(row["safety_violation_count"] for row in subset),
                    "missed_opportunity_count": sum(row["missed_opportunity_count"] for row in subset),
                    "seed_rows": subset,
                }
            )
    return summaries


def _read_audit_roots(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for path in sorted(root.glob("*/*/seed*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == "PENDING":
                continue
            mode = str(payload.get("checkpoint_selection_mode", path.parts[-3]))
            if mode not in MODES:
                continue
            required = (
                "target",
                "seed",
                "router_off_reference_test_mrr",
                "raw_routed_test_mrr",
                "test_mrr",
                "target_gate",
            )
            missing = [key for key in required if key not in payload]
            if missing:
                raise ValueError(f"{path}: missing {missing}")
            rows.append({**payload, "protocol": mode, "raw_file": path.as_posix()})
    return rows


def _read_strict_roots(roots: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    strict = {}
    for root in roots:
        for path in sorted(root.glob("*/seed*/transfer/rasp.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = (str(payload["target"]), int(payload["seed"]))
            if key in strict:
                raise ValueError(f"duplicate strict baseline for {key}")
            strict[key] = {"mrr": float(payload["test_mrr"]), "raw_file": path.as_posix()}
    return strict


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def aggregate_roots(audit_roots: list[Path], strict_roots: list[Path]) -> dict[str, Any]:
    audits = _read_audit_roots(audit_roots)
    strict = _read_strict_roots(strict_roots)
    seed_rows = []
    for row in audits:
        key = (str(row["target"]), int(row["seed"]))
        if key not in strict:
            raise ValueError(f"missing strict baseline for {key}")
        identity = float(row["router_off_reference_test_mrr"])
        route = float(row["raw_routed_test_mrr"])
        hard = float(row["test_mrr"])
        gate = float(row["target_gate"]) > 0.5
        strict_mrr = strict[key]["mrr"]
        seed_rows.append(
            {
                "target": key[0],
                "seed": key[1],
                "protocol": row["protocol"],
                "identity_mrr": identity,
                "route_mrr": route,
                "hard_mrr": hard,
                "strict_mrr": strict_mrr,
                "hard_minus_strict": hard - strict_mrr,
                "gate_open": gate,
                "gate_lcb": row.get("target_gate_lcb"),
                "safety_violation": gate and route < identity,
                "missed_opportunity": (not gate) and route > identity,
                "selector_regret": max(route, identity) - hard,
                "checkpoint_selected_epoch": row.get("checkpoint_selected_epoch"),
                "checkpoint_selection_scope": row.get("checkpoint_selection_scope"),
                "raw_file": row["raw_file"],
                "strict_raw_file": strict[key]["raw_file"],
            }
        )

    summaries = []
    targets = sorted({row["target"] for row in seed_rows})
    for target in targets:
        for mode in MODES:
            subset = [row for row in seed_rows if row["target"] == target and row["protocol"] == mode]
            if not subset:
                continue
            deltas = [row["hard_minus_strict"] for row in subset]
            lo, hi = paired_bootstrap_ci(
                deltas,
                seed=20260911 + MODES.index(mode) + 10 * targets.index(target),
            )
            summaries.append(
                {
                    "target": target,
                    "protocol": mode,
                    "n_seeds": len(subset),
                    "identity_mrr_mean": _mean([row["identity_mrr"] for row in subset]),
                    "identity_mrr_std": _std([row["identity_mrr"] for row in subset]),
                    "route_mrr_mean": _mean([row["route_mrr"] for row in subset]),
                    "route_mrr_std": _std([row["route_mrr"] for row in subset]),
                    "hard_mrr_mean": _mean([row["hard_mrr"] for row in subset]),
                    "hard_mrr_std": _std([row["hard_mrr"] for row in subset]),
                    "strict_mrr_mean": _mean([row["strict_mrr"] for row in subset]),
                    "hard_minus_strict_mean": _mean(deltas),
                    "hard_minus_strict_std": _std(deltas),
                    "hard_minus_strict_bootstrap_95_ci": [lo, hi],
                    "positive_seeds": sum(value > 0 for value in deltas),
                    "exact_two_sided_sign_flip_p": exact_sign_flip_p(deltas),
                    "gate_opens": sum(row["gate_open"] for row in subset),
                    "safety_violations": sum(row["safety_violation"] for row in subset),
                    "missed_opportunities": sum(row["missed_opportunity"] for row in subset),
                    "selector_regret_mean": _mean([row["selector_regret"] for row in subset]),
                    "statistical_scope": "descriptive because n=5" if len(subset) == 5 else f"incomplete n={len(subset)}",
                }
            )
    return {
        "audit_roots": [root.as_posix() for root in audit_roots],
        "strict_roots": [root.as_posix() for root in strict_roots],
        "seed_rows": seed_rows,
        "summaries": summaries,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", action="append", type=Path, required=True)
    parser.add_argument("--strict-root", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate_roots(args.audit_root, args.strict_root)
    if not payload["seed_rows"]:
        raise SystemExit("no completed checkpoint-boundary reports found")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(args.output_csv, payload["seed_rows"])
    print(json.dumps({"seed_rows": len(payload["seed_rows"]), "summaries": len(payload["summaries"])}))


if __name__ == "__main__":
    main()
