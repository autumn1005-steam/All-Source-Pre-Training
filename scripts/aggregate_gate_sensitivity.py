#!/usr/bin/env python3
"""Aggregate copied gate-confidence reports without recomputing or imputing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.root.glob("c*/oos_safety_decision_report.json")):
        confidence = path.parent.name.removeprefix("c")
        report = json.loads(path.read_text(encoding="utf-8"))
        for target in report.get("targets", []):
            rows.append(
                {
                    "confidence": float(confidence) / 100.0,
                    "target": target["target"],
                    "n_seeds": target["n_seeds"],
                    "gate_open_rate": target["gate_open_rate"],
                    "decision_accuracy": target["gate_decision_accuracy"],
                    "safety_violation_rate": target["safety_violation_rate"],
                    "missed_opportunity_rate": target["missed_opportunity_rate"],
                    "mean_future_route_gain": target["mean_future_route_gain"],
                    "mean_selector_regret": target["mean_selector_regret"],
                    "spearman_lcb_future_gain": target["lcb_vs_future_gain_spearman"],
                }
            )
    if not rows:
        raise SystemExit("no gate reports found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
