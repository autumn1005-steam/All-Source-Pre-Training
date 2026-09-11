"""Analyze OOS-HRCSR safety as a decision system, not only as mean MRR.

Reads existing result directories and reports:
- calibration/test route-gain sign agreement
- hard-gate decision accuracy
- safety violation rate (opened gate when route hurt)
- missed opportunity rate (closed gate when route helped)
- selector regret to a route-vs-identity oracle
- hard-gated RASP path relative to matched strict_ranked RASP

No model is retrained by this script.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a-mx)*(b-my) for a,b in zip(rx, ry))
    dx = math.sqrt(sum((a-mx)**2 for a in rx))
    dy = math.sqrt(sum((b-my)**2 for b in ry))
    return None if dx == 0.0 or dy == 0.0 else num/(dx*dy)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_dirs(root: Path, target: str) -> list[Path]:
    target_dir = root / "calib_suite" / target
    return sorted(
        [p for p in target_dir.glob("seed*") if p.is_dir()],
        key=lambda p: int(p.name.removeprefix("seed")),
    )


def analyze_target(root: Path, target: str) -> dict[str, Any]:
    rows = []
    for seed_dir in _seed_dirs(root, target):
        seed = int(seed_dir.name.removeprefix("seed"))
        suite = _load(seed_dir / "transfer" / "rasp.json")
        strict_path = root / "calib_strict_ranked" / target / f"seed{seed}" / "transfer" / "rasp.json"
        strict = _load(strict_path) if strict_path.exists() else None
        sweep = suite["safety_gate_sweep"]
        route = float(sweep["route"]["test_mrr"])
        identity = float(sweep["identity"]["test_mrr"])
        hard = float(sweep["hard"]["test_mrr"])
        soft = float(sweep["soft"]["test_mrr"])
        lcb = float(suite["target_gate_lcb"])
        cal_gain = float(suite["target_gate_calibration_mrr_delta"])
        future_gain = route - identity
        opened = bool(float(suite["target_gate"]) > 0.5)
        route_better = future_gain > 0.0
        decision_correct = opened == route_better
        selector_regret = max(route, identity) - hard
        row = {
            "seed": seed,
            "calibration_mrr_gain": cal_gain,
            "calibration_lcb": lcb,
            "gate_open": opened,
            "route_test_mrr": route,
            "identity_test_mrr": identity,
            "hard_test_mrr": hard,
            "soft_test_mrr": soft,
            "future_route_gain": future_gain,
            "decision_correct": decision_correct,
            "safety_violation": opened and not route_better,
            "missed_opportunity": (not opened) and route_better,
            "selector_regret": selector_regret,
        }
        if strict is not None:
            row["strict_ranked_test_mrr"] = float(strict["test_mrr"])
            row["hard_minus_strict_ranked"] = hard - float(strict["test_mrr"])
            row["identity_minus_strict_ranked"] = identity - float(strict["test_mrr"])
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"no calib_suite seed directories found for {target}")
    n = len(rows)
    lcbs = [r["calibration_lcb"] for r in rows]
    future = [r["future_route_gain"] for r in rows]
    result: dict[str, Any] = {
        "target": target,
        "n_seeds": n,
        "seeds": [r["seed"] for r in rows],
        "gate_open_rate": mean(float(r["gate_open"]) for r in rows),
        "gate_decision_accuracy": mean(float(r["decision_correct"]) for r in rows),
        "safety_violation_rate": mean(float(r["safety_violation"]) for r in rows),
        "missed_opportunity_rate": mean(float(r["missed_opportunity"]) for r in rows),
        "mean_future_route_gain": mean(future),
        "mean_selector_regret": mean(r["selector_regret"] for r in rows),
        "max_selector_regret": max(r["selector_regret"] for r in rows),
        "lcb_vs_future_gain_spearman": _spearman(lcbs, future),
        "rows": rows,
    }
    hard_strict = [r["hard_minus_strict_ranked"] for r in rows if "hard_minus_strict_ranked" in r]
    if hard_strict:
        result["hard_minus_strict_ranked_mean"] = mean(hard_strict)
        result["hard_minus_strict_ranked_positive_seeds"] = sum(x > 0 for x in hard_strict)
        result["hard_minus_strict_ranked_sample_std"] = stdev(hard_strict) if len(hard_strict) > 1 else 0.0
    identity_strict = [r["identity_minus_strict_ranked"] for r in rows if "identity_minus_strict_ranked" in r]
    if identity_strict:
        result["identity_minus_strict_ranked_mean"] = mean(identity_strict)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--targets", nargs="+", default=["wikipedia", "reddit", "mooc"])
    p.add_argument("--write", default=None, help="optional JSON output path")
    args = p.parse_args()
    root = Path(args.output_root)
    summaries = []
    for target in args.targets:
        try:
            summaries.append(analyze_target(root, target))
        except FileNotFoundError as exc:
            print(f"SKIP {exc}")
    payload = {"output_root": str(root), "targets": summaries}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
