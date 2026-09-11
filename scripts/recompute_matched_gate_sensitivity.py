"""Recompute gate confidence on frozen authoritative calibration evidence.

No training or prediction is performed.  Each confidence point resamples the
same stored chronological window means with the same per-seed bootstrap RNG.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sweep_transfer_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    confidences: Iterable[float],
    replicates: int,
    min_gain: float = 0.0,
) -> list[dict[str, Any]]:
    window_means = payload.get("target_gate_window_means")
    if not isinstance(window_means, list) or not window_means:
        raise ValueError("authoritative transfer payload lacks target gate window means")
    values = np.asarray(window_means, dtype=np.float64)
    rng = np.random.default_rng(seed + 52009)
    bootstrap_means = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    identity = float(payload["router_off_reference_test_mrr"])
    route = float(payload["raw_routed_test_mrr"])
    future_gain = route - identity
    rows: list[dict[str, Any]] = []
    for confidence in confidences:
        confidence = float(confidence)
        lcb = float(np.quantile(bootstrap_means, 1.0 - confidence))
        gate = float(lcb > min_gain)
        rows.append(
            {
                "target": str(payload.get("target", "")),
                "seed": int(seed),
                "confidence": confidence,
                "lcb": lcb,
                "gate": gate,
                "future_route_minus_identity": future_gain,
                "correct_decision": bool((gate > 0.5) == (future_gain > 0.0)),
                "safety_violation": bool(gate > 0.5 and future_gain < 0.0),
                "missed_opportunity": bool(gate <= 0.5 and future_gain > 0.0),
                "identity_mrr": identity,
                "route_mrr": route,
                "hard_mrr": route if gate > 0.5 else identity,
                "config_digest": payload.get("config_digest"),
                "bootstrap_replicates": int(replicates),
                "bootstrap_seed": int(seed + 52009),
                "calibration_windows": int(len(values)),
                "authoritative_c095_gate": payload.get("target_gate"),
                "authoritative_c095_lcb": payload.get("target_gate_lcb"),
            }
        )
    return rows


def load_inputs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"input must be TARGET=PATH: {spec}")
        target, raw_path = spec.split("=", 1)
        inputs.append((target, Path(raw_path)))
    return inputs


def recompute(
    specs: Iterable[str],
    *,
    confidences: tuple[float, ...],
    replicates: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    checks: list[str] = []
    for target, directory in load_inputs(specs):
        for path in sorted(directory.glob("seed*/transfer/rasp.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            seed = int(path.parts[-3].removeprefix("seed"))
            payload["target"] = target
            target_rows = sweep_transfer_payload(
                payload,
                seed=seed,
                confidences=confidences,
                replicates=replicates,
            )
            for row in target_rows:
                row["raw_file"] = str(path)
            rows.extend(target_rows)
            c095 = next((row for row in target_rows if abs(row["confidence"] - 0.95) < 1e-12), None)
            if c095 is None:
                checks.append(f"FAIL {target} seed={seed}: c=.95 absent")
                continue
            gate_equal = float(c095["gate"]) == float(c095["authoritative_c095_gate"])
            lcb_equal = abs(float(c095["lcb"]) - float(c095["authoritative_c095_lcb"])) <= 1e-12
            checks.append(
                f"{'PASS' if gate_equal and lcb_equal else 'FAIL'} {target} seed={seed}: "
                f"gate={c095['gate']:.0f}/{float(c095['authoritative_c095_gate']):.0f} "
                f"lcb_delta={c095['lcb'] - float(c095['authoritative_c095_lcb']):.3e}"
            )
    return rows, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="TARGET=directory containing seed*/transfer/rasp.json")
    parser.add_argument("--confidences", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--check-output", type=Path, required=True)
    args = parser.parse_args()
    rows, checks = recompute(
        args.input,
        confidences=tuple(args.confidences),
        replicates=args.replicates,
    )
    if not rows:
        raise SystemExit("no authoritative transfer payloads found")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overall = "PASS" if checks and all(line.startswith("PASS ") for line in checks) else "FAIL"
    args.check_output.write_text(overall + "\n" + "\n".join(checks) + "\n", encoding="utf-8")
    print(f"{overall}: wrote {len(rows)} matched sensitivity rows")
    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
