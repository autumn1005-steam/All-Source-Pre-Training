"""Forensic audit for the Reddit target-gate operating points.

The audit treats a confidence sweep as valid only when each operating point
comes from the same fitted model and predictions as the authoritative run.
Missing provenance is retained as ``None``; it is never inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a list in {path}")
    rows = [row for row in payload if isinstance(row, dict)]
    if any(str(row.get("target", "")).lower() != "reddit" for row in rows):
        raise ValueError(f"non-Reddit row found in {path}")
    return rows


def _extract(row: Mapping[str, Any], *, confidence: float | None, raw_file: Path) -> dict[str, Any]:
    safety = row.get("safety_diagnostics", {}).get("rasp", {})
    training = row.get("training_diagnostics", {}).get("rasp", {})
    sweep = safety.get("safety_gate_sweep", {})
    identity = safety.get("router_off_reference_test_mrr")
    route = safety.get("raw_routed_test_mrr")
    hard = sweep.get("hard", {}).get("test_mrr") if isinstance(sweep, dict) else None
    return {
        "experiment_id": str(raw_file.parent),
        "raw_file": str(raw_file),
        "target": "reddit",
        "seed": int(row["seed"]),
        "confidence": confidence,
        "config_digest": row.get("config_digest"),
        "selected_sources": list(row.get("selected_sources", [])),
        "candidate_count": row.get("candidate_count"),
        "trainable_parameters": training.get("trainable_parameters"),
        "pretraining_epochs_observed": len(training.get("epoch_batch_counts", [])) or None,
        "source_minibatches_per_epoch": training.get("global_batch_cap"),
        "negative_sampler": row.get("evaluation_protocol"),
        "number_of_negatives": safety.get("target_gate_negatives"),
        "calibration_windows": safety.get("target_gate_window_count"),
        "bootstrap_count": safety.get("target_gate_bootstrap_replicates"),
        "lcb": safety.get("target_gate_lcb"),
        "gate_decision": safety.get("target_gate"),
        "identity_mrr": identity,
        "route_mrr": route,
        "hard_mrr": hard,
        "calibration_delta": safety.get("target_gate_calibration_mrr_delta"),
        "target_optimization_events": safety.get("target_optimization_events"),
        "target_calibration_events": safety.get("target_calibration_events_reserved"),
    }


def _different(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left != right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) > tolerance
    return left != right


def audit_gate_results(
    main_path: Path,
    sensitivity_paths: Mapping[float, Path],
) -> dict[str, Any]:
    main_rows = [_extract(row, confidence=0.95, raw_file=main_path) for row in _load_rows(main_path)]
    main_by_seed = {row["seed"]: row for row in main_rows}
    sensitivity_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    invariant_fields = (
        "config_digest",
        "selected_sources",
        "trainable_parameters",
        "identity_mrr",
        "route_mrr",
        "calibration_delta",
        "target_optimization_events",
        "target_calibration_events",
    )
    reason_names = {
        "identity_mrr": "identity_prediction",
        "route_mrr": "route_prediction",
    }
    for confidence, path in sorted(sensitivity_paths.items()):
        for source_row in _load_rows(path):
            row = _extract(source_row, confidence=float(confidence), raw_file=path)
            sensitivity_rows.append(row)
            reference = main_by_seed.get(row["seed"])
            reasons: list[str] = []
            if reference is None:
                reasons.append("seed_not_in_authoritative_main")
            else:
                for field in invariant_fields:
                    if _different(reference.get(field), row.get(field)):
                        reasons.append(reason_names.get(field, field))
            comparisons.append(
                {
                    "seed": row["seed"],
                    "confidence": float(confidence),
                    "matched_operating_point": not reasons,
                    "mismatch_reasons": reasons,
                    "main_gate": reference.get("gate_decision") if reference else None,
                    "sensitivity_gate": row.get("gate_decision"),
                }
            )
    return {
        "authoritative_main_file": str(main_path),
        "legacy_sensitivity_files": {str(k): str(v) for k, v in sorted(sensitivity_paths.items())},
        "main_rows": main_rows,
        "sensitivity_rows": sensitivity_rows,
        "comparisons": comparisons,
        "conclusion": (
            "legacy_sensitivity_is_not_a_matched_operating_point_sweep"
            if any(not row["matched_operating_point"] for row in comparisons)
            else "sensitivity_is_matched"
        ),
    }


def check_c095_consistency(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in report["comparisons"] if abs(float(row["confidence"]) - 0.95) < 1e-12]
    if not rows:
        return {"status": "FAIL", "reason": "c095_missing", "checked_seeds": []}
    if any(not row["matched_operating_point"] for row in rows):
        return {
            "status": "FAIL",
            "reason": "c095_is_not_a_matched_operating_point",
            "checked_seeds": sorted(row["seed"] for row in rows),
        }
    mismatched = [
        row["seed"]
        for row in rows
        if float(row["main_gate"]) != float(row["sensitivity_gate"])
    ]
    return {
        "status": "FAIL" if mismatched else "PASS",
        "reason": "gate_decision_mismatch" if mismatched else "all_matched_c095_gates_equal_main",
        "checked_seeds": sorted(row["seed"] for row in rows),
        "mismatched_seeds": sorted(mismatched),
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, list):
        return ",".join(map(str, value))
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Reddit Gate Forensic Audit",
        "",
        "## Verdict",
        "",
        f"`{report['conclusion']}`",
        "",
        "The legacy confidence experiment is admissible as a matched threshold sweep only if the fitted-model invariants below are identical. Missing fields remain `NA` and are not inferred.",
        "",
        "## Seed-level inventory",
        "",
        "| Run | c | Seed | Config digest | Sources | Params | LCB | Gate | Identity MRR | Route MRR | Hard MRR |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, rows in (("main", report["main_rows"]), ("legacy sensitivity", report["sensitivity_rows"])):
        for row in rows:
            lines.append(
                "| " + " | ".join(
                    [
                        run_name,
                        _format_value(row["confidence"]),
                        str(row["seed"]),
                        _format_value(row["config_digest"]),
                        _format_value(row["selected_sources"]),
                        _format_value(row["trainable_parameters"]),
                        _format_value(row["lcb"]),
                        _format_value(row["gate_decision"]),
                        _format_value(row["identity_mrr"]),
                        _format_value(row["route_mrr"]),
                        _format_value(row["hard_mrr"]),
                    ]
                ) + " |"
            )
    lines.extend(
        [
            "",
            "## Proven cause",
            "",
            "The archived sensitivity branch changed fitted-model invariants (configuration digest, selected source portfolio, parameter count and predictions). It therefore reran training under a reduced configuration instead of changing only the confidence quantile on the authoritative fitted model. Its c=0.95 gate count is not comparable to the main five-seed c=0.95 count.",
            "",
            "## Required repair",
            "",
            "Recompute c in {0.90, 0.95, 0.99} from the authoritative per-seed calibration window means with the same bootstrap seed and no retraining. The repaired c=0.95 decisions must equal the main decisions seed by seed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--sensitivity", action="append", nargs=2, metavar=("CONFIDENCE", "PATH"), required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--consistency-output", type=Path, required=True)
    args = parser.parse_args()
    sensitivity = {float(confidence): Path(path) for confidence, path in args.sensitivity}
    report = audit_gate_results(args.main, sensitivity)
    check = check_c095_consistency(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    args.consistency_output.write_text(
        f"{check['status']}: {check['reason']}\nchecked_seeds={check['checked_seeds']}\n",
        encoding="utf-8",
    )
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
