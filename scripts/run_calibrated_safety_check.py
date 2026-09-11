"""Three-seed out-of-sample target-safety calibration diagnostic.

All variants use historical collision-free negatives, ranking MRR, equal source
pretraining minibatch budgets, and the same chronological target optimization /
calibration split.  The only intended difference among routed variants is the
final target-level gate decision.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forward_pair.pipeline import run_pipeline
from tempofound.config import ExperimentConfig
import torch


VARIANTS = {
    "calib_strict_ranked": dict(source_experts=False, risk=False, gate="force_identity"),
    # One routed training run reports route / identity / hard / soft test MRR
    # under exactly the same fitted parameters, eliminating ablation noise.
    "calib_suite": dict(source_experts=True, risk=True, gate="holdout_lcb_hard"),
    # Kept as optional controls for targeted reruns.
    "calib_route": dict(source_experts=True, risk=True, gate="force_route"),
    "calib_identity": dict(source_experts=True, risk=True, gate="force_identity"),
    "calib_soft": dict(source_experts=True, risk=True, gate="holdout_lcb_soft"),
    "calib_hard": dict(source_experts=True, risk=True, gate="holdout_lcb_hard"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--config", default=str(ROOT / "configs/rasp_real_initial.yaml"))
    parser.add_argument("--gate-confidence", type=float, default=0.95)
    parser.add_argument("--gate-temperature", type=float, default=0.01)
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    output = Path(args.output_root) / args.variant
    output.mkdir(parents=True, exist_ok=True)

    config = replace(
        ExperimentConfig.from_yaml(args.config),
        name=f"oos_safety_{args.variant}_{args.target}",
        targets=(args.target,),
        seeds=tuple(args.seeds),
        consistent_candidate_weights=True,
        temporal_paired_gate=("matched" not in Path(args.config).stem),
        expert_identity_init=bool(spec["source_experts"]),
        proxy_metric="mrr",
        source_experts=bool(spec["source_experts"]),
        risk_controlled_routing=bool(spec["risk"]),
        proxy_negative_policy="historical",
        negative_policy="historical",
        transfer_regret_weight=0.25 if spec["risk"] else 0.0,
        transfer_regret_mode="event_pairwise",
        target_gate_mode=str(spec["gate"]),
        target_gate_windows=5,
        target_gate_bootstrap_replicates=2000,
        target_gate_confidence=args.gate_confidence,
        target_gate_min_gain=0.0,
        target_gate_temperature=args.gate_temperature,
        target_calibration_fraction=0.20,
        target_calibration_min_events=128,
        target_reference_weight=1.0,
        target_gate_negatives=20,
        total_batches_per_epoch=32,
    )

    results = []
    for seed in config.seeds:
        started = time.monotonic()
        print(f"START target={args.target} variant={args.variant} seed={seed}", flush=True)
        result = run_pipeline(
            config,
            target=args.target,
            seed=seed,
            processed_dir=args.processed_dir,
            output_root=output,
            device="cuda:0",
        )
        transfer_payloads = {}
        for label in ("rasp", "all_sources"):
            transfer_path = output / args.target / f"seed{seed}" / "transfer" / f"{label}.json"
            transfer_payloads[label] = json.loads(transfer_path.read_text(encoding="utf-8"))
        result["safety_diagnostics"] = {
            label: {
                key: payload.get(key)
                for key in (
                    "target_gate", "target_gate_lcb", "target_gate_calibration_mrr_delta",
                    "target_gate_routed_calibration_mrr", "target_gate_reference_calibration_mrr",
                    "raw_routed_test_mrr", "router_off_reference_test_mrr", "safety_gate_sweep",
                    "target_optimization_events", "target_calibration_events_reserved",
                )
            }
            for label, payload in transfer_payloads.items()
        }
        if spec["risk"]:
            policy_deltas = {}
            for policy in ("route", "identity", "hard", "soft"):
                rasp_mrr = transfer_payloads["rasp"]["safety_gate_sweep"][policy]["test_mrr"]
                all_mrr = transfer_payloads["all_sources"]["safety_gate_sweep"][policy]["test_mrr"]
                policy_deltas[policy] = rasp_mrr - all_mrr
            result["gate_policy_test_mrr_deltas"] = policy_deltas
        diagnostics = {}
        for label in ("rasp", "all_sources"):
            path = output / args.target / f"seed{seed}" / "final" / label / "pretrained.pth"
            training = torch.load(path, map_location="cpu", weights_only=False)["metadata"]["pretraining"]
            assert training["epoch_batch_counts"] == [32] * config.pretrain_epochs, training
            diagnostics[label] = training
        result["training_diagnostics"] = diagnostics
        result["elapsed_seconds"] = time.monotonic() - started
        results.append(result)
        progress = output / args.target / "paired_results.json"
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"FINISH seed={seed} delta={result['test_mrr_delta']:.6f}", flush=True)

    deltas = [row["test_mrr_delta"] for row in results]
    summary = dict(
        target=args.target,
        variant=args.variant,
        seeds=list(config.seeds),
        protocol="historical collision-free ranking; chronological target-train optimization prefix + held-out calibration suffix",
        budget_scope="equal consumed source minibatches; target optimization event count matched across calibration variants",
        rasp_mean=statistics.mean(row["rasp_test_mrr"] for row in results),
        all_sources_mean=statistics.mean(row["all_sources_test_mrr"] for row in results),
        paired_delta_mean=statistics.mean(deltas),
                    paired_delta_sample_std=(statistics.stdev(deltas) if len(deltas) > 1 else 0.0),
        positive_seeds=sum(delta > 0 for delta in deltas),
    )
    if spec["risk"]:
        summary["gate_policy_delta_means"] = {
            policy: statistics.mean(row["gate_policy_test_mrr_deltas"][policy] for row in results)
            for policy in ("route", "identity", "hard", "soft")
        }
        summary["rasp_gate_values"] = [
            row["safety_diagnostics"]["rasp"]["target_gate"] for row in results
        ]
        summary["rasp_gate_lcbs"] = [
            row["safety_diagnostics"]["rasp"]["target_gate_lcb"] for row in results
        ]
    summary_path = output / args.target / "three_seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
