#!/usr/bin/env python3
"""Run the matched-width target-side HRCSR component ablation.

The authoritative source checkpoint is reused for every routed variant.  This
keeps the encoder, source portfolio, source pretraining budget, and source
experts fixed while changing only the target-side component under test.
Existing outputs are skipped so interrupted runs resume without overwriting raw
evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempofound.config import ExperimentConfig


VARIANTS = (
    "a1_expert_identity",
    "a2_response_router",
    "a3_local_acceptance",
    "a4_pairwise_regret",
)


def variant_config(base: ExperimentConfig, variant: str) -> ExperimentConfig:
    common = {
        "source_experts": True,
        "risk_controlled_routing": True,
        "checkpoint_selection_mode": "validation",
        "transfer_regret_mode": "event_pairwise",
        "target_gate_windows": 5,
        "target_gate_bootstrap_replicates": 1000,
        "target_gate_confidence": 0.95,
        "target_gate_min_gain": 0.0,
    }
    if variant == "a1_expert_identity":
        return replace(
            base,
            **common,
            router_local_acceptance=True,
            target_router_training_enabled=False,
            transfer_regret_weight=0.0,
            target_gate_mode="force_identity",
        )
    if variant == "a2_response_router":
        return replace(
            base,
            **common,
            router_local_acceptance=False,
            target_router_training_enabled=True,
            transfer_regret_weight=0.0,
            target_gate_mode="force_route",
        )
    if variant == "a3_local_acceptance":
        return replace(
            base,
            **common,
            router_local_acceptance=True,
            target_router_training_enabled=True,
            transfer_regret_weight=0.0,
            target_gate_mode="force_route",
        )
    if variant == "a4_pairwise_regret":
        return replace(
            base,
            **common,
            router_local_acceptance=True,
            target_router_training_enabled=True,
            transfer_regret_weight=0.25,
            target_gate_mode="force_route",
        )
    raise ValueError(f"unsupported ablation variant: {variant}")


def config_digest(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=list).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-template", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3, 42))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from tempofound.transfer import run_transfer

    base = ExperimentConfig.from_yaml(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for target in args.targets:
        for seed in args.seeds:
            checkpoint = Path(args.checkpoint_template.format(target=target, seed=seed))
            for variant in args.variants:
                output = output_root / variant / target / f"seed{seed}.json"
                record: dict[str, object] = {
                    "target": target,
                    "seed": seed,
                    "variant": variant,
                    "checkpoint": str(checkpoint),
                    "output": str(output),
                }
                if output.is_file():
                    record["status"] = "SKIPPED_EXISTING"
                    records.append(record)
                    continue
                if not checkpoint.is_file():
                    record["status"] = "PENDING_MISSING_CHECKPOINT"
                    records.append(record)
                    continue
                config = variant_config(base, variant)
                record["config_digest"] = config_digest(config)
                try:
                    result = run_transfer(
                        config,
                        checkpoint_path=checkpoint,
                        target=target,
                        seed=seed,
                        processed_dir=args.processed_dir,
                        mode="frozen",
                        device=args.device,
                        evaluation_protocol=config.evaluation_protocol,
                        negative_policy=config.negative_policy,
                        negatives_per_positive=config.negatives_per_positive,
                        evaluation_batch_size=config.evaluation_batch_size,
                    )
                    result["ablation_variant"] = variant
                    result["ablation_config_digest"] = record["config_digest"]
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(
                        json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    record["status"] = "COMPLETED"
                except Exception as exc:
                    record["status"] = "FAILED"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
                manifest = output_root / "component_ablation_manifest.json"
                manifest.write_text(
                    json.dumps(records, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
