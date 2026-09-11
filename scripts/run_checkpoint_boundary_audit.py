#!/usr/bin/env python3
"""Run validation, fixed-epoch, and prefix-only transfer checkpoints.

The script never substitutes a missing checkpoint or result.  Missing inputs
are recorded as ``PENDING`` in the manifest so that an audit cannot silently
turn an incomplete run into a reported result.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempofound.config import ExperimentConfig


def checkpoint_path(template: str, *, target: str, seed: int) -> Path:
    return Path(template.format(target=target, seed=seed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-template", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("validation", "fixed_epoch", "prefix_only"),
        default=("validation", "fixed_epoch", "prefix_only"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transfer-mode", choices=("frozen", "full", "scratch"), default="frozen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from tempofound.transfer import run_transfer

    base = ExperimentConfig.from_yaml(args.config)
    targets = tuple(args.targets or base.targets)
    seeds = tuple(args.seeds or base.seeds)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for target in targets:
        for seed in seeds:
            checkpoint = checkpoint_path(args.checkpoint_template, target=target, seed=seed)
            for mode in args.modes:
                output = output_root / mode / target / f"seed{seed}.json"
                record: dict[str, object] = {
                    "target": target,
                    "seed": seed,
                    "checkpoint_selection_mode": mode,
                    "checkpoint": str(checkpoint),
                    "output": str(output),
                }
                if not checkpoint.is_file():
                    record["status"] = "PENDING_MISSING_CHECKPOINT"
                    records.append(record)
                    continue
                config = replace(base, checkpoint_selection_mode=mode)
                try:
                    result = run_transfer(
                        config,
                        checkpoint_path=checkpoint,
                        target=target,
                        seed=seed,
                        processed_dir=args.processed_dir,
                        mode=args.transfer_mode,
                        device=args.device,
                        evaluation_protocol=config.evaluation_protocol,
                        negative_policy=config.negative_policy,
                        negatives_per_positive=config.negatives_per_positive,
                        evaluation_batch_size=config.evaluation_batch_size,
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    record["status"] = "COMPLETED"
                except Exception as exc:  # keep all failures auditable and continue other cells
                    record["status"] = "FAILED"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
    manifest = output_root / "checkpoint_boundary_manifest.json"
    manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
