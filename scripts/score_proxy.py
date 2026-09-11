#!/usr/bin/env python3
"""Score a checkpoint using only a target's chronological training period."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forward_pair.proxy import score_checkpoint_proxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--probe-batches", type=int, default=12)
    parser.add_argument("--calibration-fraction", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_checkpoint_proxy(
        args.checkpoint,
        target=args.target,
        processed_dir=args.processed_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        calibration_steps=args.calibration_steps,
        probe_batches=args.probe_batches,
        calibration_fraction=args.calibration_fraction,
        device=args.device,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
