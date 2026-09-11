#!/usr/bin/env python3
"""Generate or analyze source subsets for manuscript control experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forward_pair.controls import (
    exhaustive_subsets,
    exact_random_subsets,
    standard_greedy_from_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("exhaustive", "random7", "standard-greedy"), required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources = tuple(args.sources)
    if args.method == "exhaustive":
        candidates = exhaustive_subsets(sources)
        payload = {"method": args.method, "candidates": [list(item) for item in candidates]}
    elif args.method == "random7":
        candidates = exact_random_subsets(sources, count=7, seed=args.seed)
        payload = {"method": args.method, "seed": args.seed, "candidates": [list(item) for item in candidates]}
    else:
        if not args.scores_json:
            raise ValueError("--scores-json is required for standard-greedy")
        raw = json.loads(Path(args.scores_json).read_text(encoding="utf-8"))
        scores = {tuple(key.split("+")): float(value) for key, value in raw.items()}
        selected = standard_greedy_from_scores(scores, sources)
        payload = {"method": args.method, "selected_sources": list(selected)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
