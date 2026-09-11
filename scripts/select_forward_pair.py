#!/usr/bin/env python3
"""Select a singleton or anchor pair from validated proxy scores."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forward_pair.artifacts import write_selection_artifacts
from forward_pair.selector import CandidateScore, select_forward_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--margin", type=float, default=0.002)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_scores(path: str | Path) -> list[CandidateScore]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        CandidateScore(
            config=row["config"],
            sources=tuple(row["sources"].split("+")),
            proxy_ap=float(row["proxy_ap"]),
            proxy_auc=float(row["proxy_auc"]),
            seed=int(row["seed"]),
            target=row["target"],
            config_digest=row["config_digest"],
        )
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    scores = _read_scores(args.scores)
    result = select_forward_pair(scores, tuple(args.sources), args.margin)
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
    paths = write_selection_artifacts(
        args.output_dir,
        scores,
        result,
        source_commits={
            "TempoFound": provenance["tempofound_commit"],
            "Forward-Pair": provenance["forward_pair_commit"],
        },
    )
    print(paths.selection_json)


if __name__ == "__main__":
    main()
