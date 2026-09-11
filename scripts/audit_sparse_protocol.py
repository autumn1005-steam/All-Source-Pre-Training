"""Audit admissible-query and negative-sampling counters in result JSONs.

The auditor never reconstructs missing counters from MRR or from a paper
table.  Missing fields are emitted as ``NA`` so a sparse-domain protocol
difference remains visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COUNTERS = (
    "total_queries",
    "valid_queries",
    "skipped_queries",
    "fallback_count",
    "collision_rejection_count",
)


def _first(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return "NA"


def summarize_result_payload(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Return a flat, lossless audit row for one result payload."""

    total = _first(payload, "total_queries", "num_queries", "queries")
    valid = _first(payload, "valid_queries", "num_valid_queries", "num_positives")
    skipped = _first(payload, "skipped_queries", "num_skipped_queries")
    fallback = _first(payload, "fallback_count", "negative_fallback_count")
    collisions = _first(
        payload, "collision_rejection_count", "collision_rejections", "num_collision_rejections"
    )
    row: dict[str, Any] = {
        "source": source,
        "target": payload.get("target", "NA"),
        "seed": payload.get("seed", "NA"),
        "evaluation_protocol": payload.get("evaluation_protocol", "NA"),
        "negative_policy": payload.get("negative_policy", "NA"),
        "test_mrr": payload.get("test_mrr", "NA"),
        "total_queries": total,
        "valid_queries": valid,
        "skipped_queries": skipped,
        "fallback_count": fallback,
        "collision_rejection_count": collisions,
    }
    if isinstance(total, (int, float)) and total:
        row["skip_rate"] = (
            float(skipped) / float(total)
            if isinstance(skipped, (int, float))
            else "NA"
        )
    else:
        row["skip_rate"] = "NA"
    return row


def audit_root(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            rows.append(summarize_result_payload(payload, source=str(path)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = audit_root(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"root": str(args.root), "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
