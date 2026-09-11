#!/usr/bin/env python3
"""Extract measured efficiency fields without imputing unavailable costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "method", "target", "seed", "elapsed_seconds", "trainable_parameters",
    "source_expert_parameters", "peak_gpu_memory_mb", "gpu_hours",
    "calibration_seconds", "inference_seconds",
)


def profile_root(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, (dict, list)):
            continue
        payloads = payload if isinstance(payload, list) else [payload]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            diagnostics = item.get("training_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics:
                for method, values in diagnostics.items():
                    if not isinstance(values, dict):
                        continue
                    merged = dict(item)
                    merged.update(values)
                    merged["method"] = method
                    row = {field: merged.get(field, "NA") for field in FIELDS}
                    row["source"] = str(path)
                    rows.append(row)
            else:
                row = {field: item.get(field, "NA") for field in FIELDS}
                row["source"] = str(path)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = profile_root(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
