#!/usr/bin/env python3
"""Reject generated or bulky artifacts from a source delivery."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


FORBIDDEN_DIRECTORIES = {
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "logs",
    "processed",
    # Curated result ledgers are part of the final reproducibility package;
    # bulky checkpoints and raw tensors remain blocked by their own suffixes.
    "saved_checkpoints",
    "saved_models",
    "server_runs",
}
FORBIDDEN_SUFFIXES = {".gz", ".npy", ".npz", ".pt", ".pth", ".tar", ".zip"}


def find_forbidden_paths(paths: Iterable[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        parts = set(path.parts)
        forbidden_directory = bool(parts.intersection(FORBIDDEN_DIRECTORIES))
        egg_info = any(part.endswith(".egg-info") for part in path.parts)
        forbidden_suffix = path.suffix.lower() in FORBIDDEN_SUFFIXES
        if forbidden_directory or egg_info or forbidden_suffix:
            problems.append(path.as_posix())
    return sorted(set(problems))


def delivery_paths(root: Path) -> list[Path]:
    if (root / ".git").exists():
        output = subprocess.check_output(
            ["git", "ls-files"], cwd=root, text=True
        )
        return [Path(line) for line in output.splitlines() if line]
    return [path.relative_to(root) for path in root.rglob("*") if path.is_file()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    problems = find_forbidden_paths(delivery_paths(root))
    if problems:
        print("Forbidden delivery artifacts:")
        print("\n".join(problems))
        raise SystemExit(1)
    print("Package boundary check passed: source, configs, tests, and docs only.")


if __name__ == "__main__":
    main()
