from pathlib import Path

from scripts.check_package import delivery_paths, find_forbidden_paths


def test_boundary_checker_rejects_data_checkpoints_and_archives() -> None:
    paths = [
        Path("README.md"),
        Path("processed/ml_mooc.csv"),
        Path("saved_models/model.pth"),
        Path("logs/train.log"),
        Path("nested/source_bundle.zip"),
    ]

    problems = find_forbidden_paths(paths)

    assert problems == [
        "logs/train.log",
        "nested/source_bundle.zip",
        "processed/ml_mooc.csv",
        "saved_models/model.pth",
    ]


def test_curated_result_ledgers_are_delivery_safe() -> None:
    assert find_forbidden_paths([
        Path("results/raw/seed0.json"),
        Path("results/aggregated/summary.csv"),
        Path("figures/main.pdf"),
    ]) == []


def test_tracked_repository_files_are_archive_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        packaged = delivery_paths(root)
    else:
        packaged = [
            Path(line.split("  ", maxsplit=1)[1])
            for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
            if line
        ]

    assert find_forbidden_paths(packaged) == []
