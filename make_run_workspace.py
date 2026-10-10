"""Copy preserved experiment code into a fresh workspace without archived scores or matrices."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
ARCHIVED = {"results", "positive_extension", "transfer_results", "adaptation_results", "budget_control_results"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    source = (ROOT / "reproduction").resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise SystemExit("Run workspace and archived reproduction tree must be separate")
    if target.exists() and any(target.iterdir()):
        raise SystemExit("Choose a new or empty run directory")
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        if relative.parts[0] in ARCHIVED or "__pycache__" in relative.parts:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, destination)
        count += 1
    (target / "results").mkdir(exist_ok=True)
    (target / "job_logs").mkdir(exist_ok=True)
    print("Created fresh run workspace:", target, "preserved files:", count)


if __name__ == "__main__":
    main()
