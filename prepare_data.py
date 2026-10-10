"""Import exact graph inputs from the supplementary source package or verify a workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-package", type=Path, help="Exact supplementary TriView_HCL_source_package.zip")
    mode.add_argument("--verify", action="store_true", help="Verify already placed graph inputs")
    parser.add_argument("--workspace", type=Path, default=ROOT / "reproduction")
    parser.add_argument("--overwrite", action="store_true", help="Replace a differing local data file when importing")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))["files"]
    failures = []
    if args.source_package:
        with zipfile.ZipFile(args.source_package) as archive:
            # Verify all inputs before writing any file. Extract only exact named
            # inputs; no archive-directed paths or pickle contents are executed.
            names = set(archive.namelist())
            for relative, expected in manifest.items():
                member = expected["archive_member"]
                if member not in names:
                    failures.append(relative + ": missing in source package")
                elif digest(archive.read(member)) != expected["sha256"]:
                    failures.append(relative + ": source package hash mismatch")
                destination = workspace / relative
                if not destination.resolve().is_relative_to(workspace):
                    raise ValueError("Destination escapes the workspace")
                if destination.exists() and not args.overwrite and digest(destination.read_bytes()) != expected["sha256"]:
                    failures.append(relative + ": differing local file; use an empty workspace or --overwrite")
            if failures:
                raise SystemExit("Data import stopped:\n" + "\n".join(failures))
            for relative, expected in manifest.items():
                destination = workspace / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() or digest(destination.read_bytes()) != expected["sha256"]:
                    destination.write_bytes(archive.read(expected["archive_member"]))
    for relative, expected in manifest.items():
        destination = workspace / relative
        if not destination.is_file():
            failures.append(relative + ": missing")
        elif digest(destination.read_bytes()) != expected["sha256"]:
            failures.append(relative + ": hash mismatch")
    if failures:
        raise SystemExit("Data verification failed:\n" + "\n".join(failures))
    print(json.dumps({"status": "passed", "verified_files": len(manifest), "workspace": str(workspace)}, indent=2))


if __name__ == "__main__":
    main()
