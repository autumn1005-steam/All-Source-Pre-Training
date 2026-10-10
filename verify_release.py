"""Verify repository file hashes and syntax; never import or execute training scripts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for relative, expected in manifest["files_sha256"].items():
        path = root / relative
        if not path.is_file():
            errors.append(relative + ": missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(relative + ": hash mismatch")
        elif path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except SyntaxError as error:
                errors.append(relative + ": " + str(error))
    if errors:
        raise SystemExit("Release verification failed:\n" + "\n".join(errors))
    print(json.dumps({"status": "passed", "verified_files": len(manifest["files_sha256"]),
                      "record_counts": manifest["record_counts"]}, indent=2))


if __name__ == "__main__":
    main()
