import subprocess
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_all_experiment_launchers_have_valid_bash_syntax() -> None:
    scripts = sorted((ROOT / "experiments").rglob("*.sh"))
    assert scripts, "experiment launchers are missing"
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable; run this syntax check on Linux/WSL/CI")
    for script in scripts:
        result = subprocess.run(
            [bash, "-n", str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
