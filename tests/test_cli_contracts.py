import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pretrain_cli_defaults_to_paper_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pretrain.py"),
            "--config",
            str(ROOT / "configs/smoke.yaml"),
            "--print-config",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["time_encoding_mode"] == "paper_sincos"
    assert payload["total_batches_per_epoch"] == 4


def test_all_stable_clis_have_working_help() -> None:
    for script in (
        "pretrain.py",
        "score_proxy.py",
        "select_forward_pair.py",
        "transfer.py",
        "run_pipeline.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
        assert "usage:" in result.stdout.lower()


def test_rasp_launchers_reference_only_rasp_configs() -> None:
    for launcher, config in (
        ("run_rasp_smoke.sh", "rasp_smoke.yaml"),
        ("run_rasp_k4.sh", "rasp_k4.yaml"),
    ):
        contents = (ROOT / "experiments" / launcher).read_text(encoding="utf-8")
        assert config in contents
        assert "run_pipeline.py" in contents
        assert "forward_pair" not in contents.lower()
