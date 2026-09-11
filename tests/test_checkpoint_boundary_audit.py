from pathlib import Path

from scripts.run_checkpoint_boundary_audit import checkpoint_path


def test_checkpoint_template_is_explicitly_bound_to_target_and_seed() -> None:
    got = checkpoint_path(
        "/runs/{target}/seed{seed}/checkpoint.pt", target="reddit", seed=42
    )
    assert got == Path("/runs/reddit/seed42/checkpoint.pt")
