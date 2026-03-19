import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).parent.parent / "scripts"

ENTRYPOINTS = [
    "evaluation/evaluate_canonical.py",
    "evaluation/compute_bootstrap_cis.py",
    "visualization/generate_m2i_movies.py",
    "visualization/generate_movies.py",
    "visualization/generate_receding_movies.py",
    "visualization/visualize_predictions.py",
    "visualization/visualize_model_architecture.py",
    "inference/run_inference_demo.py",
    "training/train_rector.py",
    "tuning/tune_weighted_sum.py",
]


@pytest.mark.parametrize("rel_path", ENTRYPOINTS)
def test_entrypoint_help_runs(rel_path):
    script_path = SCRIPTS_ROOT / rel_path
    assert script_path.exists(), f"Missing entrypoint: {script_path}"

    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(SCRIPTS_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    if proc.returncode != 0:
        pytest.fail(
            f"--help failed for {rel_path}\n"
            f"STDOUT:\n{proc.stdout[:2000]}\n"
            f"STDERR:\n{proc.stderr[:2000]}"
        )

    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert "usage" in combined or "help" in combined
