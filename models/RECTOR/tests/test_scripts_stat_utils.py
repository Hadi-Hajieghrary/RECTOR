import importlib.util
from pathlib import Path

import numpy as np


SCRIPTS_ROOT = Path(__file__).parent.parent / "scripts"


def _load_module(module_name: str, rel_path: str):
    path = SCRIPTS_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bootstrap_ci_constant_data_corner_case():
    mod = _load_module("compute_bootstrap_cis", "evaluation/compute_bootstrap_cis.py")

    data = np.array([5.0] * 64)
    point, lo, hi = mod.bootstrap_ci(data, n_bootstrap=300, ci=0.95)

    assert point == 5.0
    assert lo == 5.0
    assert hi == 5.0


def test_bootstrap_ci_bounds_for_binary_data():
    mod = _load_module("compute_bootstrap_cis", "evaluation/compute_bootstrap_cis.py")

    data = np.array([0] * 80 + [100] * 20, dtype=np.float32)
    point, lo, hi = mod.bootstrap_ci(data, n_bootstrap=400, ci=0.95)

    assert 0.0 <= lo <= point <= hi <= 100.0


def test_paired_wilcoxon_small_nonzero_corner_case():
    mod = _load_module("compute_bootstrap_cis", "evaluation/compute_bootstrap_cis.py")

    a = np.array([1, 1, 1, 1, 1])
    b = np.array([1, 1, 1, 1, 2])
    result = mod.paired_wilcoxon(a, b, "a", "b")

    assert "note" in result
    assert result["n_nonzero"] < 10


def test_grid_search_weights_common_usage_shape_and_keys():
    mod = _load_module("tune_weighted_sum", "tuning/tune_weighted_sum.py")

    n, m = 20, 6
    rng = np.random.default_rng(7)
    tier_scores = rng.uniform(low=0.0, high=1.0, size=(n, m, 4)).astype(np.float32)

    # Make mode 0 consistently better for S+L to create a stable optimum.
    tier_scores[:, 0, 0] *= 0.05
    tier_scores[:, 0, 1] *= 0.05

    result = mod.grid_search_weights(
        tier_scores_all=tier_scores,
        gt_positions=None,
        pred_positions=None,
        tune_fraction=0.25,
        seed=42,
    )

    assert "best_weights" in result
    assert len(result["best_weights"]) == 4
    assert "tuned" in result and "default" in result
    assert "SL_pct" in result["tuned"]
    assert 0.0 <= result["tuned"]["SL_pct"] <= 100.0
