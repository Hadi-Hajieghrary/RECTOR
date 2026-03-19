#!/usr/bin/env python3
"""
Validate Waymax installation and WOMD data compatibility.

Run this script after container setup to confirm that all components
required for the RECTOR closed-loop demonstration plan are operational.

Usage:
    python .devcontainer/scripts/validate-waymax.py
    # or via alias:
    rector-validate-waymax
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

errors: list[str] = []
warnings: list[str] = []


def check(label: str, passed: bool, detail: str = "", warn_only: bool = False):
    if passed:
        print(f"  {PASS} {label}" + (f"  ({detail})" if detail else ""))
    elif warn_only:
        print(f"  {WARN} {label}" + (f"  ({detail})" if detail else ""))
        warnings.append(label)
    else:
        print(f"  {FAIL} {label}" + (f"  ({detail})" if detail else ""))
        errors.append(label)


# ── 1. Core imports ─────────────────────────────────────────────────
print("\n1. Core Imports")
try:
    import numpy as np

    check("numpy", True, np.__version__)
except ImportError as e:
    check("numpy", False, str(e))

try:
    import jax
    import jax.numpy as jnp

    check("jax", True, jax.__version__)
except ImportError as e:
    check("jax", False, str(e))

try:
    import torch

    check("torch", True, torch.__version__)
except ImportError as e:
    check("torch", False, str(e))

try:
    import tensorflow as tf

    check("tensorflow", True, tf.__version__)
except ImportError as e:
    check("tensorflow", False, str(e))

# ── 2. JAX GPU backend ─────────────────────────────────────────────
print("\n2. JAX GPU Backend")
try:
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    check("JAX sees GPU", len(gpu_devices) > 0, f"{len(gpu_devices)} GPU(s)")
    if gpu_devices:
        # Quick compute test
        x = jnp.ones((100, 100), device=gpu_devices[0])
        y = jnp.dot(x, x)
        check("JAX GPU compute", float(y[0, 0]) == 100.0)
except Exception as e:
    check("JAX GPU", False, str(e))

# ── 3. PyTorch GPU (RECTOR inference) ──────────────────────────────
print("\n3. PyTorch GPU")
try:
    check("PyTorch CUDA", torch.cuda.is_available(), torch.version.cuda or "")
    if torch.cuda.is_available():
        t = torch.randn(100, 100, device="cuda")
        check("PyTorch GPU compute", True, torch.cuda.get_device_name(0))
except Exception as e:
    check("PyTorch GPU", False, str(e))

# ── 4. GPU Memory Budget ───────────────────────────────────────────
print("\n4. GPU Memory Budget")
try:
    preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    mem_frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "not set")
    check(
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        preallocate.lower() == "false",
        f"current: {preallocate}",
    )
    check(
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        mem_frac != "not set",
        f"current: {mem_frac}",
    )
except Exception as e:
    check("GPU memory env vars", False, str(e))

# ── 5. Waymax ───────────────────────────────────────────────────────
print("\n5. Waymax Simulator")
try:
    from waymax import config as waymax_config
    from waymax import dataloader, dynamics, env

    check("waymax import", True)
except ImportError as e:
    check("waymax import", False, str(e))

try:
    from waymax import agents

    # Check that IDMRoutePolicy is available
    check(
        "IDMRoutePolicy available",
        hasattr(agents, "IDMRoutePolicy") or hasattr(agents, "create_idm_actor"),
    )
except Exception as e:
    check("Waymax agents", False, str(e), warn_only=True)

# ── 6. Waymo Open Dataset ──────────────────────────────────────────
print("\n6. Waymo Open Dataset")
try:
    import waymo_open_dataset

    check("waymo_open_dataset", True)
except ImportError as e:
    check("waymo_open_dataset", False, str(e))

# ── 7. WOMD Data Paths ─────────────────────────────────────────────
print("\n7. WOMD Data Paths")
data_root = os.environ.get(
    "WAYMO_DATA_ROOT",
    "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0",
)
check("WAYMO_DATA_ROOT set", bool(data_root), data_root)
check("Data root exists", Path(data_root).is_dir(), data_root, warn_only=True)

val_interactive = (
    Path(data_root) / "processed" / "augmented" / "scenario" / "validation_interactive"
)
tfrecord_glob = (
    list(val_interactive.glob("*.tfrecord*")) if val_interactive.is_dir() else []
)
check(
    "validation_interactive data",
    len(tfrecord_glob) > 0,
    f"{len(tfrecord_glob)} files in {val_interactive}",
    warn_only=True,
)

# ── 8. Closed-loop output dirs ─────────────────────────────────────
print("\n8. Closed-Loop Output Directories")
for subdir in ["figures", "videos", "traces", "mining", "ws_tuning"]:
    p = Path("/workspace/output/closedloop") / subdir
    check(f"output/closedloop/{subdir}/", p.is_dir(), warn_only=True)

# ── 9. Experiment dependencies ──────────────────────────────────────
print("\n9. Experiment Dependencies")
for pkg, import_name in [
    ("pandas", "pandas"),
    ("statsmodels", "statsmodels"),
    ("hydra-core", "hydra"),
    ("omegaconf", "omegaconf"),
    ("plotly", "plotly"),
    ("imageio", "imageio"),
    ("scipy", "scipy"),
    ("shapely", "shapely"),
]:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "ok")
        check(pkg, True, ver)
    except ImportError:
        check(pkg, False, "not installed", warn_only=True)

# ── 10. RECTOR source availability ──────────────────────────────────
print("\n10. RECTOR Source Code")
rector_paths = [
    "models/RECTOR/scripts/models/rule_aware_generator_v2.py",
    "models/RECTOR/scripts/proxies/aggregator.py",
]
for rp in rector_paths:
    p = Path("/workspace") / rp
    check(rp, p.is_file(), warn_only=True)

check(
    "simulation_engine package",
    (Path("/workspace/scripts/simulation_engine/__init__.py")).is_file(),
    warn_only=True,
)

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"{FAIL} {len(errors)} error(s), {len(warnings)} warning(s)")
    print("  Errors (must fix):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
elif warnings:
    print(f"{WARN} 0 errors, {len(warnings)} warning(s) — check items above")
    sys.exit(0)
else:
    print(f"{PASS} All checks passed — ready for closed-loop experiments")
    sys.exit(0)
