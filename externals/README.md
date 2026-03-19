# External Dependencies

This directory hosts third-party source trees that are cloned during devcontainer setup and are **not tracked by git** (excluded via `.gitignore`). Only this `README.md` is version-controlled.

## Expected Layout

| Directory | Source | Purpose |
|-----------|--------|---------|
| `M2I/` | [github.com/Tsinghua-MARS-Lab/M2I](https://github.com/Tsinghua-MARS-Lab/M2I) | DenseTNT/M2I backbone and utilities |
| `waymo-open-dataset/` | [github.com/waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) | Waymo proto definitions and data tooling |

## Setup

These repositories are cloned automatically when you open this project in the provided devcontainer:

```bash
.devcontainer/scripts/setup-externals.sh
```

If you are working outside the devcontainer, run that script manually (or clone the repositories above into the corresponding subdirectories) before executing any scripts that depend on them.
