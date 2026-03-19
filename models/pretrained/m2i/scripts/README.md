# Pretrained M2I Support Scripts

This directory contains helper scripts associated with the pretrained M2I assets used by the RECTOR workspace. These scripts are primarily useful when inspecting, adapting, or visualizing the external M2I models that serve as a backbone for parts of the repository.

## Role in the project

The pretrained M2I code is an upstream dependency rather than the main scientific contribution of this repository. In RECTOR, it is mainly used for:

- scene encoding,
- candidate trajectory generation,
- and comparison against or integration with rule-aware selection.

## What the scripts are used for

The scripts in this directory generally support one of the following tasks:

- running M2I inference on WOMD-like inputs,
- generating qualitative videos,
- exploring receding-horizon behavior,
- or adapting tensor formats between M2I and RECTOR utilities.

## Recommendation for paper writing

For manuscript purposes, this directory should usually be described as the implementation support for the pretrained M2I backbone. The main methodological discussion should remain centered on the RECTOR-specific code in [models/RECTOR](../../RECTOR).

## Related documentation

- [../../README.md](../../README.md) — model inventory for the workspace.
- [../../RECTOR/README.md](../../RECTOR/README.md) — main RECTOR method summary.
