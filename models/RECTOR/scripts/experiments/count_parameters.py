#!/usr/bin/env python3
"""
Count model parameters from the trained RECTOR checkpoint.
This generates REAL efficiency data for the paper.
"""

import sys
import json
from pathlib import Path

# Add model paths
sys.path.insert(0, "/workspace/models/RECTOR/scripts")
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

import torch


def count_parameters():
    """Load model and count parameters."""

    checkpoint_path = Path("/workspace/models/RECTOR/models/best.pt")
    output_path = Path(
        "/workspace/models/RECTOR/output/artifacts/real_param_counts.json"
    )

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"Found {len(state_dict)} parameter tensors")

    # Count parameters by component
    components = {
        "scene_encoder": 0,
        "applicability_head": 0,
        "trajectory_head": 0,
        "tiered_scorer": 0,
        "rule_proxies": 0,
        "other": 0,
    }

    total_params = 0
    trainable_params = 0

    for name, param in state_dict.items():
        num_params = param.numel()
        total_params += num_params

        # Categorize by component
        if "scene_encoder" in name or "m2i" in name.lower():
            components["scene_encoder"] += num_params
        elif "applicability" in name:
            components["applicability_head"] += num_params
        elif "trajectory" in name or "cvae" in name or "decoder" in name:
            components["trajectory_head"] += num_params
        elif "scorer" in name or "tiered" in name:
            components["tiered_scorer"] += num_params
        elif "proxy" in name or "rule" in name:
            components["rule_proxies"] += num_params
        else:
            components["other"] += num_params

    # Format results
    results = {
        "checkpoint": str(checkpoint_path),
        "total_parameters": total_params,
        "total_parameters_millions": round(total_params / 1e6, 2),
        "components": {k: v for k, v in components.items() if v > 0},
        "components_millions": {
            k: round(v / 1e6, 2) for k, v in components.items() if v > 0
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL PARAMETER COUNT (REAL DATA)")
    print("=" * 60)
    print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print("\nBy Component:")
    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / total_params
            print(f"  {comp:25s}: {count:>12,} ({pct:5.1f}%)")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    try:
        results = count_parameters()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
