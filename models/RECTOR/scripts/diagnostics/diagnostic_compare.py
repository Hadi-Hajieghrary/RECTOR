#!/usr/bin/env python3
"""
Diagnostic: Compare prepare_model_input() vs training pipeline at t=10.
At t=10, these should produce IDENTICAL inputs.
"""
import os, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

import glob
import random
import numpy as np
import torch

from training.train_rector import WaymoDataset, TRAJECTORY_SCALE
from visualization.generate_receding_movies import (
    ScenarioLoader,
    prepare_model_input,
    HISTORY_LENGTH,
)

import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2


def main():
    data_dir = "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive"
    val_files = sorted(glob.glob(os.path.join(data_dir, "*")))

    random.seed(42)
    # Take one file
    test_file = random.choice(val_files[:10])
    print(f"Test file: {test_file}")

    device = torch.device("cpu")

    # ---- Method A: Training pipeline (WaymoDataset) ----
    print("\n=== Method A: Training pipeline (WaymoDataset) ===")
    dataset = WaymoDataset([test_file], is_training=False)

    training_samples = []
    for sample in dataset:
        if sample is not None:
            training_samples.append(sample)
        if len(training_samples) >= 3:
            break

    if not training_samples:
        print("No training samples found!")
        return

    print(f"Got {len(training_samples)} training samples")

    # ---- Method B: Receding pipeline (ScenarioLoader + prepare_model_input) ----
    print(
        "\n=== Method B: Receding pipeline (ScenarioLoader + prepare_model_input) ==="
    )
    loader = ScenarioLoader([test_file])
    print(f"Got {len(loader)} receding scenarios")

    if not loader.scenarios:
        print("No scenarios loaded!")
        return

    # Compare the first scenario at t=10
    scenario = loader.scenarios[0]
    print(f"Scenario ID: {scenario['scenario_id']}")

    receding_input = prepare_model_input(scenario, current_t=10, device=device)

    # Find matching training sample (same scenario)
    # They should be from the same TFRecord file, same scenario
    training_sample = training_samples[0]

    print("\n--- Ego History Comparison ---")
    ego_train = training_sample["ego_history"].numpy()  # [11, 4]
    ego_reced = receding_input["ego_history"][0].numpy()  # [11, 4]

    print(f"Training ego_history shape: {ego_train.shape}")
    print(f"Receding ego_history shape: {ego_reced.shape}")

    print(f"\nTraining ego_history (last 3 frames):")
    for t in range(8, 11):
        print(
            f"  t={t}: x={ego_train[t, 0]:.6f}, y={ego_train[t, 1]:.6f}, h={ego_train[t, 2]:.6f}, v={ego_train[t, 3]:.6f}"
        )

    print(f"\nReceding ego_history at t=10 (last 3 frames):")
    for t in range(8, 11):
        print(
            f"  t={t}: x={ego_reced[t, 0]:.6f}, y={ego_reced[t, 1]:.6f}, h={ego_reced[t, 2]:.6f}, v={ego_reced[t, 3]:.6f}"
        )

    diff = np.abs(ego_train - ego_reced)
    print(f"\nMax absolute diff: {diff.max():.8f}")
    print(f"Mean absolute diff: {diff.mean():.8f}")

    # Check if they match
    if diff.max() < 1e-5:
        print("✓ Ego histories MATCH!")
    else:
        print("✗ Ego histories DO NOT MATCH!")
        print("  Per-column max diff:", diff.max(axis=0))
        # Show specific differences
        for t in range(11):
            if diff[t].max() > 1e-5:
                print(f"  Diff at t={t}: {diff[t]}")

    # Agent states
    print("\n--- Agent States Comparison ---")
    agent_train = training_sample["agent_states"].numpy()  # [32, 11, 4]
    agent_reced = receding_input["agent_states"][0].numpy()  # [32, 11, 4]

    print(f"Training agent_states shape: {agent_train.shape}")
    print(f"Receding agent_states shape: {agent_reced.shape}")

    # Compare non-zero agents
    train_nonzero = (np.abs(agent_train).sum(axis=(1, 2)) > 0).sum()
    reced_nonzero = (np.abs(agent_reced).sum(axis=(1, 2)) > 0).sum()
    print(f"Non-zero agents: training={train_nonzero}, receding={reced_nonzero}")

    agent_diff = np.abs(agent_train - agent_reced)
    print(f"Max agent diff: {agent_diff.max():.8f}")
    if agent_diff.max() < 1e-4:
        print("✓ Agent states MATCH!")
    else:
        print("✗ Agent states DO NOT MATCH!")
        # Find which agents differ
        for a in range(32):
            adiff = np.abs(agent_train[a] - agent_reced[a]).max()
            if adiff > 1e-4:
                print(f"  Agent {a}: max diff = {adiff:.6f}")
                # Show sample
                for t in [0, 5, 10]:
                    if (
                        np.abs(agent_train[a, t]).sum() > 0
                        or np.abs(agent_reced[a, t]).sum() > 0
                    ):
                        print(f"    t={t} train: {agent_train[a, t]}")
                        print(f"    t={t} reced: {agent_reced[a, t]}")

    # Lane centers
    print("\n--- Lane Centers Comparison ---")
    lane_train = training_sample["lane_centers"].numpy()  # [64, 20, 2]
    lane_reced = receding_input["lane_centers"][0].numpy()  # [64, 20, 2]

    train_lane_count = (np.abs(lane_train).sum(axis=(1, 2)) > 0).sum()
    reced_lane_count = (np.abs(lane_reced).sum(axis=(1, 2)) > 0).sum()
    print(f"Non-zero lanes: training={train_lane_count}, receding={reced_lane_count}")

    lane_diff = np.abs(lane_train - lane_reced)
    print(f"Max lane diff: {lane_diff.max():.8f}")
    if lane_diff.max() < 1e-4:
        print("✓ Lane centers MATCH!")
    else:
        print("✗ Lane centers DO NOT MATCH!")
        for l in range(64):
            ldiff = np.abs(lane_train[l] - lane_reced[l]).max()
            if ldiff > 1e-4:
                print(f"  Lane {l}: max diff = {ldiff:.6f}")
                if ldiff > 0.01:
                    print(f"    train first pt: {lane_train[l, 0]}")
                    print(f"    reced first pt: {lane_reced[l, 0]}")

    # ---- Now check with model ----
    print("\n\n=== Model Output Comparison ===")
    from models.rule_aware_generator import RuleAwareGenerator

    checkpoint_path = "/workspace/models/RECTOR/models/best.pt"
    m2i_path = "/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin"

    model = RuleAwareGenerator(
        embed_dim=256,
        decoder_hidden_dim=256,
        decoder_num_layers=4,
        latent_dim=64,
        num_modes=6,
        use_m2i_encoder=True,
        m2i_checkpoint=m2i_path,
        freeze_m2i=True,
        trajectory_length=50,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        # Method A: training pipeline input
        out_train = model(
            ego_history=training_sample["ego_history"].unsqueeze(0),
            agent_states=training_sample["agent_states"].unsqueeze(0),
            lane_centers=training_sample["lane_centers"].unsqueeze(0),
        )

        # Method B: receding pipeline input
        out_reced = model(
            ego_history=receding_input["ego_history"],
            agent_states=receding_input["agent_states"],
            lane_centers=receding_input["lane_centers"],
        )

    traj_train = out_train["trajectories"][0].numpy()  # [6, 50, 4]
    traj_reced = out_reced["trajectories"][0].numpy()  # [6, 50, 4]

    conf_train = out_train["confidence"][0].numpy()
    conf_reced = out_reced["confidence"][0].numpy()

    conf_train_sm = np.exp(conf_train) / np.exp(conf_train).sum()
    conf_reced_sm = np.exp(conf_reced) / np.exp(conf_reced).sum()

    print(f"\nTraining confidence: {conf_train_sm}")
    print(f"Receding confidence: {conf_reced_sm}")

    best_train = conf_train_sm.argmax()
    best_reced = conf_reced_sm.argmax()
    print(f"Best mode: training={best_train}, receding={best_reced}")

    traj_diff = np.abs(traj_train - traj_reced)
    print(f"\nTrajectory max diff: {traj_diff.max():.8f}")
    print(f"Trajectory mean diff: {traj_diff.mean():.8f}")

    # Show best trajectory endpoints (in normalized coords)
    print(f"\nBest traj endpoint (normalized):")
    print(
        f"  Training: x={traj_train[best_train, -1, 0]:.6f}, y={traj_train[best_train, -1, 1]:.6f}"
    )
    print(
        f"  Receding: x={traj_reced[best_reced, -1, 0]:.6f}, y={traj_reced[best_reced, -1, 1]:.6f}"
    )

    # Convert to meters
    print(f"\nBest traj endpoint (meters):")
    print(
        f"  Training: x={traj_train[best_train, -1, 0]*50:.1f}m, y={traj_train[best_train, -1, 1]*50:.1f}m"
    )
    print(
        f"  Receding: x={traj_reced[best_reced, -1, 0]*50:.1f}m, y={traj_reced[best_reced, -1, 1]*50:.1f}m"
    )

    # Also show GT for reference
    if "traj_gt" in training_sample:
        gt = training_sample["traj_gt"].numpy()
        print(f"  GT:       x={gt[-1, 0]*50:.1f}m, y={gt[-1, 1]*50:.1f}m")

    # Check the actual values
    print(f"\nBest trajectory (first 5 timesteps, meters):")
    for t in range(5):
        print(
            f"  t={t}: train=({traj_train[best_train,t,0]*50:.2f}, {traj_train[best_train,t,1]*50:.2f}) "
            f"reced=({traj_reced[best_reced,t,0]*50:.2f}, {traj_reced[best_reced,t,1]*50:.2f})"
        )

    print("\n\nDONE - If inputs/outputs don't match, that's the source of the problem.")


if __name__ == "__main__":
    main()
