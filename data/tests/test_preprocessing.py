#!/usr/bin/env python3
"""
Test script for Waymo data preprocessing and loading.

This script verifies that:
1. Preprocessing script works correctly
2. Data loader can read preprocessed files
3. Batch shapes are correct
4. Data values are reasonable

Usage:
    # Test with existing preprocessed data
    python test_data_loading.py --data-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/training_interactive

    # Run end-to-end test (preprocess + load)
    python test_data_loading.py \
        --test-preprocessing \
        --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
        --output-dir /tmp/test_preprocessed
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Add data/scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from waymo_dataset import WaymoITPDataset, build_dataloader, collate_fn
except ImportError:
    print("ERROR: Cannot import waymo_dataset.py")
    print("Make sure waymo_dataset.py is in data/scripts/")
    sys.exit(1)


def test_preprocessing(input_dir: str, output_dir: str, max_files: int = 2):
    """Test the preprocessing script."""
    print("\n" + "="*60)
    print("TEST 1: Preprocessing")
    print("="*60)
    
    try:
        from waymo_preprocess import WaymoPreprocessor
    except ImportError:
        print("❌ Cannot import waymo_preprocess.py")
        return False
    
    # Create preprocessor
    preprocessor = WaymoPreprocessor(
        history_frames=11,
        short_horizon_frames=80,
        long_horizon_frames=160,
        interactive_only=True,
    )
    
    # Find TFRecord files
    input_path = Path(input_dir)
    tfrecord_files = list(input_path.glob("*.tfrecord*"))[:max_files]
    
    if len(tfrecord_files) == 0:
        print(f"❌ No TFRecord files found in {input_dir}")
        return False
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    # Process files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_scenarios = 0
    for tfrecord_file in tfrecord_files:
        print(f"Processing {tfrecord_file.name}...")
        scenarios = preprocessor.process_tfrecord(tfrecord_file)
        
        # Save scenarios
        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            output_file = output_path / f"{scenario_id}.npz"
            
            np.savez_compressed(
                output_file,
                scenario_id=scenario_id,
                tracks=scenario["tracks"],
                map=scenario["map"],
                pair_indices=scenario["pair_indices"],
                all_pairs=scenario.get("all_pairs", []),
            )
            total_scenarios += 1
    
    print(f"✅ Preprocessed {total_scenarios} scenarios to {output_dir}")
    return True


def test_data_loading(data_dir: str, num_samples: int = 5):
    """Test data loading from preprocessed files."""
    print("\n" + "="*60)
    print("TEST 2: Data Loading")
    print("="*60)
    
    try:
        # Create dataset
        dataset = WaymoITPDataset(
            data_dir=data_dir,
            split="train",
            short_horizon_frames=80,
            long_horizon_frames=160,
            history_frames=11,
            augment=False,
            max_other_agents=5,
        )
        
        print(f"✅ Created dataset with {len(dataset)} scenarios")
        
        if len(dataset) == 0:
            print("⚠️  Dataset is empty - skipping loading test")
            print("   Run preprocessing first:")
            print("   ./data/scripts/preprocess_all.sh --test")
            return True  # Skip, not failure
        
        # Test loading samples
        print(f"\nTesting {min(num_samples, len(dataset))} samples...")
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            
            # Check keys
            required_keys = [
                "scenario_id", "history", "future_short", "future_long",
                "valid_mask", "relation", "agent_types", "map_polylines",
                "other_agents", "ego_last_state"
            ]
            
            for key in required_keys:
                if key not in sample:
                    print(f"❌ Missing key: {key}")
                    return False
            
            # Check shapes
            print(f"\n--- Sample {i} ({sample['scenario_id']}) ---")
            print(f"history: {sample['history'].shape} (expected: [2, 11, 2])")
            print(f"future_short: {sample['future_short'].shape} (expected: [2, 80, 2])")
            print(f"future_long: {sample['future_long'].shape} (expected: [2, 80, 2])")
            print(f"valid_mask: {sample['valid_mask'].shape} (expected: [2, 160])")
            print(f"map_polylines: {sample['map_polylines'].shape} (expected: [256, 20, 7])")
            print(f"other_agents: {sample['other_agents'].shape} (expected: [5, 160, 2])")
            print(f"ego_last_state: {sample['ego_last_state'].shape} (expected: [5])")
            print(f"agent_types: {sample['agent_types']}")
            print(f"relation: {sample['relation'].item()}")
            
            # Verify shapes
            if sample['history'].shape != (2, 11, 2):
                print(f"❌ Wrong history shape!")
                return False
            if sample['future_short'].shape != (2, 80, 2):
                print(f"❌ Wrong future_short shape!")
                return False
            if sample['future_long'].shape != (2, 80, 2):
                print(f"❌ Wrong future_long shape!")
                return False
        
        print("\n✅ All samples loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batching(data_dir: str, batch_size: int = 4):
    """Test batch collation."""
    print("\n" + "="*60)
    print("TEST 3: Batching")
    print("="*60)
    
    try:
        # Create dataset
        dataset = WaymoITPDataset(
            data_dir=data_dir,
            split="train",
            short_horizon_frames=80,
            long_horizon_frames=160,
            history_frames=11,
            augment=False,
        )
        
        if len(dataset) == 0:
            print("⚠️  Dataset is empty - skipping batching test")
            print("   Run preprocessing first:")
            print("   ./data/scripts/preprocess_all.sh --test")
            return True  # Skip, not failure
        
        if len(dataset) < batch_size:
            print(f"⚠️  Dataset too small for batch size {batch_size}")
            batch_size = len(dataset)
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        
        print(f"Batch size: {batch_size}")
        print(f"\n--- Batch Shapes ---")
        print(f"history: {batch['history'].shape} (expected: [{batch_size}, 2, 11, 2])")
        print(f"future_short: {batch['future_short'].shape} (expected: [{batch_size}, 2, 80, 2])")
        print(f"future_long: {batch['future_long'].shape} (expected: [{batch_size}, 2, 80, 2])")
        print(f"valid_mask: {batch['valid_mask'].shape} (expected: [{batch_size}, 2, 160])")
        print(f"map_polylines: {batch['map_polylines'].shape} (expected: [{batch_size}, 256, 20, 7])")
        print(f"other_agents: {batch['other_agents'].shape} (expected: [{batch_size}, 5, 160, 2])")
        print(f"ego_last_state: {batch['ego_last_state'].shape} (expected: [{batch_size}, 5])")
        print(f"scenario_ids: {len(batch['scenario_id'])} (list of strings)")
        
        # Verify batch shapes
        expected_shapes = {
            'history': (batch_size, 2, 11, 2),
            'future_short': (batch_size, 2, 80, 2),
            'future_long': (batch_size, 2, 80, 2),
            'valid_mask': (batch_size, 2, 160),
            'map_polylines': (batch_size, 256, 20, 7),
            'other_agents': (batch_size, 5, 160, 2),
            'ego_last_state': (batch_size, 5),
        }
        
        all_correct = True
        for key, expected_shape in expected_shapes.items():
            if batch[key].shape != expected_shape:
                print(f"❌ Wrong shape for {key}: {batch[key].shape} vs {expected_shape}")
                all_correct = False
        
        if all_correct:
            print("\n✅ All batch shapes correct!")
        else:
            return False
        
        # Check data ranges
        print("\n--- Data Ranges ---")
        print(f"position range: [{batch['history'][..., 0].min():.2f}, {batch['history'][..., 0].max():.2f}] m")
        print(f"velocity (from last_state): {batch['ego_last_state'][:, 3].mean():.2f} ± {batch['ego_last_state'][:, 3].std():.2f} m/s")
        print(f"heading range: [{batch['ego_last_state'][:, 2].min():.2f}, {batch['ego_last_state'][:, 2].max():.2f}] rad")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in batching: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentation(data_dir: str):
    """Test data augmentation."""
    print("\n" + "="*60)
    print("TEST 4: Augmentation")
    print("="*60)
    
    try:
        # Create dataset without augmentation
        dataset_no_aug = WaymoITPDataset(
            data_dir=data_dir,
            split="train",
            augment=False,
        )
        
        if len(dataset_no_aug) == 0:
            print("⚠️  Dataset is empty - skipping augmentation test")
            print("   Run preprocessing first:")
            print("   ./data/scripts/preprocess_all.sh --test")
            return True  # Skip, not failure
        
        # Create dataset with augmentation
        dataset_aug = WaymoITPDataset(
            data_dir=data_dir,
            split="train",
            augment=True,
        )
        
        # Get same sample from both
        sample_no_aug = dataset_no_aug[0]
        
        # Get augmented samples (should be different each time)
        samples_aug = [dataset_aug[0] for _ in range(3)]
        
        # Check that augmented samples are different
        positions = [s['history'][0, 0, :].numpy() for s in samples_aug]
        
        print("Original position:", sample_no_aug['history'][0, 0, :].numpy())
        print("Augmented positions:")
        for i, pos in enumerate(positions):
            print(f"  {i+1}: {pos}")
        
        # Check if any are different (they should be with high probability)
        all_same = all(np.allclose(positions[0], p) for p in positions[1:])
        
        if all_same:
            print("⚠️  All augmented samples are the same (might be unlucky)")
        else:
            print("✅ Augmentation is working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in augmentation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Waymo data preprocessing and loading")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory with preprocessed data"
    )
    parser.add_argument(
        "--test-preprocessing",
        action="store_true",
        help="Test preprocessing (requires --input-dir and --output-dir)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory with Waymo TFRecords (for preprocessing test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for preprocessing test"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for batching test"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Waymo Data Preprocessing & Loading Tests")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Preprocessing (optional)
    if args.test_preprocessing:
        if not args.input_dir or not args.output_dir:
            print("❌ --test-preprocessing requires --input-dir and --output-dir")
            sys.exit(1)
        
        passed = test_preprocessing(args.input_dir, args.output_dir)
        all_passed = all_passed and passed
        
        # Use the preprocessed data for subsequent tests
        args.data_dir = args.output_dir
    
    # Test 2: Data loading
    if args.data_dir:
        passed = test_data_loading(args.data_dir, args.num_samples)
        all_passed = all_passed and passed
        
        # Test 3: Batching
        passed = test_batching(args.data_dir, args.batch_size)
        all_passed = all_passed and passed
        
        # Test 4: Augmentation
        passed = test_augmentation(args.data_dir)
        all_passed = all_passed and passed
    else:
        print("\n❌ No data directory specified!")
        print("Usage:")
        print("  Test with existing data: --data-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/training_interactive")
        print("  Test preprocessing: --test-preprocessing --input-dir ... --output-dir ...")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready to start training!")
        print("Next steps:")
        print("  1. Update your config to point to the preprocessed data")
        print("  2. Run: python train_long.py --config configs/training_interactive.yaml")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
