#!/usr/bin/env python3
"""Test loading tf_example data with WaymoTFExampleDataset."""

import sys
from pathlib import Path

# Add data scripts to path
sys.path.insert(0, str(Path(__file__).parent / "data" / "scripts" / "lib"))

from waymo_dataset import WaymoTFExampleDataset, build_tfexample_dataloader


def test_split(data_root: str, split: str, max_scenarios: int = 5):
    """Test loading a specific split."""
    print(f"\n{'='*60}")
    print(f"Testing split: {split}")
    print(f"{'='*60}")
    
    # Construct full path to split directory
    split_dir = Path(data_root) / split
    
    try:
        # Create dataset
        dataset = WaymoTFExampleDataset(
            data_dir=str(split_dir),
            split=split,  # This will be stored but not used for pathing
            history_frames=11,
            short_horizon_frames=80,
            long_horizon_frames=200,
            max_scenarios=max_scenarios,
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Total scenarios: {len(dataset)}")
        
        # Load a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n✓ Sample loaded successfully")
            print(f"  Scenario ID: {sample['scenario_id']}")
            print(f"  Agent trajectories shape: {sample['agent_trajectories'].shape}")
            print(f"  Agent valid shape: {sample['agent_valid'].shape}")
            print(f"  Roadgraph shape: {sample['roadgraph_xyz'].shape}")
            print(f"  Interactive pairs shape: {sample['interactive_pairs'].shape}")
            
            # Check for valid agents
            num_agents, num_frames = sample['agent_valid'].shape
            valid_agents = sample['agent_valid'][:, :11].any(dim=1)
            num_valid = valid_agents.sum().item()
            print(f"  Valid agents in history: {num_valid}/{num_agents}")
            print(f"  Total frames: {num_frames}")
            print(f"  Interactive pairs found: {len(sample['interactive_pairs'])}")
            
            # Test dataloader
            print(f"\n✓ Testing dataloader...")
            dataloader = build_tfexample_dataloader(
                data_dir=str(split_dir),
                split=split,
                batch_size=2,
                num_workers=0,  # Use 0 for testing
                shuffle=False,
                max_scenarios=max_scenarios,
            )
            
            batch = next(iter(dataloader))
            print(f"  Batch size: {len(batch['scenario_id'])}")
            print(f"  Batch agent_trajectories shape: {batch['agent_trajectories'].shape}")
            
            print(f"\n✅ Split '{split}' test PASSED\n")
            return True
        else:
            print(f"⚠️  Dataset is empty")
            return False
            
    except Exception as e:
        print(f"\n❌ Split '{split}' test FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all interactive splits."""
    data_root = "/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example"
    
    splits_to_test = [
        "testing_interactive",
        "training_interactive", 
        "validation_interactive",
    ]
    
    results = {}
    for split in splits_to_test:
        results[split] = test_split(data_root, split, max_scenarios=10)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for split, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{split:30s} {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
