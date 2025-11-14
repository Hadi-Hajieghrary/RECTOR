#!/usr/bin/env python3
"""
Filter Interactive Training Data from Waymo Dataset

This script extracts interactive scenarios from the Waymo training dataset.
Only validation/testing splits have pre-labeled interactive scenarios,
so this script identifies interactive pairs from the training split.

Usage:
    python filter_interactive_training.py \\
        --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training \\
        --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training_interactive \\
        --type v2v

    # Or filter all interactive types:
    python filter_interactive_training.py \\
        --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training \\
        --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training_interactive

Based on M2I repository: https://github.com/Tsinghua-MARS-Lab/M2I
"""

import argparse
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

# Road graph features
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# State features of agents
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

# Traffic light features
traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

# Combine all features
features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)
features_description['scenario/id'] = tf.io.FixedLenFeature([1], tf.string, default_value=None)
features_description['state/objects_of_interest'] = tf.io.FixedLenFeature([128], tf.int64, default_value=None)

# Agent type mappings: vehicle=1, pedestrian=2, cyclist=3
OBJECT_TYPE_DICT = {
    "v2v": [1, 1],  # vehicle-vehicle
    "v2p": [1, 2],  # vehicle-pedestrian
    "v2c": [1, 3],  # vehicle-cyclist
}


def filter_interactive_data(
    input_dir: str,
    output_dir: str,
    interactive_type: str = None,
    max_files: int = None,
):
    """
    Filter interactive scenarios from training data.
    
    Args:
        input_dir: Input directory with TFRecord files
        output_dir: Output directory for filtered TFRecord files
        interactive_type: Type of interaction filter
        max_files: Maximum number of files to process (for testing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all TFRecord files
    tfrecord_files = sorted(input_path.glob("*.tfrecord*"))
    
    if max_files is not None:
        tfrecord_files = tfrecord_files[:max_files]
    
    print(f"Found {len(tfrecord_files)} TFRecord files in {input_dir}")
    print(f"Output directory: {output_dir}")
    filter_msg = interactive_type if interactive_type else 'all'
    print(f"Interactive type filter: {filter_msg}")
    print()
    
    total_sample_cnt = 0
    total_interactive_cnt = 0
    total_interactive_type_cnt = 0
    
    for tfrecord_file in tqdm(tfrecord_files, desc="Processing TFRecords"):
        dataset = tf.data.TFRecordDataset(
            str(tfrecord_file), compression_type=''
        )
        interactive_data = []
        sample_cnt = 0
        interactive_cnt = 0
        interactive_type_cnt = 0
        
        for data in dataset:
            sample_cnt += 1
            
            # Parse raw tf.Example to avoid schema issues
            example = tf.train.Example()
            example.ParseFromString(data.numpy())
            
            # Get objects of interest and types
            ooi_values = example.features.feature[
                'state/objects_of_interest'
            ].int64_list.value
            type_values = example.features.feature[
                'state/type'
            ].float_list.value
            
            # Find interactive agents (objects_of_interest > 0)
            objects_indices = [i for i, v in enumerate(ooi_values) if v > 0]
            objects_type = [int(type_values[i]) for i in objects_indices]
            
            # Check if exactly 2 agents are marked as interactive
            if len(objects_indices) == 2:
                interactive_cnt += 1
                
                # Sort types for consistent comparison
                objects_type_sorted = sorted(objects_type)
                
                # Filter by interaction type if specified
                if interactive_type is not None:
                    v2v = OBJECT_TYPE_DICT["v2v"]
                    v2p = OBJECT_TYPE_DICT["v2p"]
                    v2c = OBJECT_TYPE_DICT["v2c"]
                    
                    if interactive_type == 'v2v' and objects_type_sorted == v2v:
                        interactive_data.append(data)
                        interactive_type_cnt += 1
                    elif interactive_type == 'v2p' and objects_type_sorted == v2p:
                        interactive_data.append(data)
                        interactive_type_cnt += 1
                    elif interactive_type == 'v2c' and objects_type_sorted == v2c:
                        interactive_data.append(data)
                        interactive_type_cnt += 1
                    elif interactive_type == 'others':
                        if objects_type_sorted not in [v2v, v2p, v2c]:
                            interactive_data.append(data)
                            interactive_type_cnt += 1
                else:
                    # No filter - keep all interactive scenarios
                    interactive_data.append(data)
                    interactive_type_cnt += 1
        
        # Write interactive data to output file
        if len(interactive_data) > 0:
            output_file = output_path / tfrecord_file.name
            with tf.io.TFRecordWriter(str(output_file)) as writer:
                for data in interactive_data:
                    writer.write(data.numpy())
        
        total_sample_cnt += sample_cnt
        total_interactive_cnt += interactive_cnt
        total_interactive_type_cnt += interactive_type_cnt
    
    print()
    print("=" * 60)
    print("Filtering Complete!")
    print("=" * 60)
    print(f"Total scenarios processed:     {total_sample_cnt}")
    print(f"Interactive scenarios found:   {total_interactive_cnt}")
    print(f"Filtered scenarios saved:      {total_interactive_type_cnt}")
    print(f"Percentage interactive:        {100 * total_interactive_cnt / total_sample_cnt:.2f}%")
    if interactive_type:
        print(f"Percentage of type '{interactive_type}': {100 * total_interactive_type_cnt / total_interactive_cnt:.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Filter interactive training scenarios from Waymo dataset"
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Input directory with TFRecord files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for filtered TFRecord files'
    )
    parser.add_argument(
        '--type',
        type=str,
        default=None,
        choices=['v2v', 'v2p', 'v2c', 'others'],
        help='Filter by interaction type (v2v=vehicle-vehicle, v2p=vehicle-pedestrian, v2c=vehicle-cyclist, others=rest)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of TFRecord files to process (for testing)'
    )
    
    args = parser.parse_args()
    
    filter_interactive_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        interactive_type=args.type,
        max_files=args.max_files,
    )


if __name__ == '__main__':
    main()
