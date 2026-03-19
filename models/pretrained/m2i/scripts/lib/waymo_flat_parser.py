"""
Custom TFRecord parser for M2I that handles the flat format TFRecords
in the RECTOR workspace.

The original M2I waymo_tutorial.py expects Waymo Motion Dataset format with
shaped tensors like [20000, 3] for roadgraph_samples/xyz.

The RECTOR workspace has TFRecords with flat format like [90000] for xyz.

This module provides a drop-in replacement for waymo_tutorial._parse()
"""

import numpy as np
import tensorflow as tf


# ============================================================================
# Flat format feature description (for RECTOR workspace TFRecords)
# ============================================================================

# Roadgraph is stored flat: 30000 points * 3 coords = 90000
roadgraph_features_flat = {
    'roadgraph_samples/dir': tf.io.FixedLenFeature([90000], tf.float32, default_value=None),
    'roadgraph_samples/id': tf.io.FixedLenFeature([30000], tf.int64, default_value=None),
    'roadgraph_samples/type': tf.io.FixedLenFeature([30000], tf.int64, default_value=None),
    'roadgraph_samples/valid': tf.io.FixedLenFeature([30000], tf.int64, default_value=None),
    'roadgraph_samples/xyz': tf.io.FixedLenFeature([90000], tf.float32, default_value=None),
}

# State features are stored flat or with different shapes
state_features_flat = {
    'state/id': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc': tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict': tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/height': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/length': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/timestamp_micros': tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/valid': tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/vel_yaw': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/velocity_x': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/velocity_y': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/width': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/x': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/y': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/current/z': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/future/bbox_yaw': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/height': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/length': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/timestamp_micros': tf.io.FixedLenFeature([10240], tf.int64, default_value=None),
    'state/future/valid': tf.io.FixedLenFeature([10240], tf.int64, default_value=None),
    'state/future/vel_yaw': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/velocity_x': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/velocity_y': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/width': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/x': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/y': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/future/z': tf.io.FixedLenFeature([10240], tf.float32, default_value=None),
    'state/past/bbox_yaw': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/height': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/length': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/timestamp_micros': tf.io.FixedLenFeature([1280], tf.int64, default_value=None),
    'state/past/valid': tf.io.FixedLenFeature([1280], tf.int64, default_value=None),
    'state/past/vel_yaw': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/velocity_x': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/velocity_y': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/width': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/x': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/y': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
    'state/past/z': tf.io.FixedLenFeature([1280], tf.float32, default_value=None),
}

# Traffic light features - flat format
traffic_light_features_flat = {
    'traffic_light_state/current/state': tf.io.FixedLenFeature([16], tf.int64, default_value=None),
    'traffic_light_state/current/valid': tf.io.FixedLenFeature([16], tf.int64, default_value=None),
    'traffic_light_state/current/id': tf.io.FixedLenFeature([16], tf.int64, default_value=None),
    'traffic_light_state/current/x': tf.io.FixedLenFeature([16], tf.float32, default_value=None),
    'traffic_light_state/current/y': tf.io.FixedLenFeature([16], tf.float32, default_value=None),
    'traffic_light_state/current/z': tf.io.FixedLenFeature([16], tf.float32, default_value=None),
    'traffic_light_state/past/state': tf.io.FixedLenFeature([160], tf.int64, default_value=None),
    'traffic_light_state/past/valid': tf.io.FixedLenFeature([160], tf.int64, default_value=None),
    'traffic_light_state/past/x': tf.io.FixedLenFeature([160], tf.float32, default_value=None),
    'traffic_light_state/past/y': tf.io.FixedLenFeature([160], tf.float32, default_value=None),
    'traffic_light_state/past/z': tf.io.FixedLenFeature([160], tf.float32, default_value=None),
    'traffic_light_state/past/id': tf.io.FixedLenFeature([160], tf.int64, default_value=None),
}

features_description_flat = {}
features_description_flat.update(roadgraph_features_flat)
features_description_flat.update(state_features_flat)
features_description_flat.update(traffic_light_features_flat)
features_description_flat['scenario/id'] = tf.io.FixedLenFeature([], tf.string, default_value='')
features_description_flat['state/objects_of_interest'] = tf.io.FixedLenFeature([128], tf.int64, default_value=None)


def reshape_flat_to_shaped(decoded_example):
    """
    Reshape the flat format data to the shaped format expected by M2I.
    
    Transforms:
    - roadgraph_samples/xyz: [90000] -> [30000, 3]  (NOT truncating - keep all points!)
    - state/current/*: [128] -> [128, 1]
    - state/past/*: [1280] -> [128, 10]
    - state/future/*: [10240] -> [128, 80]
    - traffic_light_state/current/*: [16] -> [1, 16]
    - traffic_light_state/past/*: [160] -> [10, 16]
    
    Also remaps lane IDs to fit within M2I's max_lane_num constraint (1000).
    Also clamps road types to valid range [0, 19] for M2I compatibility.
    """
    reshaped = {}
    
    # Roadgraph - reshape (keep all 30000 points, don't truncate!)
    reshaped['roadgraph_samples/dir'] = tf.reshape(
        decoded_example['roadgraph_samples/dir'], [30000, 3])
    
    # Clamp road types to [0, 19] range (M2I asserts type < 20)
    # Waymo has types up to 20, so we map 20 -> 19
    raw_types = tf.reshape(decoded_example['roadgraph_samples/type'], [30000, 1])
    clamped_types = tf.clip_by_value(raw_types, clip_value_min=-1, clip_value_max=19)
    reshaped['roadgraph_samples/type'] = clamped_types
    
    reshaped['roadgraph_samples/valid'] = tf.reshape(
        decoded_example['roadgraph_samples/valid'], [30000, 1])
    reshaped['roadgraph_samples/xyz'] = tf.reshape(
        decoded_example['roadgraph_samples/xyz'], [30000, 3])
    
    # Remap lane IDs to be within [0, 999] range (M2I max_lane_num = 1000)
    raw_ids = decoded_example['roadgraph_samples/id']
    raw_ids_reshaped = tf.reshape(raw_ids, [30000, 1])
    # Use modulo to ensure IDs are within range
    remapped_ids = tf.cast(tf.math.mod(tf.cast(raw_ids_reshaped, tf.int32), 999), tf.int64)
    reshaped['roadgraph_samples/id'] = remapped_ids
    
    # State ID and type - keep as is
    reshaped['state/id'] = decoded_example['state/id']
    reshaped['state/type'] = decoded_example['state/type']
    reshaped['state/is_sdc'] = decoded_example['state/is_sdc']
    reshaped['state/tracks_to_predict'] = decoded_example['state/tracks_to_predict']
    reshaped['state/objects_of_interest'] = decoded_example['state/objects_of_interest']
    
    # Scenario ID
    reshaped['scenario/id'] = tf.expand_dims(decoded_example['scenario/id'], 0)
    
    # Current state - add trailing dimension [128] -> [128, 1]
    for key in ['bbox_yaw', 'height', 'length', 'valid', 'vel_yaw', 
                'velocity_x', 'velocity_y', 'width', 'x', 'y', 'z']:
        full_key = f'state/current/{key}'
        reshaped[full_key] = tf.expand_dims(decoded_example[full_key], -1)
    
    # Handle timestamp_micros separately (int64)
    reshaped['state/current/timestamp_micros'] = tf.expand_dims(
        decoded_example['state/current/timestamp_micros'], -1)
    
    # Past state - reshape [1280] -> [128, 10]
    for key in ['bbox_yaw', 'height', 'length', 'valid', 'vel_yaw',
                'velocity_x', 'velocity_y', 'width', 'x', 'y', 'z']:
        full_key = f'state/past/{key}'
        reshaped[full_key] = tf.reshape(decoded_example[full_key], [128, 10])
    reshaped['state/past/timestamp_micros'] = tf.reshape(
        decoded_example['state/past/timestamp_micros'], [128, 10])
    
    # Future state - reshape [10240] -> [128, 80]
    for key in ['bbox_yaw', 'height', 'length', 'valid', 'vel_yaw',
                'velocity_x', 'velocity_y', 'width', 'x', 'y', 'z']:
        full_key = f'state/future/{key}'
        reshaped[full_key] = tf.reshape(decoded_example[full_key], [128, 80])
    reshaped['state/future/timestamp_micros'] = tf.reshape(
        decoded_example['state/future/timestamp_micros'], [128, 80])
    
    # Traffic light - current [16] -> [1, 16]
    for key in ['state', 'valid', 'id', 'x', 'y', 'z']:
        full_key = f'traffic_light_state/current/{key}'
        reshaped[full_key] = tf.expand_dims(decoded_example[full_key], 0)
    
    # Traffic light - past [160] -> [10, 16]
    for key in ['state', 'valid', 'id', 'x', 'y', 'z']:
        full_key = f'traffic_light_state/past/{key}'
        reshaped[full_key] = tf.reshape(decoded_example[full_key], [10, 16])
    
    return reshaped


def _parse_flat(value):
    """
    Parse a flat-format TFRecord and reshape to M2I expected format.
    
    This is a drop-in replacement for waymo_tutorial._parse()
    """
    decoded_example_flat = tf.io.parse_single_example(value, features_description_flat)
    decoded_example = reshape_flat_to_shaped(decoded_example_flat)
    
    # Now build the same outputs as the original _parse function
    past_states = tf.stack([
        decoded_example['state/past/x'],
        decoded_example['state/past/y'],
        decoded_example['state/past/length'],
        decoded_example['state/past/width'],
        decoded_example['state/past/bbox_yaw'],
        decoded_example['state/past/velocity_x'],
        decoded_example['state/past/velocity_y']
    ], -1)

    cur_states = tf.stack([
        decoded_example['state/current/x'],
        decoded_example['state/current/y'],
        decoded_example['state/current/length'],
        decoded_example['state/current/width'],
        decoded_example['state/current/bbox_yaw'],
        decoded_example['state/current/velocity_x'],
        decoded_example['state/current/velocity_y']
    ], -1)

    input_states = tf.concat([past_states, cur_states], 1)[..., :2]

    future_states = tf.stack([
        decoded_example['state/future/x'],
        decoded_example['state/future/y'],
        decoded_example['state/future/length'],
        decoded_example['state/future/width'],
        decoded_example['state/future/bbox_yaw'],
        decoded_example['state/future/velocity_x'],
        decoded_example['state/future/velocity_y']
    ], -1)

    gt_future_states = tf.concat([past_states, cur_states, future_states], 1)

    past_is_valid = decoded_example['state/past/valid'] > 0
    current_is_valid = decoded_example['state/current/valid'] > 0
    future_is_valid = decoded_example['state/future/valid'] > 0
    gt_future_is_valid = tf.concat([past_is_valid, current_is_valid, future_is_valid], 1)

    # Build sample_is_valid from current validity
    sample_is_valid = tf.squeeze(current_is_valid, axis=-1)

    # Build tracks_to_predict
    tracks_to_predict = decoded_example['state/tracks_to_predict'] > 0
    
    # Build interactive_tracks_to_predict (objects_of_interest)
    interactive_tracks_to_predict = decoded_example['state/objects_of_interest'] > 0

    inputs = {
        'input_states': input_states,
        'gt_future_states': gt_future_states,
        'gt_future_is_valid': gt_future_is_valid,
        'tracks_to_predict': tracks_to_predict,
        'sample_is_valid': sample_is_valid,
        'interactive_tracks_to_predict': interactive_tracks_to_predict,
    }

    return inputs, decoded_example


# Monkey-patch the original waymo_tutorial module if needed
def patch_waymo_tutorial():
    """
    Patch the waymo_tutorial module to use our flat parser.
    Call this before using M2I with flat-format TFRecords.
    """
    import sys
    if 'waymo_tutorial' in sys.modules:
        import waymo_tutorial
        waymo_tutorial._parse = _parse_flat
        print("Patched waymo_tutorial._parse to use flat format parser")
    else:
        print("Warning: waymo_tutorial not yet imported, patch will be applied on import")
