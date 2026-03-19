# `data/datasets/` — Dataset Storage (Placeholder)

This directory contains augmented TFRecords for testing_interactive, training_interactive, and possibly validation_interactive splits.
In the intended workspace layout, this is where **large datasets** live (and are usually mounted as a volume).

## Expected Contents (Waymo Motion Dataset Example)

A common on-disk layout used by the scripts in this repository:

```
data/datasets/
  waymo_open_dataset/
    motion_v_1_3_0/
      raw/
        scenario/
          training_20s/
          training_interactive/
          validation_interactive/
          testing_interactive/
          ...
      processed/
        tf/
          training_20s/
          training_interactive/
          validation_interactive/
          testing_interactive/
          ...
```

- **`raw/scenario/*`** contains TFRecords where each record is a serialized `scenario_pb2.Scenario`
- **`processed/tf/*`** contains TFRecords where each record is a serialized `tf.train.Example` in the Motion TF Example format used by M2I

## How This Folder Is Used

- `data/WOMD/scripts/lib/filter_interactive_scenario.py` reads from `raw/scenario/...`
- `data/WOMD/scripts/lib/scenario_to_example.py` writes to `processed/tf/...`
- Visualization scripts can read from either raw or processed data, depending on CLI flags
- M2I inference scripts primarily read from `processed/tf/...`

## Tip: Keep It Out of Git

Large dataset folders should not be committed.
This repository keeps `data/datasets/` empty on purpose so you can mount your data at runtime.
