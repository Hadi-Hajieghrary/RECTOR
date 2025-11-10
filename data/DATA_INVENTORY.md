# RECTOR Data Folder Inventory

Auto-generated data folder structure (updated on commit).

## Data Directory Structure

```
data/
├── cache
├── movies
│   ├── scenario
│   │   ├── testing
│   │   ├── testing_interactive
│   │   ├── training
│   │   ├── validation
│   │   └── validation_interactive
│   └── tf_example
│       ├── testing
│       ├── training
│       └── validation
├── processed
└── waymo_open_dataset_motion_v_1_3_0
    ├── lidar_and_camera
    │   ├── testing
    │   ├── training
    │   └── validation
    ├── scenario
    │   ├── testing
    │   ├── testing_interactive
    │   ├── training
    │   ├── training_20s
    │   ├── validation
    │   └── validation_interactive
    └── tf_example
        ├── testing
        ├── training
        └── validation

29 directories
```

## Data Statistics

### cache/
- **Files**: 1
- **Total Size**: 4.0K


### movies/
- **Files**: 73
- **Total Size**: 117M

**Files:**
```
README.md
scenario/testing_interactive/2fcbbbb8a4db2a7f.gif
scenario/testing_interactive/2fcbbbb8a4db2a7f.mp4
scenario/testing_interactive/310f98b182de5b5a.gif
scenario/testing_interactive/310f98b182de5b5a.mp4
scenario/testing_interactive/6e04de6d151c38a4.gif
scenario/testing_interactive/6e04de6d151c38a4.mp4
scenario/training/28fe360951cf98d6.gif
scenario/training/28fe360951cf98d6.mp4
scenario/training/2a0be260c6478180.gif
scenario/training/2a0be260c6478180.mp4
scenario/training/610c7fcfe8e7eb4d.gif
scenario/training/610c7fcfe8e7eb4d.mp4
scenario/training/bcafb1298382d410.gif
scenario/training/bcafb1298382d410.mp4
scenario/training/bce9167fc4acc6c8.gif
scenario/training/bce9167fc4acc6c8.mp4
scenario/validation/ae0532d70d1198c.gif
scenario/validation/ae0532d70d1198c.mp4
scenario/validation/b85e1bd6cc8e74c0.gif
scenario/validation/b85e1bd6cc8e74c0.mp4
scenario/validation/d9fa8be36c4d3547.gif
scenario/validation/d9fa8be36c4d3547.mp4
scenario/validation/f010a76faccf7bc2.gif
scenario/validation/f010a76faccf7bc2.mp4
scenario/validation/f1c75f0f2e2d01a2.gif
scenario/validation/f1c75f0f2e2d01a2.mp4
scenario/validation_interactive/234dfbe99b740c80.gif
scenario/validation_interactive/234dfbe99b740c80.mp4
scenario/validation_interactive/517e33af3e9907c6.gif
scenario/validation_interactive/517e33af3e9907c6.mp4
scenario/validation_interactive/7ae30118456e74a7.gif
scenario/validation_interactive/7ae30118456e74a7.mp4
tf_example/testing/testing_tfexample_0000.gif
tf_example/testing/testing_tfexample_0000.mp4
tf_example/testing/testing_tfexample_0001.gif
tf_example/testing/testing_tfexample_0001.mp4
tf_example/testing/testing_tfexample_0002.gif
tf_example/testing/testing_tfexample_0002.mp4
tf_example/testing/testing_tfexample_0003.gif
tf_example/testing/testing_tfexample_0003.mp4
tf_example/testing/testing_tfexample_0004.gif
tf_example/testing/testing_tfexample_0004.mp4
tf_example/testing/testing_tfexample_0005.gif
tf_example/testing/testing_tfexample_0005.mp4
tf_example/testing/testing_tfexample_0006.gif
tf_example/testing/testing_tfexample_0006.mp4
tf_example/testing/testing_tfexample_0007.gif
tf_example/testing/testing_tfexample_0007.mp4
tf_example/testing/testing_tfexample_0008.gif
tf_example/testing/testing_tfexample_0008.mp4
tf_example/testing/testing_tfexample_0009.gif
tf_example/testing/testing_tfexample_0009.mp4
tf_example/validation/validation_tfexample_0000.gif
tf_example/validation/validation_tfexample_0000.mp4
tf_example/validation/validation_tfexample_0001.gif
tf_example/validation/validation_tfexample_0001.mp4
tf_example/validation/validation_tfexample_0002.gif
tf_example/validation/validation_tfexample_0002.mp4
tf_example/validation/validation_tfexample_0003.gif
tf_example/validation/validation_tfexample_0003.mp4
tf_example/validation/validation_tfexample_0004.gif
tf_example/validation/validation_tfexample_0004.mp4
tf_example/validation/validation_tfexample_0005.gif
tf_example/validation/validation_tfexample_0005.mp4
tf_example/validation/validation_tfexample_0006.gif
tf_example/validation/validation_tfexample_0006.mp4
tf_example/validation/validation_tfexample_0007.gif
tf_example/validation/validation_tfexample_0007.mp4
tf_example/validation/validation_tfexample_0008.gif
tf_example/validation/validation_tfexample_0008.mp4
tf_example/validation/validation_tfexample_0009.gif
tf_example/validation/validation_tfexample_0009.mp4
```

### processed/
- **Files**: 1
- **Total Size**: 4.0K


### waymo_open_dataset_motion_v_1_3_0/
- **Files**: 52
- **Total Size**: 15G

**Waymo Dataset Splits**:

#### Scenario Format
- **testing**: 5 files (1021M)
  ```
  testing.tfrecord-00000-of-00150 (195016429 bytes)
  testing.tfrecord-00001-of-00150 (204309365 bytes)
  testing.tfrecord-00002-of-00150 (209368873 bytes)
  testing.tfrecord-00003-of-00150 (235137785 bytes)
  testing.tfrecord-00004-of-00150 (225847072 bytes)
  ```
- **testing_interactive**: 3 files (555M)
  ```
  testing_interactive.tfrecord-00000-of-00150 (209256472 bytes)
  testing_interactive.tfrecord-00001-of-00150 (199796304 bytes)
  testing_interactive.tfrecord-00002-of-00150 (172237150 bytes)
  ```
- **training**: 5 files (2.2G)
  ```
  training.tfrecord-00000-of-01000 (454771008 bytes)
  training.tfrecord-00001-of-01000 (481162498 bytes)
  training.tfrecord-00002-of-01000 (503552651 bytes)
  training.tfrecord-00003-of-01000 (449454935 bytes)
  training.tfrecord-00004-of-01000 (444764080 bytes)
  ```
- **training_20s**: empty
- **validation**: 5 files (1.3G)
  ```
  validation.tfrecord-00000-of-00150 (262636608 bytes)
  validation.tfrecord-00001-of-00150 (286425611 bytes)
  validation.tfrecord-00002-of-00150 (271464873 bytes)
  validation.tfrecord-00003-of-00150 (277970112 bytes)
  validation.tfrecord-00004-of-00150 (260872317 bytes)
  ```
- **validation_interactive**: 3 files (767M)
  ```
  validation_interactive.tfrecord-00000-of-00150 (264769762 bytes)
  validation_interactive.tfrecord-00001-of-00150 (280782912 bytes)
  validation_interactive.tfrecord-00002-of-00150 (258587650 bytes)
  ```

#### TF Example Format
- **testing**: 5 files (2.7G)
  ```
  testing_tfexample.tfrecord-00000-of-00150 (603222670 bytes)
  testing_tfexample.tfrecord-00001-of-00150 (552006037 bytes)
  testing_tfexample.tfrecord-00002-of-00150 (565998453 bytes)
  testing_tfexample.tfrecord-00003-of-00150 (541792746 bytes)
  testing_tfexample.tfrecord-00004-of-00150 (537308800 bytes)
  ```
- **training**: 5 files (4.1G)
  ```
  training_tfexample.tfrecord-00000-of-01000 (853640898 bytes)
  training_tfexample.tfrecord-00001-of-01000 (838374658 bytes)
  training_tfexample.tfrecord-00002-of-01000 (856053317 bytes)
  training_tfexample.tfrecord-00003-of-01000 (917744611 bytes)
  training_tfexample.tfrecord-00004-of-01000 (850300520 bytes)
  ```
- **validation**: 5 files (2.5G)
  ```
  validation_tfexample.tfrecord-00000-of-00150 (507109827 bytes)
  validation_tfexample.tfrecord-00001-of-00150 (529834406 bytes)
  validation_tfexample.tfrecord-00002-of-00150 (503840997 bytes)
  validation_tfexample.tfrecord-00003-of-00150 (553774234 bytes)
  validation_tfexample.tfrecord-00004-of-00150 (552286912 bytes)
  ```

#### Lidar & Camera Format
- **testing**: empty
- **training**: empty
- **validation**: empty


*Last updated: 2025-11-10 02:08:12 UTC*
