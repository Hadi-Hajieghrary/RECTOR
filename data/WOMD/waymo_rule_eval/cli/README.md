# CLI Module - Command-Line Interface

## Overview

The `cli` module provides a user-friendly command-line interface for running rule evaluations on Waymo Open Motion Dataset scenarios. It bridges the gap between raw TFRecord data and actionable safety analysis results, offering a simple entry point for both interactive exploration and batch processing pipelines.

> **Execution context:** Run examples from `/workspace/data/WOMD` (or set `PYTHONPATH=/workspace/data/WOMD`) so `python -m waymo_rule_eval.cli.wrun` can import the package correctly.

## Philosophy

### Design Principles

**1. Single Entry Point, Multiple Data Sources**
The CLI abstracts away the complexity of different data formats through a unified interface. Whether evaluating synthetic test scenarios (JSON) or real Waymo Motion data (TFRecord), the same command structure applies:

```bash
# Real Waymo data
python -m waymo_rule_eval.cli.wrun --scenario /path/to/tfrecords/*.tfrecord --out results.jsonl

# Synthetic test scenarios
python -m waymo_rule_eval.cli.wrun --synthetic scenario.json --out results.jsonl
```

**2. Streaming Output Architecture**
Results are written incrementally as they are computed, using streaming sinks (JSONL, CSV, Parquet). This design supports:
- Large-scale batch processing without memory exhaustion
- Real-time progress monitoring
- Fault tolerance (partial results preserved on interruption)

**3. Flexible Output Formats**
Output format is determined by file extension, supporting diverse downstream workflows:
- `.jsonl` - Line-delimited JSON for streaming/log analysis
- `.csv` - Tabular format for spreadsheet analysis
- `.parquet` - Columnar storage for big data analytics

**4. Reproducibility Through Run IDs**
Every evaluation run is tagged with a unique identifier (`--run-id`), ensuring that results can be traced back to specific pipeline configurations and enabling A/B comparisons between different rule versions.

## Command Reference

### Basic Usage

```bash
python -m waymo_rule_eval.cli.wrun [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--scenario GLOB` | Glob pattern for Waymo TFRecord files |
| `--synthetic PATH` | Path to synthetic JSON scenario (mutually exclusive with `--scenario`) |
| `--out PATH` | Output file path (format determined by extension) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--window-size` | 8.0 | Evaluation window size in seconds |
| `--stride` | 2.0 | Stride between consecutive windows in seconds |
| `--run-id` | run-0001 | Identifier for tracking this evaluation run |
| `--rules` | all | Comma-separated list of rule IDs to evaluate |
| `-v, --verbose` | False | Enable debug logging |

## Examples

### Basic Evaluation

```bash
# Evaluate all scenarios in a directory
python -m waymo_rule_eval.cli.wrun \
    --scenario "/data/waymo/*.tfrecord" \
    --out results.jsonl
```

### Custom Window Configuration

```bash
# 10-second windows with 5-second overlap
python -m waymo_rule_eval.cli.wrun \
    --scenario "/data/waymo/*.tfrecord" \
    --window-size 10.0 \
    --stride 5.0 \
    --out results.jsonl
```

### Selective Rule Evaluation

```bash
# Only evaluate collision and following distance rules
python -m waymo_rule_eval.cli.wrun \
    --scenario "/data/waymo/*.tfrecord" \
    --rules "L10.R1,L6.R2,L0.R2" \
    --out collision_results.jsonl
```

### Tagged Evaluation Run

```bash
# Tag results for A/B comparison
python -m waymo_rule_eval.cli.wrun \
    --scenario "/data/waymo/*.tfrecord" \
    --run-id "experiment-v2-2024-01" \
    --out experiment_results.jsonl
```

### Synthetic Scenario Testing

```bash
# Evaluate a hand-crafted test scenario
python -m waymo_rule_eval.cli.wrun \
    --synthetic test_cases/collision_scenario.json \
    --out test_results.jsonl
```

## Output Format

Each output record contains:

```json
{
    "run_id": "run-0001",
    "scenario_id": "scenario_12345",
    "window_start": 0.0,
    "window_end": 8.0,
    "rule_id": "L10.R1",
    "applicable": true,
    "severity": 0.15,
    "normalized_severity": 0.075,
    "explanation": "Collision detected with vehicle at t=3.2s; penetration=0.15m"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Evaluation run identifier |
| `scenario_id` | string | Unique scenario identifier from dataset |
| `window_start` | float | Window start time (seconds) |
| `window_end` | float | Window end time (seconds) |
| `rule_id` | string | Rule identifier (e.g., "L10.R1") |
| `applicable` | bool | Whether the rule applied to this window |
| `severity` | float | Raw violation severity (rule-specific units) |
| `normalized_severity` | float | Severity normalized to [0, 1] scale |
| `explanation` | string | Human-readable violation description |

## Architecture

### Module Structure

```
cli/
├── __init__.py       # Package exports
├── wrun.py           # Main CLI implementation
└── README.md         # This documentation
```

### Execution Flow

```
1. parse_args()          Parse command-line arguments
        │
        ▼
2. make_sink()           Create output sink based on file extension
        │
        ▼
3. iter_*_scenarios()    Load scenarios from data source
        │
        ▼
4. WindowedExecutor      Execute rules across all windows
        │
        ▼
5. sink.write()          Stream results to output file
```

### Data Source Adapters

The CLI uses specialized adapters for each data source:

**TFRecord Scenarios (`iter_tfrecord_scenarios`)**
```python
def iter_tfrecord_scenarios(glob_pattern, window_size, stride):
    """
    Load Waymo Motion scenarios from TFRecord files.

    Uses MotionScenarioReader to parse the Waymo protobuf format
    and extract trajectories, map elements, and metadata.
    """
    adapter = MotionScenarioReader()
    for fpath in sorted(glob.glob(glob_pattern)):
        for ctx in adapter.load_file(fpath):
            yield ctx
```

**Synthetic Scenarios (`iter_synthetic_scenarios`)**
```python
def iter_synthetic_scenarios(path, window_size, stride):
    """
    Load synthetic scenarios from JSON.

    Uses SyntheticScenarioAdapter for hand-crafted test cases
    with explicit trajectory and map definitions.
    """
    adapter = SyntheticScenarioAdapter()
    base_ctx = adapter.load(json.load(open(path)))
    windows = make_windows_timed(duration, window_size, stride)
    for window in windows:
        yield base_ctx
```

## Error Handling

### Graceful Degradation

The CLI continues processing even when individual scenarios fail:

```python
for fpath in files:
    try:
        for ctx in adapter.load_file(fpath):
            yield ctx
    except Exception as e:
        log.error(f"Failed to load {fpath}: {e}")
        continue  # Skip to next file
```

### Interrupt Handling

The CLI handles keyboard interrupts (Ctrl+C) gracefully:

```python
try:
    # Processing loop
except KeyboardInterrupt:
    log.warning("Interrupted by user (SIGINT)")
    sink.close()  # Ensure partial results are saved
    return 130
```

## Context-Aware Logging

The CLI uses structured logging with context variables for better traceability:

```python
from ..utils.wre_logging import set_ctx, reset_ctx

# Set context for all log messages in this run
tokens = set_ctx(run_id=parsed.run_id)
try:
    # All log messages include run_id context
    log.info("Processing scenario", extra={"scenario_id": sid})
finally:
    reset_ctx(tokens)
```

## Integration with Other Modules

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (wrun)                          │
├─────────────────────────────────────────────────────────────┤
│  Uses:                                                      │
│  ├── core.context.ScenarioContext                           │
│  ├── pipeline.rule_executor.WindowedExecutor                │
│  ├── data_access.adapter_motion_scenario.MotionScenarioReader│
│  ├── data_access.adapter_synthetic.SyntheticScenarioAdapter │
│  ├── io.{JsonlSink, CsvSink, ParquetSink}                   │
│  └── utils.wre_logging.{get_logger, set_ctx, reset_ctx}     │
└─────────────────────────────────────────────────────────────┘
```

## Advanced Usage

### Programmatic Invocation

The CLI can be invoked programmatically for testing or automation:

```python
from waymo_rule_eval.cli import main

# Equivalent to command-line invocation
exit_code = main([
    "--scenario", "/data/waymo/*.tfrecord",
    "--out", "results.jsonl",
    "--run-id", "automated-test-001"
])
```

### Custom Output Processing

For advanced analysis, process JSONL output directly:

```python
import json

with open("results.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record["severity"] > 0.5:
            print(f"High severity: {record['rule_id']} in {record['scenario_id']}")
```

### Parallel Processing Wrapper

For large-scale evaluation, wrap the CLI in a parallel processor:

```bash
#!/bin/bash
# Process shards in parallel
for shard in /data/waymo/shard_*.tfrecord; do
    python -m waymo_rule_eval.cli.wrun \
        --scenario "$shard" \
        --out "results/$(basename $shard .tfrecord).jsonl" &
done
wait
```

## Best Practices

1. **Use Meaningful Run IDs**: Include date, experiment name, or configuration in run IDs for easy tracking

2. **Validate with Small Samples**: Test on a single file before running large batches

3. **Monitor Disk Space**: JSONL output grows linearly with scenarios; use Parquet for compression

4. **Check Logs for Errors**: Errors in individual scenarios don't stop the pipeline—review logs for failures

5. **Version Your Rules**: Use `--rules` to limit evaluation when testing specific rule changes

## Troubleshooting

### No Files Found

```
WARNING - No files found matching: /path/to/*.tfrecord
```
**Solution**: Verify the glob pattern and ensure TFRecord files exist at the path.

### Import Errors

```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution**: Install required dependencies: `pip install tensorflow shapely numpy scipy`

### Memory Issues

```
MemoryError: Unable to allocate...
```
**Solution**: Process files one at a time or use smaller window sizes. Results stream to disk so memory usage should remain bounded.

### Parquet Not Available

```
WARNING - PyArrow not available, falling back to JSONL
```
**Solution**: Install PyArrow for Parquet support: `pip install pyarrow`

## See Also

- **[Pipeline README](../pipeline/README.md)** - Executor implementation details
- **[Data Access README](../data_access/README.md)** - Adapter documentation
- **[IO README](../io/README.md)** - Output sink formats
- **[Main README](../README.md)** - Package overview
