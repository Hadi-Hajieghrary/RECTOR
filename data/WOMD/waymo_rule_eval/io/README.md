# IO Module - Output Sinks and Data Serialization

## Overview

The `io` module provides a unified interface for writing rule evaluation results to various output formats. It implements the **Sink Pattern**, where all output destinations expose the same interface regardless of the underlying format (JSONL, CSV, Parquet).

## Philosophy

### Design Principles

**1. Format Agnosticism**
The CLI and pipeline code should not care about output format. The sink abstraction allows seamless switching between formats based on file extension:

```python
sink = make_sink("results.jsonl")  # Creates JsonlSink
sink = make_sink("results.csv")    # Creates CsvSink
sink = make_sink("results.parquet") # Creates ParquetSink
```

**2. Streaming-First Architecture**
For large-scale evaluations, results must be written incrementally to avoid memory exhaustion. All sinks support streaming writes:

```python
with JsonlSink("results.jsonl") as sink:
    for result in evaluation_results:
        sink.write(result)  # Written immediately to disk
```

**3. Graceful Degradation**
Optional dependencies (like PyArrow for Parquet) degrade gracefully:

```python
# If PyArrow not installed, falls back to JSONL
sink = make_sink("results.parquet")
# WARNING: PyArrow not available, falling back to JSONL
```

**4. Context Manager Protocol**
All sinks support Python's context manager protocol for automatic resource cleanup:

```python
with make_sink("results.jsonl") as sink:
    sink.write(record)
# File automatically closed, even on exceptions
```

## Module Structure

```
io/
├── __init__.py     # Sink implementations and factory
└── README.md       # This documentation
```

## Available Sinks

### JsonlSink (JSON Lines)

**Best For**: Log analysis, streaming processing, human readability

```python
from waymo_rule_eval.io import JsonlSink

with JsonlSink("results.jsonl") as sink:
    sink.write({"rule_id": "L10.R1", "severity": 0.5})
    sink.write({"rule_id": "L6.R2", "severity": 0.3})
```

**Output Format**:
```jsonl
{"rule_id": "L10.R1", "severity": 0.5}
{"rule_id": "L6.R2", "severity": 0.3}
```

**Characteristics**:
- One JSON object per line
- Human-readable
- Append-friendly (can add records to existing file)
- Easy to process with `jq`, Python, or streaming tools
- No schema enforcement (flexible but less safe)

### CsvSink

**Best For**: Spreadsheet analysis, simple tabular data, Excel import

```python
from waymo_rule_eval.io import CsvSink

with CsvSink("results.csv") as sink:
    sink.write({"rule_id": "L10.R1", "severity": 0.5})
    sink.write({"rule_id": "L6.R2", "severity": 0.3})
```

**Output Format**:
```csv
rule_id,severity
L10.R1,0.5
L6.R2,0.3
```

**Characteristics**:
- Standard tabular format
- Column names from first record (or specified explicitly)
- Compatible with spreadsheet software
- Limited type support (everything becomes strings)

### ParquetSink

**Best For**: Big data analytics, columnar queries, compression

```python
from waymo_rule_eval.io import ParquetSink

with ParquetSink("results.parquet") as sink:
    sink.write({"rule_id": "L10.R1", "severity": 0.5})
    sink.write({"rule_id": "L6.R2", "severity": 0.3})
# Data written on close()
```

**Characteristics**:
- Columnar storage (efficient for analytics)
- Built-in compression
- Schema-aware (preserves types)
- Requires PyArrow: `pip install pyarrow`
- **Note**: Records are buffered and written on `close()`

## Factory Function

The `make_sink()` factory automatically selects the appropriate sink:

```python
from waymo_rule_eval.io import make_sink

# Automatic format detection
sink = make_sink("results.jsonl")   # JsonlSink
sink = make_sink("results.csv")     # CsvSink
sink = make_sink("results.parquet") # ParquetSink

# Fallback behavior
sink = make_sink("results.parquet")
# If PyArrow unavailable, creates JsonlSink at results.jsonl
```

## API Reference

### Common Interface

All sinks implement the same interface:

```python
class Sink:
    def write(self, record: Dict[str, Any]) -> None:
        """Write a single record."""
        ...

    def write_many(self, records: Iterable[Dict[str, Any]]) -> None:
        """Write multiple records."""
        ...

    def close(self) -> None:
        """Close the sink and flush any buffers."""
        ...

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (calls close)."""
        self.close()
        return False
```

### JsonlSink

```python
class JsonlSink:
    def __init__(self, path: str):
        """
        Initialize JSONL sink.

        Args:
            path: Output file path (.jsonl)

        Creates parent directories if needed.
        Opens file in append mode for resumability.
        """
```

### CsvSink

```python
class CsvSink:
    def __init__(self, path: str, columns: Optional[list] = None):
        """
        Initialize CSV sink.

        Args:
            path: Output file path (.csv)
            columns: Column names (auto-detected from first record if None)
        """
```

### ParquetSink

```python
class ParquetSink:
    def __init__(self, path: str):
        """
        Initialize Parquet sink.

        Args:
            path: Output file path (.parquet)

        Raises:
            ImportError: If PyArrow is not installed

        Note: Records are buffered in memory and written on close().
        """
```

## Usage Examples

### Basic Pipeline Integration

```python
from waymo_rule_eval.io import make_sink
from waymo_rule_eval.pipeline import WindowedExecutor

executor = WindowedExecutor()
results = executor.run_batch(scenarios)

with make_sink("output/results.jsonl") as sink:
    for window_result in results:
        for rule_result in window_result.results:
            sink.write({
                "scenario_id": window_result.scenario_id,
                "rule_id": rule_result.rule_id,
                "severity": rule_result.severity,
            })
```

### Custom Columns for CSV

```python
from waymo_rule_eval.io import CsvSink

# Specify column order explicitly
with CsvSink("results.csv", columns=["scenario_id", "rule_id", "severity", "applicable"]) as sink:
    sink.write({"rule_id": "L10.R1", "severity": 0.5, "scenario_id": "s1", "applicable": True})
```

### Batch Writing

```python
from waymo_rule_eval.io import JsonlSink

records = [
    {"rule_id": "L10.R1", "severity": 0.5},
    {"rule_id": "L6.R2", "severity": 0.3},
    {"rule_id": "L0.R2", "severity": 0.0},
]

with JsonlSink("results.jsonl") as sink:
    sink.write_many(records)
```

### Error-Safe Writing

```python
from waymo_rule_eval.io import make_sink

sink = make_sink("results.jsonl")
try:
    for result in process_scenarios():
        sink.write(result)
except Exception:
    pass  # Partial results preserved
finally:
    sink.close()  # Always close
```

## Output Record Schema

While sinks accept arbitrary dictionaries, the standard output schema includes:

```python
{
    # Identification
    "run_id": str,              # Evaluation run identifier
    "scenario_id": str,         # Unique scenario identifier

    # Temporal context
    "window_start": float,      # Window start time (seconds)
    "window_end": float,        # Window end time (seconds)

    # Rule evaluation
    "rule_id": str,             # Rule identifier (e.g., "L10.R1")
    "applicable": bool,         # Whether rule applied
    "severity": float,          # Raw violation severity
    "normalized_severity": float,  # Severity in [0, 1]
    "explanation": str,         # Human-readable description

    # Optional metadata
    "rule_version": str,        # Rule implementation version
    "engine_version": str,      # Pipeline version
    "timestamp": str,           # ISO 8601 timestamp
}
```

## Format Comparison

| Feature | JSONL | CSV | Parquet |
|---------|-------|-----|---------|
| Human Readable | ✅ Yes | ✅ Yes | ❌ No |
| Streaming Write | ✅ Yes | ✅ Yes | ❌ Buffered |
| Compression | ❌ No | ❌ No | ✅ Built-in |
| Type Preservation | ✅ Yes | ❌ All strings | ✅ Yes |
| Schema Enforcement | ❌ No | ⚠️ First record | ✅ Yes |
| Analytics-Ready | ⚠️ Okay | ⚠️ Okay | ✅ Excellent |
| Dependencies | None | None | PyArrow |
| Append Support | ✅ Yes | ⚠️ Headers | ❌ No |

## Best Practices

### Choosing a Format

1. **Use JSONL for**:
   - Development and debugging
   - Log aggregation systems
   - Streaming pipelines
   - When human readability matters

2. **Use CSV for**:
   - Quick analysis in Excel/Google Sheets
   - Simple reporting
   - Legacy system integration

3. **Use Parquet for**:
   - Large-scale analytics
   - Data warehousing
   - When storage efficiency matters
   - Integration with Spark/Pandas

### Directory Creation

Sinks automatically create parent directories:

```python
sink = JsonlSink("output/subdir/results.jsonl")
# Creates output/subdir/ if needed
```

### Resumable Writes

JSONL supports appending to existing files:

```python
# First run
with JsonlSink("results.jsonl") as sink:
    sink.write({"batch": 1, "count": 100})

# Second run (appends)
with JsonlSink("results.jsonl") as sink:
    sink.write({"batch": 2, "count": 200})
```

### Memory Considerations

- **JSONL/CSV**: Each `write()` flushes to disk immediately
- **Parquet**: All records buffered until `close()`

For large Parquet outputs, consider chunking:

```python
chunk_size = 100000
records = []

for result in results:
    records.append(result)
    if len(records) >= chunk_size:
        with ParquetSink(f"results_{chunk_id}.parquet") as sink:
            sink.write_many(records)
        records = []
        chunk_id += 1
```

## Troubleshooting

### PyArrow Not Installed

```
ImportError: PyArrow required for Parquet output.
Install with: pip install pyarrow
```

**Solution**: Install PyArrow or use JSONL format instead.

### Empty Parquet File

```
WARNING: ParquetSink closed with no records
```

**Cause**: No records written before close.
**Solution**: Ensure records are written before closing.

### Permission Errors

```
PermissionError: [Errno 13] Permission denied: '/path/to/results.jsonl'
```

**Solution**: Check write permissions on target directory.

### Large Memory Usage

**Cause**: Parquet buffering all records.
**Solution**: Use JSONL for streaming, or chunk Parquet writes.

## See Also

- **[CLI README](../cli/README.md)** - How sinks are used in CLI
- **[Pipeline README](../pipeline/README.md)** - Result generation
- **[Main README](../README.md)** - Package overview
