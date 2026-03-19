"""
Output sinks for storing rule evaluation results.

Provides JSONL and Parquet output formats for storing
per-window violation metrics.
"""

import csv
import json
import logging
import os
from typing import Any, Dict, Iterable, Optional

log = logging.getLogger(__name__)


class JsonlSink:
    """
    JSON Lines output sink.

    Writes each record as a separate JSON line, enabling
    streaming writes and easy appending.
    """

    def __init__(self, path: str):
        """Initialize JSONL sink at the given file path."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path = path
        # Append mode: JSONL is log-structured, safe to append across runs
        self._fp = open(path, "a", encoding="utf-8")
        log.info(f"JsonlSink opened: {path}")

    def write(self, record: Dict[str, Any]) -> None:
        """Write a single record."""
        self._fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()

    def write_many(self, records: Iterable[Dict[str, Any]]) -> None:
        """Write multiple records."""
        for rec in records:
            self.write(rec)

    def close(self) -> None:
        """Close the sink."""
        try:
            self._fp.close()
            log.info(f"JsonlSink closed: {self._path}")
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CsvSink:
    """
    CSV output sink for simple tabular output.
    """

    def __init__(self, path: str, columns: Optional[list] = None):
        """
        Initialize CSV sink.

        Args:
            path: Output file path (.csv)
            columns: Column names (auto-detected if None)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path = path
        self._columns = columns
        self._header_written = False
        self._fp = open(path, "w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fp, quoting=csv.QUOTE_MINIMAL)
        log.info(f"CsvSink opened: {path}")

    def write(self, record: Dict[str, Any]) -> None:
        """Write a single record."""
        if not self._header_written:
            if self._columns is None:
                self._columns = list(record.keys())
            self._writer.writerow(self._columns)
            self._header_written = True

        values = [str(record.get(c, "")) for c in self._columns]
        self._writer.writerow(values)
        self._fp.flush()

    def write_many(self, records: Iterable[Dict[str, Any]]) -> None:
        """Write multiple records."""
        for rec in records:
            self.write(rec)

    def close(self) -> None:
        """Close the sink."""
        try:
            self._fp.close()
            log.info(f"CsvSink closed: {self._path}")
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Optional Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class ParquetSink:
    """
    Parquet output sink for efficient columnar storage.

    Requires pyarrow to be installed.
    """

    def __init__(self, path: str):
        """
        Initialize Parquet sink.

        Args:
            path: Output file path (.parquet)
        """
        if not PARQUET_AVAILABLE:
            raise ImportError(
                "PyArrow required for Parquet output. "
                "Install with: pip install pyarrow"
            )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path = path
        self._records: list = []
        self._closed = False
        log.info(f"ParquetSink initialized: {path}")

    def write(self, record: Dict[str, Any]) -> None:
        """Buffer a single record."""
        self._records.append(record)

    def write_many(self, records: Iterable[Dict[str, Any]]) -> None:
        """Buffer multiple records."""
        self._records.extend(records)

    def close(self) -> None:
        """Write buffered records and close. Idempotent."""
        if self._closed:
            return
        self._closed = True

        if not self._records:
            log.warning("ParquetSink closed with no records")
            return

        try:
            table = pa.Table.from_pylist(self._records)
            pq.write_table(table, self._path)
            log.info(f"ParquetSink wrote {len(self._records)} records to {self._path}")
        except Exception as e:
            log.error(f"ParquetSink write failed: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def make_sink(path: str):
    """
    Create appropriate sink based on file extension.

    Args:
        path: Output file path

    Returns:
        Appropriate sink object
    """
    if path.endswith(".parquet"):
        if PARQUET_AVAILABLE:
            return ParquetSink(path)
        else:
            log.warning("PyArrow not available, falling back to JSONL")
            return JsonlSink(path.replace(".parquet", ".jsonl"))
    elif path.endswith(".csv"):
        return CsvSink(path)
    else:
        return JsonlSink(path)
