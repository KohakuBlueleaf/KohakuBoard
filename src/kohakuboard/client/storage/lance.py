"""Lance-based storage for scalar metrics (per-metric files)

Uses one Lance file per metric (like MLflow but with Lance instead of text):
- metrics/train__loss.lance
- metrics/val__acc.lance

Fixed schema per file: step, global_step, timestamp, value
No dynamic schema issues!
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
from lance.dataset import write_dataset
from loguru import logger


class LanceMetricsStorage:
    """Lance-based storage with per-metric files (MLflow-style)

    Architecture:
    - One .lance file per metric
    - Fixed schema: step, global_step, timestamp, value
    - No dynamic schema issues
    - Independent writes per metric

    Benefits:
    - Simple schema (no ALTER TABLE)
    - Efficient columnar storage
    - Fast random access
    - Independent metric writes
    """

    def __init__(self, base_dir: Path):
        """Initialize Lance metrics storage

        Args:
            base_dir: Base directory for metric files
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_dir = base_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        # Per-metric buffers (one buffer per metric)
        self.buffers: Dict[str, List[Dict[str, Any]]] = {}

        # Per-metric last flush time tracking
        self.last_flush_time: Dict[str, float] = {}

        # Flush configuration - AGGRESSIVE BATCHING
        # SQLite and Lance are efficient - batch as much as possible
        self.flush_threshold = 1000  # Large batch size
        self.flush_interval = 2.0  # Flush every 2 seconds (not too aggressive)

        # Fixed schema for all metrics
        # Use float32 for values (sufficient precision for ML metrics, saves space)
        self.schema = pa.schema(
            [
                pa.field("step", pa.int64()),
                pa.field("global_step", pa.int64()),
                pa.field("timestamp", pa.int64()),
                pa.field("value", pa.float32()),  # float32, not float64
            ]
        )

    def append_metrics(
        self,
        step: int,
        global_step: Optional[int],
        metrics: Dict[str, Any],
        timestamp: Any,
    ):
        """Append metrics for a step (per-metric file approach)

        Args:
            step: Auto-increment step
            global_step: Explicit global step (optional)
            metrics: Dict of metric name -> value (can contain "/" for namespaces)
            timestamp: Timestamp (datetime object)
        """
        # Convert timestamp to Unix milliseconds (integer)
        if hasattr(timestamp, "timestamp"):
            timestamp_ms = int(timestamp.timestamp() * 1000)
        else:
            timestamp_ms = int(timestamp * 1000) if timestamp else None

        # Append to each metric's buffer separately
        for metric_name, value in metrics.items():
            # Escape metric name (replace "/" with "__")
            escaped_name = metric_name.replace("/", "__")

            # Initialize buffer for this metric if needed
            if escaped_name not in self.buffers:
                self.buffers[escaped_name] = []

            # Convert NaN/inf to None (Lance limitation)
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    value = None

            # Append row to this metric's buffer
            self.buffers[escaped_name].append(
                {
                    "step": step,
                    "global_step": global_step,
                    "timestamp": timestamp_ms,
                    "value": value,
                }
            )

            # Initialize last flush time if needed
            if escaped_name not in self.last_flush_time:
                self.last_flush_time[escaped_name] = time.time()

            # Don't auto-flush - writer will call flush() periodically
            # This allows batching ALL pending data at once

    def _flush_metric(self, metric_name: str):
        """Flush a single metric's buffer to its Lance file

        Args:
            metric_name: Escaped metric name
        """
        if metric_name not in self.buffers or not self.buffers[metric_name]:
            return

        try:
            # Create Arrow table from buffer (fixed schema!)
            table = pa.Table.from_pylist(self.buffers[metric_name], schema=self.schema)

            # Metric file path
            metric_file = self.metrics_dir / f"{metric_name}.lance"

            # Write to Lance (append if exists)
            if metric_file.exists():
                write_dataset(table, str(metric_file), mode="append")
            else:
                write_dataset(table, str(metric_file))

            logger.debug(
                f"Flushed {len(self.buffers[metric_name])} rows to {metric_name}.lance"
            )
            self.buffers[metric_name].clear()

            # Update last flush time
            self.last_flush_time[metric_name] = time.time()

        except Exception as e:
            logger.error(f"Failed to flush metric '{metric_name}' to Lance: {e}")
            logger.exception(e)

    def flush(self):
        """Flush all metric buffers"""
        for metric_name in list(self.buffers.keys()):
            self._flush_metric(metric_name)

    def close(self):
        """Close storage - flush all remaining buffers"""
        self.flush()
        logger.debug("Lance metrics storage closed")
