"""KohakuVault ColumnVault-based storage for scalar metrics (per-metric files)

Uses one ColumnVault DB per metric (like Lance but with SQLite):
- metrics/train__loss.db
- metrics/val__acc.db

Fixed schema per file: step, global_step, timestamp, value
True SWMR with SQLite WAL mode!
"""

import math
import time
from pathlib import Path
from typing import Any

from kohakuvault import ColumnVault

from kohakuboard.logger import get_logger


class ColumnVaultMetricsStorage:
    """ColumnVault-based storage with per-metric files

    Architecture:
    - One .db file per metric (SQLite with ColumnVault)
    - Fixed columns: step, global_step, timestamp, value
    - WAL mode for true SWMR (Single-Writer-Multiple-Reader)
    - Independent writes per metric

    Benefits:
    - Single file per metric (simpler than Lance)
    - True SWMR with SQLite WAL mode
    - Efficient columnar storage
    - Fast bulk appends with .extend()
    """

    def __init__(self, base_dir: Path, logger=None):
        """Initialize ColumnVault metrics storage

        Args:
            base_dir: Base directory for metric files
            logger: Optional logger instance (if None, creates file-only logger)
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Optional logger injection
        if logger is not None:
            self.logger = logger
        else:
            # Default: file-only logger (for client/writer)
            log_file = base_dir.parent / "logs" / "storage.log"
            self.logger = get_logger("STORAGE", file_only=True, log_file=log_file)

        self.metrics_dir = base_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        # Cache of ColumnVault instances (keep open for performance)
        self.vaults: dict[str, ColumnVault] = {}

        # Per-metric buffers (one buffer per metric)
        self.buffers: dict[str, list[dict[str, Any]]] = {}

        # Per-metric last flush time tracking
        self.last_flush_time: dict[str, float] = {}

        # Flush configuration - AGGRESSIVE BATCHING
        self.flush_threshold = 1000  # Large batch size
        self.flush_interval = 2.0  # Flush every 2 seconds

    def _get_or_create_vault(self, metric_name: str) -> ColumnVault:
        """Get or create ColumnVault instance for a metric

        Args:
            metric_name: Escaped metric name

        Returns:
            ColumnVault instance
        """
        if metric_name not in self.vaults:
            db_path = self.metrics_dir / f"{metric_name}.db"

            # Check if database file exists before creating ColumnVault
            is_new_db = not db_path.exists()

            # Create ColumnVault (WAL is default in SQLite)
            cv = ColumnVault(str(db_path))

            # Create columns only if database is new
            if is_new_db:
                # New database - create schema
                cv.create_column("step", "i64")
                cv.create_column("global_step", "i64")
                cv.create_column("timestamp", "i64")
                cv.create_column("value", "f64")  # Use f64 for better precision

            self.vaults[metric_name] = cv
            self.logger.debug(f"Opened ColumnVault for metric: {metric_name}")

        return self.vaults[metric_name]

    def append_metrics(
        self,
        step: int,
        global_step: int | None,
        metrics: dict[str, Any],
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

            # Convert NaN/inf to 0.0 (SQLite f64 might handle NaN, but safer to normalize)
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    value = 0.0  # Use 0.0 instead of None for numeric column

            # Append row to this metric's buffer
            self.buffers[escaped_name].append(
                {
                    "step": step,
                    "global_step": global_step if global_step is not None else step,
                    "timestamp": timestamp_ms,
                    "value": float(value),
                }
            )

            # Initialize last flush time if needed
            if escaped_name not in self.last_flush_time:
                self.last_flush_time[escaped_name] = time.time()

    def _flush_metric(self, metric_name: str):
        """Flush a single metric's buffer to its ColumnVault DB

        Args:
            metric_name: Escaped metric name
        """
        if metric_name not in self.buffers or not self.buffers[metric_name]:
            return

        try:
            cv = self._get_or_create_vault(metric_name)
            buffer = self.buffers[metric_name]

            # Extract columns for batch extend
            steps = [row["step"] for row in buffer]
            global_steps = [row["global_step"] for row in buffer]
            timestamps = [row["timestamp"] for row in buffer]
            values = [row["value"] for row in buffer]

            # Batch append with .extend() for efficiency
            cv["step"].extend(steps)
            cv["global_step"].extend(global_steps)
            cv["timestamp"].extend(timestamps)
            cv["value"].extend(values)

            self.logger.debug(f"Flushed {len(buffer)} rows to {metric_name}.db")
            buffer.clear()

            # Update last flush time
            self.last_flush_time[metric_name] = time.time()

        except Exception as e:
            self.logger.error(
                f"Failed to flush metric '{metric_name}' to ColumnVault: {e}"
            )

    def flush(self):
        """Flush all metric buffers"""
        for metric_name in list(self.buffers.keys()):
            self._flush_metric(metric_name)

    def close(self):
        """Close storage - flush all remaining buffers and close vaults"""
        self.flush()

        # Close all ColumnVault instances
        for metric_name, cv in self.vaults.items():
            try:
                cv.close()
            except:
                pass  # Ignore close errors

        self.vaults.clear()
        self.logger.debug("ColumnVault metrics storage closed")
