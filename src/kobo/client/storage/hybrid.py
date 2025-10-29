"""Hybrid storage backend: Lance for metrics + SQLite for metadata

Combines the best of both worlds:
- Lance: Dynamic schema, efficient columnar storage for metrics
- SQLite: Fixed schema, excellent concurrency for media/tables
- Adaptive histograms: Lance with percentile-based range tracking
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from kohakuboard.client.storage.histogram import HistogramStorage
from kohakuboard.client.storage.lance import LanceMetricsStorage
from kohakuboard.client.storage.sqlite import SQLiteMetadataStorage


class HybridStorage:
    """Hybrid storage: Lance for metrics + SQLite for metadata

    Architecture:
    - Metrics (scalars): Lance columnar format (dynamic schema)
    - Media: SQLite (fixed schema, good for metadata)
    - Tables: SQLite (fixed schema)
    - Histograms: Skipped for now (wandb doesn't log locally)

    Benefits:
    - Fast metric writes (Lance batch append)
    - Fast media/table writes (SQLite autocommit)
    - No connection overhead
    - Multi-connection friendly (SQLite WAL mode)
    """

    def __init__(self, base_dir: Path):
        """Initialize hybrid storage

        Args:
            base_dir: Base directory for all storage files
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-storages
        self.metrics_storage = LanceMetricsStorage(base_dir)
        self.metadata_storage = SQLiteMetadataStorage(base_dir)
        self.histogram_storage = HistogramStorage(base_dir, num_bins=64)

        logger.debug("Hybrid storage initialized (Lance + SQLite + Histograms)")

    def append_metrics(
        self,
        step: int,
        global_step: Optional[int],
        metrics: Dict[str, Any],
        timestamp: Any,
    ):
        """Append scalar metrics

        Args:
            step: Auto-increment step
            global_step: Explicit global step
            metrics: Dict of metric name -> value
            timestamp: Timestamp (datetime object)
        """
        # Convert timestamp to ms
        if hasattr(timestamp, "timestamp"):
            timestamp_ms = int(timestamp.timestamp() * 1000)
        else:
            timestamp_ms = int(timestamp * 1000) if timestamp else None

        # Store step info in SQLite for base column queries
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)

        # Store metrics in Lance
        self.metrics_storage.append_metrics(step, global_step, metrics, timestamp)

    def append_media(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        media_list: List[Dict[str, Any]],
        caption: Optional[str] = None,
    ):
        """Append media log entry

        Args:
            step: Auto-increment step
            global_step: Explicit global step
            name: Media log name
            media_list: List of media metadata dicts
            caption: Optional caption
        """
        # Record step info (use current timestamp)
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)

        self.metadata_storage.append_media(step, global_step, name, media_list, caption)

    def append_table(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        table_data: Dict[str, Any],
    ):
        """Append table log entry

        Args:
            step: Auto-increment step
            global_step: Explicit global step
            name: Table log name
            table_data: Table dict
        """
        # Record step info
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)

        self.metadata_storage.append_table(step, global_step, name, table_data)

    def append_histogram(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        values: Optional[List[float]] = None,
        num_bins: int = 64,
        precision: str = "compact",
        bins: Optional[List[float]] = None,
        counts: Optional[List[int]] = None,
    ):
        """Append histogram with configurable precision

        Args:
            step: Step number
            global_step: Global step
            name: Histogram name (e.g., "gradients/layer1")
            values: Raw values array (if not precomputed)
            num_bins: Number of bins
            precision: "compact" (uint8) or "exact" (int32)
            bins: Precomputed bin edges (optional)
            counts: Precomputed bin counts (optional)
        """
        # Record step info
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)

        self.histogram_storage.append_histogram(
            step,
            global_step,
            name,
            values,
            num_bins,
            precision,
            bins=bins,
            counts=counts,
        )

    def flush_metrics(self):
        """Flush metrics buffer to Lance"""
        self.metrics_storage.flush()

    def flush_media(self):
        """Flush media buffer"""
        self.metadata_storage._flush_media()

    def flush_tables(self):
        """Flush tables buffer"""
        self.metadata_storage._flush_tables()

    def flush_histograms(self):
        """Flush histogram buffer"""
        self.histogram_storage.flush()

    def flush_all(self):
        """Flush all buffers"""
        self.flush_metrics()
        self.metadata_storage._flush_steps()  # CRITICAL: Flush step info!
        self.flush_media()
        self.flush_tables()
        self.flush_histograms()
        logger.info("Flushed all buffers (hybrid storage)")

    def close(self):
        """Close all storage backends"""
        self.metrics_storage.close()
        self.metadata_storage.close()
        self.histogram_storage.close()
        logger.debug("Hybrid storage closed")
