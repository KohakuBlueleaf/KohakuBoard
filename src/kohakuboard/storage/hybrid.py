"""Hybrid storage backend: Three-tier SQLite architecture

Three specialized SQLite implementations for optimal performance:
1. KohakuVault KVault - K-V table with index on K (B+Tree-based disk KV store)
2. KohakuVault ColumnVault - Blob storage with Rust columnar layout management
3. Standard SQLite - Traditional relational tables for metadata

Architecture:
- Metrics/Histograms: ColumnVault (blob-based columnar, Rust-managed chunks)
- Media blobs: KVault (K-V table with B+Tree index, content-addressable)
- Metadata: Standard SQLite (traditional relational tables)

Why this design?
- All use SQLite but with specialized implementations
- KVault: Efficient B+Tree-based KV store with .cache() for bulk ops
- ColumnVault: Columnar layout in blobs, managed by Rust for performance
- Standard SQLite: Traditional ACID tables for structured metadata
- Single dependency (kohakuvault), no external services
- Rust performance with Pythonic API
- Simple deployment (just .db files, no infrastructure)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kohakuboard.logger import get_logger
from kohakuboard.storage.columnar import ColumnVaultMetricsStorage
from kohakuboard.storage.columnar_histogram import ColumnVaultHistogramStorage
from kohakuboard.storage.sqlite import SQLiteMetadataStorage
from kohakuboard.storage.tensor import TensorKVStorage


class HybridStorage:
    """Hybrid storage: Three-tier SQLite architecture

    Powered by KohakuVault (https://github.com/KohakuBlueleaf/KohakuVault):

    1. KohakuVault KVault (K-V Store):
       - K-V table with index on K → B+Tree-based disk KV store
       - Used for: Media blobs (content-addressable storage)
       - Benefits: .cache() for bulk ops, efficient key lookups

    2. KohakuVault ColumnVault (Columnar Storage):
       - Blobs managed by Rust for columnar layout
       - Used for: Metrics (per-metric DBs), Histograms (namespace-grouped)
       - Benefits: Fast .extend(), true SWMR, dynamic chunking

    3. Standard SQLite (Relational):
       - Traditional SQLite tables
       - Used for: Metadata (media/table metadata, step info)
       - Benefits: ACID guarantees, standard SQL queries

    Why this design?
    - All three use SQLite, but with different specializations
    - KVault: Optimized for KV lookups (B+Tree index)
    - ColumnVault: Optimized for append-heavy time-series (Rust-managed blobs)
    - Standard SQLite: Optimized for structured metadata (relational)
    - Single dependency, no external services, simple deployment
    """

    def __init__(self, base_dir: Path, logger=None):
        """Initialize hybrid storage

        Args:
            base_dir: Base directory for all storage files
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

        # Initialize sub-storages (pass logger to them)
        self.metrics_storage = ColumnVaultMetricsStorage(base_dir, logger=self.logger)
        self.metadata_storage = SQLiteMetadataStorage(base_dir, logger=self.logger)
        self.histogram_storage = ColumnVaultHistogramStorage(
            base_dir, num_bins=64, logger=self.logger
        )
        self.tensor_storage = TensorKVStorage(base_dir, logger=self.logger)

        self.logger.debug(
            "Hybrid storage initialized (Three-tier SQLite: KVault + ColumnVault + Standard)"
        )

    def append_metrics(
        self,
        step: int,
        global_step: int | None,
        metrics: dict[str, Any],
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

        # Store metrics in ColumnVault
        self.metrics_storage.append_metrics(step, global_step, metrics, timestamp)

    def append_media(
        self,
        step: int,
        global_step: int | None,
        name: str,
        media_list: list[dict[str, Any]],
        caption: str | None = None,
    ) -> list[int]:
        """Append media log entry with deduplication

        NEW in v0.2.0: Returns list of media IDs for reference in tables.

        Args:
            step: Auto-increment step
            global_step: Explicit global step
            name: Media log name
            media_list: List of media metadata dicts (from media_handler.process_media())
            caption: Optional caption

        Returns:
            List of media IDs (SQLite auto-increment IDs)
        """
        # Record step info (use current timestamp)
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)

        # Delegate to SQLite metadata storage (now returns IDs)
        return self.metadata_storage.append_media(
            step, global_step, name, media_list, caption
        )

    def append_table(
        self,
        step: int,
        global_step: int | None,
        name: str,
        table_data: dict[str, Any],
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
        global_step: int | None,
        name: str,
        values: list[float] | None = None,
        num_bins: int = 64,
        precision: str = "compact",
        bins: list[float] | None = None,
        counts: list[int] | None = None,
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

    def append_tensor(
        self,
        step: int,
        global_step: int | None,
        name: str,
        payload: bytes,
        tensor_meta: dict[str, Any],
    ):
        """Append tensor payload to KVault + metadata table."""
        namespace = name.split("/", 1)[0] if "/" in name else name
        namespace_file, kv_key, size_bytes = self.tensor_storage.store_tensor(
            name, step, payload
        )

        self.metadata_storage.append_step_info(
            step, global_step, int(datetime.now(timezone.utc).timestamp() * 1000)
        )

        self.metadata_storage.append_tensor_metadata(
            step,
            global_step,
            name,
            namespace,
            namespace_file,
            kv_key,
            tensor_meta,
            size_bytes,
        )

    def append_kernel_density(
        self,
        step: int,
        global_step: int | None,
        name: str,
        payload: bytes,
        kde_meta: dict[str, Any],
    ):
        """Append kernel density coefficients."""
        namespace_file, kv_key, _ = self.tensor_storage.store_tensor(
            name, step, payload
        )

        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)

        num_points = kde_meta.get("num_points")
        if not num_points and "num_points" in kde_meta:
            num_points = int(kde_meta["num_points"])

        self.metadata_storage.append_kernel_density_metadata(
            step,
            global_step,
            name,
            namespace_file,
            kv_key,
            kde_meta,
            num_points=num_points or 0,
        )

    def get_latest_step(self) -> dict[str, Any] | None:
        """Return latest step info from metadata storage."""
        return self.metadata_storage.get_latest_step()

    def flush_metrics(self):
        """Flush metrics buffer to ColumnVault"""
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
        self.logger.info("Flushed all buffers (hybrid storage)")

    def close(self):
        """Close all storage backends"""
        self.metrics_storage.close()
        self.metadata_storage.close()
        self.histogram_storage.close()
        self.tensor_storage.close()
        self.logger.debug("Hybrid storage closed")
