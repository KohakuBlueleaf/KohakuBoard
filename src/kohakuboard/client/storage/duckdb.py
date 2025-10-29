"""Storage backend using DuckDB for true incremental appends

NEW ARCHITECTURE (Multi-File + Short-Lived Connections):
- 4 separate DuckDB files (metrics, media, tables, histograms)
- Connection-per-operation pattern (open → write → close)
- Enables concurrent read/write (connections released between batches)
- Retry logic for transient lock conflicts
- Heavy logging isolated (separate files/queues)
"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
from loguru import logger


class DuckDBStorage:
    """DuckDB-based storage with multi-file architecture

    Architecture:
    - metrics.duckdb: Scalar metrics (step, global_step, timestamp, dynamic columns)
    - media.duckdb: Media metadata (images, videos, audio)
    - tables.duckdb: Table logs
    - histograms.duckdb: Histogram data

    Benefits:
    - Concurrent read/write (different files)
    - Heavy logging isolation (separate queues/files)
    - True incremental append
    - NaN/inf preservation (direct INSERT)
    """

    def __init__(self, base_dir: Path):
        """Initialize DuckDB storage with multi-file architecture

        Args:
            base_dir: Base directory for database files
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Separate database files (one per data type)
        self.metrics_file = base_dir / "metrics.duckdb"
        self.media_file = base_dir / "media.duckdb"
        self.tables_file = base_dir / "tables.duckdb"
        self.histograms_file = base_dir / "histograms.duckdb"

        # NO persistent connections - use connection-per-operation pattern
        # This enables concurrent read access between write batches

        # Initialize tables on first write (lazy initialization)
        self.tables_initialized = False

        # In-memory buffers (one per data type)
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.media_buffer: List[Dict[str, Any]] = []
        self.tables_buffer: List[Dict[str, Any]] = []
        self.histograms_buffer: List[Dict[str, Any]] = []

        # Track known columns for schema evolution
        self.known_metric_cols = {"step", "global_step", "timestamp"}

        # Flush thresholds (BALANCED)
        # Balance between connection overhead and queue responsiveness
        self.flush_threshold = 50  # Metrics (reasonable batch size)
        self.media_flush_threshold = 20  # Media (smaller batches, less frequent)
        self.tables_flush_threshold = 20  # Tables (smaller batches)
        self.histogram_flush_threshold = 100  # Histograms (can batch more)

        # Retry configuration
        self.max_retries = 5
        self.retry_delay = 0.1  # Initial delay in seconds

    def _connect_with_retry(
        self, db_file: Path, read_only: bool = False
    ) -> duckdb.DuckDBPyConnection:
        """Open DuckDB connection with retry logic for lock conflicts

        Args:
            db_file: Path to database file
            read_only: Whether to open in read-only mode

        Returns:
            DuckDB connection

        Raises:
            Exception: If all retries exhausted
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                return duckdb.connect(str(db_file), read_only=read_only)
            except duckdb.IOException as e:
                # IOException = file lock conflict, retry
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (
                        2 ** (attempt - 1)
                    )  # Exponential backoff
                    logger.debug(
                        f"Write locked (IOException), retry {attempt}/{self.max_retries} after {delay:.2f}s: {db_file.name}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed to connect to {db_file.name} after {self.max_retries} retries: {e}"
                    )
                    raise
            except Exception as e:
                # Other errors, don't retry
                logger.error(
                    f"Non-lock error opening {db_file.name}: {type(e).__name__}: {e}"
                )
                raise

        # Should never reach here
        raise last_error

    def _ensure_tables_initialized(self):
        """Ensure database tables are initialized (lazy initialization)"""
        if self.tables_initialized:
            return

        # Initialize each database file with its table schema
        # Use short-lived connections

        # Metrics table
        conn = self._connect_with_retry(self.metrics_file)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    step BIGINT NOT NULL,
                    global_step BIGINT,
                    timestamp TIMESTAMP NOT NULL
                )
            """
            )
            conn.commit()
        finally:
            conn.close()

        # Media table
        conn = self._connect_with_retry(self.media_file)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS media (
                    step BIGINT NOT NULL,
                    global_step BIGINT,
                    name VARCHAR NOT NULL,
                    caption VARCHAR,
                    media_id VARCHAR,
                    type VARCHAR,
                    filename VARCHAR,
                    path VARCHAR,
                    size_bytes BIGINT,
                    format VARCHAR,
                    width INTEGER,
                    height INTEGER
                )
            """
            )
            conn.commit()
        finally:
            conn.close()

        # Tables table
        conn = self._connect_with_retry(self.tables_file)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tables (
                    step BIGINT NOT NULL,
                    global_step BIGINT,
                    name VARCHAR NOT NULL,
                    columns VARCHAR,
                    column_types VARCHAR,
                    rows VARCHAR
                )
            """
            )
            conn.commit()
        finally:
            conn.close()

        # Histograms table
        conn = self._connect_with_retry(self.histograms_file)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS histograms (
                    step BIGINT NOT NULL,
                    global_step BIGINT,
                    name VARCHAR NOT NULL,
                    num_bins INTEGER,
                    bins VARCHAR,
                    counts VARCHAR
                )
            """
            )
            conn.commit()
        finally:
            conn.close()

        self.tables_initialized = True
        logger.debug("Database tables initialized")

    def append_metrics(
        self,
        step: int,
        global_step: Optional[int],
        metrics: Dict[str, Any],
        timestamp: Any,
    ):
        """Append metrics for a step

        Args:
            step: Auto-increment step
            global_step: Explicit global step (optional)
            metrics: Dict of metric name -> value (can contain "/" for namespaces)
            timestamp: Timestamp of log event (datetime object)
        """
        # Lazy initialization on first write
        self._ensure_tables_initialized()

        # Escape metric names (replace "/" with "__" for DuckDB)
        escaped_metrics = {k.replace("/", "__"): v for k, v in metrics.items()}

        row = {
            "step": step,
            "global_step": global_step,
            "timestamp": timestamp,
            **escaped_metrics,
        }
        self.metrics_buffer.append(row)

        # Check for new columns and add them to schema
        new_cols = set(metrics.keys()) - self.known_metric_cols
        if new_cols:
            self._add_metric_columns(new_cols)

        # Flush when threshold reached
        if len(self.metrics_buffer) >= self.flush_threshold:
            self.flush_metrics()

    def _add_metric_columns(self, new_cols: set):
        """Add new metric columns to metrics table

        Args:
            new_cols: Set of new column names
        """
        if not new_cols:
            return

        # Open connection, add columns, close
        conn = self._connect_with_retry(self.metrics_file)
        try:
            for col in new_cols:
                try:
                    # Escape column name (replace "/" with "__" for DuckDB compatibility)
                    escaped_col = col.replace("/", "__")
                    # Add column as DOUBLE (works for most ML metrics)
                    conn.execute(
                        f'ALTER TABLE metrics ADD COLUMN IF NOT EXISTS "{escaped_col}" DOUBLE'
                    )
                    self.known_metric_cols.add(col)
                    logger.debug(f"Added new metric column: {col} (as {escaped_col})")
                except Exception as e:
                    logger.error(f"Failed to add column {col}: {e}")

            conn.commit()
        finally:
            conn.close()

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
        for media_meta in media_list:
            row = {
                "step": step,
                "global_step": global_step,
                "name": name,
                "caption": caption or "",
                **media_meta,
            }
            self.media_buffer.append(row)

        # Batch flush when threshold reached
        if len(self.media_buffer) >= self.media_flush_threshold:
            self.flush_media()

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
            table_data: Table dict with columns, column_types, rows
        """
        row = {
            "step": step,
            "global_step": global_step,
            "name": name,
            "columns": json.dumps(table_data["columns"]),
            "column_types": json.dumps(table_data["column_types"]),
            "rows": json.dumps(table_data["rows"]),
        }
        self.tables_buffer.append(row)

        # Batch flush when threshold reached
        if len(self.tables_buffer) >= self.tables_flush_threshold:
            self.flush_tables()

    def flush_metrics(self):
        """Flush metrics buffer to DuckDB (preserves NaN/inf, short-lived connection)"""
        if not self.metrics_buffer:
            return

        # Open connection with retry
        conn = self._connect_with_retry(self.metrics_file)
        try:
            # Direct SQL INSERT to preserve NaN/inf as IEEE 754 values (not NULL)
            for row in self.metrics_buffer:
                columns = list(row.keys())
                values = list(row.values())

                col_names = ", ".join(f'"{col}"' for col in columns)
                placeholders = ", ".join("?" * len(columns))
                query = f"INSERT INTO metrics ({col_names}) VALUES ({placeholders})"

                conn.execute(query, values)

            conn.commit()

            logger.debug(
                f"Appended {len(self.metrics_buffer)} metrics rows (preserving NaN/inf)"
            )
            self.metrics_buffer.clear()

        except KeyboardInterrupt:
            logger.warning("Metrics flush interrupted")
            self.metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            logger.exception(e)
        finally:
            conn.close()  # CRITICAL: Always close connection

    def flush_media(self):
        """Flush media buffer to DuckDB (direct INSERT, short-lived connection)"""
        if not self.media_buffer:
            return

        conn = self._connect_with_retry(self.media_file)
        try:
            # Direct SQL INSERT
            for row in self.media_buffer:
                columns = list(row.keys())
                values = list(row.values())

                col_names = ", ".join(f'"{col}"' for col in columns)
                placeholders = ", ".join("?" * len(columns))
                query = f"INSERT INTO media ({col_names}) VALUES ({placeholders})"

                conn.execute(query, values)

            conn.commit()

            logger.debug(f"Appended {len(self.media_buffer)} media rows")
            self.media_buffer.clear()

        except KeyboardInterrupt:
            logger.warning("Media flush interrupted")
            self.media_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush media: {e}")
            logger.exception(e)
        finally:
            conn.close()

    def flush_tables(self):
        """Flush tables buffer to DuckDB (direct INSERT, short-lived connection)"""
        if not self.tables_buffer:
            return

        conn = self._connect_with_retry(self.tables_file)
        try:
            # Direct SQL INSERT
            for row in self.tables_buffer:
                columns = list(row.keys())
                values = list(row.values())

                col_names = ", ".join(f'"{col}"' for col in columns)
                placeholders = ", ".join("?" * len(columns))
                query = f"INSERT INTO tables ({col_names}) VALUES ({placeholders})"

                conn.execute(query, values)

            conn.commit()

            logger.debug(f"Appended {len(self.tables_buffer)} table rows")
            self.tables_buffer.clear()

        except KeyboardInterrupt:
            logger.warning("Tables flush interrupted")
            self.tables_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush tables: {e}")
            logger.exception(e)
        finally:
            conn.close()

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
        """Append histogram log entry (pre-computed bins to save space)

        Args:
            step: Auto-increment step
            global_step: Explicit global step
            name: Histogram log name
            values: List of values to create histogram from (if not precomputed)
            num_bins: Number of bins for histogram
            precision: Ignored for DuckDB backend
            bins: Precomputed bin edges (optional)
            counts: Precomputed bin counts (optional)
        """
        # Check if precomputed
        if bins is not None and counts is not None:
            # Use precomputed bins/counts
            bin_edges = np.array(bins, dtype=np.float32)
            counts_array = np.array(counts, dtype=np.int32)
            num_bins = len(counts_array)
        else:
            # Compute histogram (bins + counts) instead of storing raw values
            values_array = np.array(values, dtype=np.float32)
            counts_array, bin_edges = np.histogram(values_array, bins=num_bins)

        row = {
            "step": step,
            "global_step": global_step,
            "name": name,
            "num_bins": num_bins,
            "bins": json.dumps(bin_edges.tolist()),  # Bin edges
            "counts": json.dumps(counts_array.tolist()),  # Counts per bin
        }
        self.histograms_buffer.append(row)

        # Batch flush when threshold reached (not immediate!)
        if len(self.histograms_buffer) >= self.histogram_flush_threshold:
            self.flush_histograms()

    def flush_histograms(self):
        """Flush histograms buffer to DuckDB (direct INSERT, short-lived connection)"""
        if not self.histograms_buffer:
            return

        conn = self._connect_with_retry(self.histograms_file)
        try:
            # Direct SQL INSERT
            for row in self.histograms_buffer:
                columns = list(row.keys())
                values = list(row.values())

                col_names = ", ".join(f'"{col}"' for col in columns)
                placeholders = ", ".join("?" * len(columns))
                query = f"INSERT INTO histograms ({col_names}) VALUES ({placeholders})"

                conn.execute(query, values)

            conn.commit()

            logger.debug(f"Appended {len(self.histograms_buffer)} histogram rows")
            self.histograms_buffer.clear()

        except KeyboardInterrupt:
            logger.warning("Histograms flush interrupted")
            self.histograms_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush histograms: {e}")
            logger.exception(e)
        finally:
            conn.close()

    def flush_all(self):
        """Flush all buffers"""
        self.flush_metrics()
        self.flush_media()
        self.flush_tables()
        self.flush_histograms()
        logger.info("Flushed all buffers to DuckDB")

    def close(self):
        """Close storage (no-op with connection-per-operation pattern)

        Connections are opened and closed for each operation,
        so there's nothing to clean up here.
        """
        logger.debug(
            "Storage close called (connection-per-operation: nothing to close)"
        )
