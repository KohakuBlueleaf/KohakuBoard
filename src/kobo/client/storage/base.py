"""Storage backend using Parquet for efficient columnar storage"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


class ParquetStorage:
    """Parquet-based storage for metrics, media, and table logs

    Uses PyArrow ParquetWriter for true incremental appends (no read-rewrite).
    """

    def __init__(self, base_dir: Path):
        """Initialize Parquet storage

        Args:
            base_dir: Base directory for all parquet files
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.metrics_file = self.base_dir / "metrics.parquet"
        self.media_file = self.base_dir / "media.parquet"
        self.tables_file = self.base_dir / "tables.parquet"

        # In-memory buffers for batching
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.media_buffer: List[Dict[str, Any]] = []
        self.tables_buffer: List[Dict[str, Any]] = []

        # Schema tracking for optimization (only rewrite when schema changes)
        self.metrics_columns: Optional[set] = None

        # Flush thresholds - lower for better online sync
        # Best-effort flush: flush often to make data available quickly
        self.flush_threshold = 10  # Flush every 10 rows (aggressive for online sync)

    def append_metrics(
        self, step: int, global_step: Optional[int], metrics: Dict[str, Any]
    ):
        """Append metrics for a step

        Args:
            step: Auto-increment step
            global_step: Explicit global step (optional)
            metrics: Dict of metric name -> value
        """
        row = {
            "step": step,
            "global_step": global_step,
            **metrics,
        }
        self.metrics_buffer.append(row)

        # Check if schema changed (new columns)
        current_cols = set(row.keys())
        schema_changed = (
            self.metrics_columns is not None and current_cols != self.metrics_columns
        )

        # Flush immediately if:
        # 1. Schema changed (need to rewrite file with new columns)
        # 2. Buffer threshold reached
        if schema_changed or len(self.metrics_buffer) >= self.flush_threshold:
            self.flush_metrics()
            self.metrics_columns = current_cols

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
            # Store step/global_step + all metadata as flattened columns
            row = {
                "step": step,
                "global_step": global_step,
                "name": name,
                "caption": caption or "",
                **media_meta,  # Flatten metadata into columns (type, path, filename, width, height, etc.)
            }
            self.media_buffer.append(row)

        # Best-effort flush
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

        # Best-effort flush
        self.flush_tables()

    def flush_metrics(self):
        """Flush metrics buffer to Parquet file

        Hybrid approach:
        - Same schema: Use PyArrow for true incremental append (fast!)
        - Schema changed: Use pandas read-concat-write (slow but handles evolution)
        """
        if not self.metrics_buffer:
            return

        try:
            # Convert buffer to Arrow table
            df_new = pd.DataFrame(self.metrics_buffer)
            table_new = pa.Table.from_pandas(df_new, preserve_index=False)
            new_cols = set(table_new.column_names)

            if self.metrics_file.exists():
                # Read existing schema (metadata only, no data read!)
                existing_schema = pq.read_schema(self.metrics_file)
                existing_cols = set(existing_schema.names)

                # Check if schema is compatible
                if new_cols == existing_cols:
                    # SAME SCHEMA - Read existing and append
                    # Note: PyArrow doesn't support true append, need to read old data
                    # But at least we skip schema merging logic
                    df_old = pd.read_parquet(self.metrics_file)
                    df = pd.concat([df_old, df_new], ignore_index=True)

                    df.to_parquet(
                        self.metrics_file,
                        engine="pyarrow",
                        compression="snappy",
                        index=False,
                    )

                    logger.debug(
                        f"Flushed {len(self.metrics_buffer)} metrics rows (same schema, total: {len(df)})"
                    )
                else:
                    # SCHEMA CHANGED - Need to rewrite whole file
                    logger.info(
                        f"Schema changed: {existing_cols} -> {new_cols}, rewriting file..."
                    )
                    df_old = pd.read_parquet(self.metrics_file)
                    df = pd.concat([df_old, df_new], ignore_index=True, sort=False)
                    df.to_parquet(
                        self.metrics_file,
                        engine="pyarrow",
                        compression="snappy",
                        index=False,
                    )
                    logger.debug(
                        f"Flushed {len(self.metrics_buffer)} metrics rows (FULL REWRITE, total: {len(df)})"
                    )
            else:
                # Create new file
                pq.write_table(table_new, self.metrics_file, compression="snappy")
                logger.debug(
                    f"Created metrics file with {len(self.metrics_buffer)} rows"
                )

            self.metrics_buffer.clear()

        except KeyboardInterrupt:
            logger.warning("Flush interrupted, data in buffer may be lost")
            self.metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            logger.exception(e)

    def flush_media(self):
        """Flush media buffer to Parquet file"""
        if not self.media_buffer:
            return

        try:
            df_new = pd.DataFrame(self.media_buffer)

            if self.media_file.exists():
                df_old = pd.read_parquet(self.media_file)
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new

            df.to_parquet(
                self.media_file,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )

            logger.debug(f"Flushed {len(self.media_buffer)} media rows")
            self.media_buffer.clear()

        except KeyboardInterrupt:
            logger.warning("Media flush interrupted, data in buffer may be lost")
            self.media_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush media: {e}")

    def flush_tables(self):
        """Flush tables buffer to Parquet file"""
        if not self.tables_buffer:
            return

        try:
            df_new = pd.DataFrame(self.tables_buffer)

            if self.tables_file.exists():
                df_old = pd.read_parquet(self.tables_file)
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new

            df.to_parquet(
                self.tables_file,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )

            logger.debug(f"Flushed {len(self.tables_buffer)} table rows")
            self.tables_buffer.clear()

        except KeyboardInterrupt:
            logger.warning("Tables flush interrupted, data in buffer may be lost")
            self.tables_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush tables: {e}")

    def flush_all(self):
        """Flush all buffers"""
        self.flush_metrics()
        self.flush_media()
        self.flush_tables()
        logger.info("Flushed all buffers to disk")
