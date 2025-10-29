"""Utility functions for reading board data from DuckDB files

NEW: Connection-per-operation pattern with retry logic
- Opens read-only connection when needed
- Closes immediately after read
- Retries on lock conflicts (enables reading while writer is active)

HYBRID BACKEND: Auto-detects Lance+SQLite format and delegates
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from loguru import logger

from kohakuboard.api.utils.board_reader_hybrid import HybridBoardReader


class BoardReader:
    """Read-only interface for accessing board data (auto-detects backend)"""

    def __new__(cls, board_dir: Path):
        """Factory method - returns appropriate reader based on detected backend

        Args:
            board_dir: Path to board directory

        Returns:
            HybridBoardReader, or DuckDBBoardReader instance
        """
        board_dir = Path(board_dir)

        # Detect backend type
        metrics_lance_dir = board_dir / "data" / "metrics"  # Per-metric Lance files
        sqlite_db = board_dir / "data" / "metadata.db"
        metrics_duckdb = board_dir / "data" / "metrics.duckdb"
        legacy_duckdb = board_dir / "data" / "board.duckdb"

        # Priority: hybrid > multi-file duckdb > legacy duckdb > parquet
        if metrics_lance_dir.exists() or sqlite_db.exists():
            logger.debug(f"Detected hybrid backend for {board_dir.name}")
            return HybridBoardReader(board_dir)
        else:
            logger.debug(f"Detected DuckDB backend for {board_dir.name}")
            return DuckDBBoardReader.__new__(DuckDBBoardReader, board_dir)


class DuckDBBoardReader:
    """DuckDB-based board reader (for backward compatibility)"""

    def __init__(self, board_dir: Path):
        """Initialize DuckDB board reader

        Args:
            board_dir: Path to board directory
        """
        self.board_dir = Path(board_dir)
        self.metadata_path = self.board_dir / "metadata.json"
        self.media_dir = self.board_dir / "media"

        # Multi-file DuckDB structure
        self.metrics_db = self.board_dir / "data" / "metrics.duckdb"
        self.media_db = self.board_dir / "data" / "media.duckdb"
        self.tables_db = self.board_dir / "data" / "tables.duckdb"
        self.histograms_db = self.board_dir / "data" / "histograms.duckdb"

        # Legacy single file
        self.legacy_db = self.board_dir / "data" / "board.duckdb"

        # Determine which structure to use
        self.use_legacy = self.legacy_db.exists() and not self.metrics_db.exists()

        # Validate paths
        if not self.board_dir.exists():
            raise FileNotFoundError(f"Board directory not found: {board_dir}")

        # Retry configuration
        self.max_retries = 5
        self.retry_delay = 0.05

    def get_metadata(self) -> Dict[str, Any]:
        """Get board metadata

        Returns:
            Metadata dict with board info, config, timestamps
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def _connect_with_retry(self, db_path: Path) -> duckdb.DuckDBPyConnection:
        """Open read-only DuckDB connection with retry logic

        Args:
            db_path: Path to database file

        Returns:
            DuckDB connection (read-only)

        Raises:
            Exception: If all retries exhausted
        """
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                return duckdb.connect(str(db_path), read_only=True)
            except duckdb.IOException as e:
                # IOException = file lock conflict, retry
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (
                        2 ** (attempt - 1)
                    )  # Exponential backoff
                    logger.debug(
                        f"Read locked (IOException), retry {attempt}/{self.max_retries} after {delay:.3f}s: {db_path.name}"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"Failed to open {db_path.name} after {self.max_retries} retries: {e}"
                    )
                    raise
            except Exception as e:
                # Other errors, don't retry
                logger.error(
                    f"Non-lock error opening {db_path.name}: {type(e).__name__}: {e}"
                )
                raise

        # Should never reach here
        raise last_error

    def _get_metrics_connection(self) -> duckdb.DuckDBPyConnection:
        """Get read-only connection to metrics database with retry

        Returns:
            DuckDB connection
        """
        db_path = self.legacy_db if self.use_legacy else self.metrics_db
        return self._connect_with_retry(db_path)

    def _get_media_connection(self) -> duckdb.DuckDBPyConnection:
        """Get read-only connection to media database with retry

        Returns:
            DuckDB connection
        """
        db_path = self.legacy_db if self.use_legacy else self.media_db
        return self._connect_with_retry(db_path)

    def _get_tables_connection(self) -> duckdb.DuckDBPyConnection:
        """Get read-only connection to tables database with retry

        Returns:
            DuckDB connection
        """
        db_path = self.legacy_db if self.use_legacy else self.tables_db
        return self._connect_with_retry(db_path)

    def _get_histograms_connection(self) -> duckdb.DuckDBPyConnection:
        """Get read-only connection to histograms database with retry

        Returns:
            DuckDB connection
        """
        db_path = self.legacy_db if self.use_legacy else self.histograms_db
        return self._connect_with_retry(db_path)

    def get_available_metrics(self) -> List[str]:
        """Get list of available scalar metrics

        Returns:
            List of metric names (INCLUDING step/global_step for x-axis selection)
        """
        conn = self._get_metrics_connection()
        try:
            # Get all columns from metrics table
            result = conn.execute("PRAGMA table_info(metrics)").fetchall()
            columns = [row[1] for row in result]  # Column name is at index 1

            # Return all columns (including step, global_step, timestamp for x-axis)
            # step, global_step, timestamp should be first for better UX
            # Also convert "__" back to "/" for namespace support
            axis_cols = [
                col for col in columns if col in ("step", "global_step", "timestamp")
            ]
            other_cols = [
                col.replace("__", "/")  # Convert back to namespace format
                for col in columns
                if col not in ("step", "global_step", "timestamp")
            ]

            return axis_cols + other_cols
        finally:
            conn.close()

    def get_scalar_data(
        self, metric: str, limit: Optional[int] = None
    ) -> Dict[str, List]:
        """Get scalar data for a specific metric

        Args:
            metric: Metric name (can be 'step', 'global_step', or any other metric)
            limit: Optional row limit

        Returns:
            Dict with columnar format: {steps: [], global_steps: [], timestamps: [], values: []}
        """
        conn = self._get_metrics_connection()
        try:
            # Escape metric name (convert "/" to "__" for DuckDB column name)
            escaped_metric = metric.replace("/", "__")

            # Build query - select ALL rows (don't filter out NaN!)
            # This is critical: NaN values are data, not missing data
            query = f'SELECT step, global_step, timestamp, "{escaped_metric}" as value FROM metrics'

            if limit:
                query += f" LIMIT {limit}"

            result = conn.execute(query).fetchall()

            # Convert to columnar format (more efficient than row-based)
            # Format: {steps: [], global_steps: [], timestamps: [], values: []}
            steps = []
            global_steps = []
            timestamps = []
            values = []

            for row in result:
                steps.append(row[0])
                global_steps.append(row[1])
                # Convert timestamp to Unix seconds (integer) for efficiency
                timestamps.append(int(row[2].timestamp()) if row[2] else None)

                value = row[3]
                # Convert special values to string markers for JSON compatibility
                # null = sparse/missing data (not logged at this step)
                # "NaN" = explicitly logged NaN value
                # "Infinity"/"-Infinity" = explicitly logged inf values
                if value is not None:
                    if math.isnan(value):
                        value = "NaN"
                    elif math.isinf(value):
                        value = "Infinity" if value > 0 else "-Infinity"

                values.append(value)

            return {
                "steps": steps,
                "global_steps": global_steps,
                "timestamps": timestamps,
                "values": values,
            }
        finally:
            conn.close()

    def get_available_media_names(self) -> List[str]:
        """Get list of available media log names

        Returns:
            List of unique media log names
        """
        conn = self._get_media_connection()
        try:
            result = conn.execute(
                "SELECT DISTINCT name FROM media ORDER BY name"
            ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.warning(f"Failed to query media table: {e}")
            return []
        finally:
            conn.close()

    def get_media_data(
        self, name: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get media data for a specific log name

        Args:
            name: Media log name
            limit: Optional row limit

        Returns:
            List of dicts with step, global_step, caption, media metadata
            (filename is derived as {media_hash}.{format})
        """
        conn = self._get_media_connection()
        try:
            query = f"SELECT * FROM media WHERE name = ?"
            if limit:
                query += f" LIMIT {limit}"

            result = conn.execute(query, [name]).fetchall()

            # Get column names
            columns = [desc[0] for desc in conn.description]

            # Convert to list of dicts and derive filename
            media_list = []
            for row in result:
                media_dict = dict(zip(columns, row))
                # Derive filename from hash + format (v0.2.0+)
                media_dict["filename"] = (
                    f"{media_dict['media_hash']}.{media_dict['format']}"
                )
                media_list.append(media_dict)

            return media_list
        finally:
            conn.close()

    def get_available_table_names(self) -> List[str]:
        """Get list of available table log names

        Returns:
            List of unique table log names
        """
        conn = self._get_tables_connection()
        try:
            result = conn.execute(
                "SELECT DISTINCT name FROM tables ORDER BY name"
            ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.warning(f"Failed to query tables table: {e}")
            return []
        finally:
            conn.close()

    def get_table_data(
        self, name: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get table data for a specific log name

        Args:
            name: Table log name
            limit: Optional row limit

        Returns:
            List of dicts with step, global_step, columns, column_types, rows
        """
        conn = self._get_tables_connection()
        try:
            query = f"SELECT * FROM tables WHERE name = ?"
            if limit:
                query += f" LIMIT {limit}"

            result = conn.execute(query, [name]).fetchall()

            # Get column names
            columns = [desc[0] for desc in conn.description]

            # Convert to list of dicts, parsing JSON fields
            data = []
            for row in result:
                row_dict = dict(zip(columns, row))

                # Parse JSON fields
                if row_dict.get("columns"):
                    row_dict["columns"] = json.loads(row_dict["columns"])
                if row_dict.get("column_types"):
                    row_dict["column_types"] = json.loads(row_dict["column_types"])
                if row_dict.get("rows"):
                    row_dict["rows"] = json.loads(row_dict["rows"])

                data.append(row_dict)

            return data
        finally:
            conn.close()

    def get_available_histogram_names(self) -> List[str]:
        """Get list of available histogram log names

        Returns:
            List of unique histogram log names
        """
        conn = self._get_histograms_connection()
        try:
            result = conn.execute(
                "SELECT DISTINCT name FROM histograms ORDER BY name"
            ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.warning(f"Failed to query histograms table: {e}")
            return []
        finally:
            conn.close()

    def get_histogram_data(
        self, name: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get histogram data for a specific log name

        Args:
            name: Histogram log name
            limit: Optional row limit

        Returns:
            List of dicts with step, global_step, bins, values
        """
        conn = self._get_histograms_connection()
        try:
            query = f"SELECT * FROM histograms WHERE name = ?"
            if limit:
                query += f" LIMIT {limit}"

            result = conn.execute(query, [name]).fetchall()

            # Get column names
            columns = [desc[0] for desc in conn.description]

            # Convert to list of dicts, parsing JSON fields
            data = []
            for row in result:
                row_dict = dict(zip(columns, row))

                # Parse JSON bins and counts fields
                if row_dict.get("bins"):
                    row_dict["bins"] = json.loads(row_dict["bins"])
                if row_dict.get("counts"):
                    row_dict["counts"] = json.loads(row_dict["counts"])

                data.append(row_dict)

            return data
        finally:
            conn.close()

    def get_media_file_path(self, filename: str) -> Optional[Path]:
        """Get full path to media file

        Args:
            filename: Media filename

        Returns:
            Full path to media file, or None if not found
        """
        media_path = self.media_dir / filename
        if media_path.exists():
            return media_path
        return None

    def get_media_by_id(self, media_id: int) -> Optional[Dict[str, Any]]:
        """Get media metadata by database ID

        NEW in v0.2.0: Resolves <media id=123> tags to media metadata.
        Filename is derived as {media_hash}.{format}.

        Args:
            media_id: Media database ID (SQLite auto-increment ID)

        Returns:
            Media metadata dict with derived 'filename' field, or None if not found
        """
        conn = self._get_media_connection()
        try:
            query = "SELECT * FROM media WHERE id = ? LIMIT 1"
            cursor = conn.execute(query, (media_id,))
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()

            if row:
                media_dict = dict(zip(columns, row))
                # Derive filename from hash + format
                media_dict["filename"] = (
                    f"{media_dict['media_hash']}.{media_dict['format']}"
                )
                return media_dict
            return None

        finally:
            conn.close()

    def get_summary(self) -> Dict[str, Any]:
        """Get board summary with available data

        Returns:
            Dict with metadata, metrics, media, tables counts
        """
        metadata = self.get_metadata()

        # Count rows from each database (use separate connections)
        metrics_count = 0
        media_count = 0
        tables_count = 0
        histograms_count = 0

        try:
            conn = self._get_metrics_connection()
            metrics_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to count metrics: {e}")

        try:
            conn = self._get_media_connection()
            media_count = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to count media: {e}")

        try:
            conn = self._get_tables_connection()
            tables_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to count tables: {e}")

        try:
            conn = self._get_histograms_connection()
            histograms_count = conn.execute(
                "SELECT COUNT(*) FROM histograms"
            ).fetchone()[0]
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to count histograms: {e}")

        return {
            "metadata": metadata,
            "metrics_count": metrics_count,
            "media_count": media_count,
            "tables_count": tables_count,
            "histograms_count": histograms_count,
            "available_metrics": self.get_available_metrics(),
            "available_media": self.get_available_media_names(),
            "available_tables": self.get_available_table_names(),
            "available_histograms": self.get_available_histogram_names(),
        }


def list_boards(base_dir: Path) -> List[Dict[str, Any]]:
    """List all boards in base directory

    Args:
        base_dir: Base directory containing boards

    Returns:
        List of dicts with board_id and metadata
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.warning(f"Board data directory does not exist: {base_dir}")
        return []

    boards = []
    for board_dir in base_dir.iterdir():
        if not board_dir.is_dir():
            continue

        # Check if it has metadata.json
        metadata_path = board_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            boards.append(
                {
                    "board_id": board_dir.name,
                    "name": metadata.get("name", board_dir.name),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at"),
                    "config": metadata.get("config", {}),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to read metadata for {board_dir.name}: {e}")

    return sorted(boards, key=lambda x: x.get("created_at", ""), reverse=True)
