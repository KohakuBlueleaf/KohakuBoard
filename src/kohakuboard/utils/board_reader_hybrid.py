"""Board reader for hybrid storage backend (Three-tier SQLite)

Reads from three specialized SQLite implementations:
1. KohakuVault ColumnVault - Metrics/histograms (blob-based columnar)
2. KohakuVault KVault - Media blobs (K-V table with B+Tree index)
3. Standard SQLite - Metadata (traditional relational tables)

All powered by KohakuVault for efficient data access.
"""

import json
import math
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from kohakuvault import ColumnVault, KVault

from kohakuboard.logger import get_logger

# Get logger for board reader
logger = get_logger("READER")


class HybridBoardReader:
    """Reader for hybrid storage (Three-tier SQLite architecture)

    Reads from three specialized SQLite implementations:
    1. KohakuVault ColumnVault - Metrics (blob-based columnar, Rust-managed)
    2. KohakuVault KVault - Media blobs (K-V table with B+Tree index)
    3. Standard SQLite - Metadata (traditional relational tables)
    """

    def __init__(self, board_dir: Path):
        """Initialize hybrid board reader

        Args:
            board_dir: Path to board directory
        """
        self.board_dir = Path(board_dir)
        self.metadata_path = self.board_dir / "metadata.json"
        self.media_dir = self.board_dir / "media"

        # Storage paths
        self.metrics_dir = (
            self.board_dir / "data" / "metrics"
        )  # Per-metric .db files (ColumnVault)
        self.sqlite_db = self.board_dir / "data" / "metadata.db"
        self.media_kv_path = self.board_dir / "media" / "blobs.db"

        # Validate
        if not self.board_dir.exists():
            raise FileNotFoundError(f"Board directory not found: {board_dir}")

        # Initialize KVault storage (read mode)
        self.media_kv = None
        if self.media_kv_path.exists():
            self.media_kv = KVault(str(self.media_kv_path))

        # Retry configuration (for SQLite locks)
        self.max_retries = 5
        self.retry_delay = 0.05

    def get_metadata(self) -> dict[str, Any]:
        """Get board metadata

        Returns:
            Metadata dict
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def get_latest_step(self) -> dict[str, Any] | None:
        """Get latest step info from steps table

        Returns:
            Dict with step, global_step, timestamp or None
        """
        if not self.sqlite_db.exists():
            return None

        conn = self._get_sqlite_connection()
        try:
            cursor = conn.execute(
                "SELECT step, global_step, timestamp FROM steps ORDER BY step DESC LIMIT 1"
            )
            row = cursor.fetchone()

            if row:
                return {
                    "step": row[0],
                    "global_step": row[1],
                    "timestamp": row[2],
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get latest step: {e}")
            return None
        finally:
            conn.close()

    def _get_metric_db_file(self, metric: str) -> Path:
        """Get ColumnVault DB file path for a metric

        Args:
            metric: Metric name

        Returns:
            Path to metric's .db file
        """
        escaped_name = metric.replace("/", "__")
        return self.metrics_dir / f"{escaped_name}.db"

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        """Get SQLite connection (with retry)

        Returns:
            SQLite connection
        """
        if not self.sqlite_db.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.sqlite_db}")

        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                return sqlite3.connect(str(self.sqlite_db))
            except sqlite3.OperationalError as e:
                # SQLite lock error
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.debug(
                        f"SQLite connection retry {attempt}/{self.max_retries} after {delay:.3f}s"
                    )
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                logger.error(f"Non-lock error opening SQLite: {type(e).__name__}: {e}")
                raise

        raise last_error

    def get_available_metrics(self) -> list[str]:
        """Get list of available scalar metrics from ColumnVault DB files

        Returns:
            List of metric names
        """
        if not self.metrics_dir.exists():
            return ["step", "global_step", "timestamp"]  # Base columns always available

        try:
            # List all .db files in metrics directory
            db_files = list(self.metrics_dir.glob("*.db"))

            # Convert filenames back to metric names
            metrics = []
            for db_file in db_files:
                # Remove .db extension and convert __ back to /
                metric_name = db_file.stem.replace("__", "/")
                metrics.append(metric_name)

            # Add base columns at the beginning
            return ["step", "global_step", "timestamp"] + sorted(metrics)

        except Exception as e:
            logger.error(f"Failed to list metrics: {e}")
            return ["step", "global_step", "timestamp"]

    def get_scalar_data(self, metric: str, limit: int | None = None) -> dict[str, list]:
        """Get scalar data for a metric

        Args:
            metric: Metric name (can be base column: step/global_step/timestamp)
            limit: Optional row limit

        Returns:
            Columnar format: {steps: [], global_steps: [], timestamps: [], values: []}
        """
        # Handle base columns from SQLite steps table
        if metric in ("step", "global_step", "timestamp"):
            return self._get_base_column_data(metric, limit)

        # Handle regular metrics from ColumnVault DB files
        metric_file = self._get_metric_db_file(metric)

        if not metric_file.exists():
            return {"steps": [], "global_steps": [], "timestamps": [], "values": []}

        try:
            # Open metric's ColumnVault database
            cv = ColumnVault(str(metric_file))

            # Read all columns (efficient columnar access!)
            steps = list(cv["step"])
            global_steps = list(cv["global_step"])
            timestamps_ms = list(cv["timestamp"])
            raw_values = list(cv["value"])

            # Apply limit if specified
            if limit:
                steps = steps[-limit:]
                global_steps = global_steps[-limit:]
                timestamps_ms = timestamps_ms[-limit:]
                raw_values = raw_values[-limit:]

            # Convert timestamp ms to seconds
            timestamps = [int(ts / 1000) if ts else None for ts in timestamps_ms]

            # Process values (handle NaN/inf)
            values = []
            for value in raw_values:
                if value is None:
                    value = None  # Treat NULL as sparse
                elif isinstance(value, float):
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
        except Exception as e:
            logger.error(f"Failed to read metric '{metric}' from ColumnVault: {e}")
            return {"steps": [], "global_steps": [], "timestamps": [], "values": []}

    def _get_base_column_data(
        self, column: str, limit: int | None = None
    ) -> dict[str, list]:
        """Get base column data from SQLite steps table

        Args:
            column: Column name (step, global_step, or timestamp)
            limit: Optional row limit

        Returns:
            Columnar format with the requested column as values
        """
        if not self.sqlite_db.exists():
            return {"steps": [], "global_steps": [], "timestamps": [], "values": []}

        conn = self._get_sqlite_connection()
        try:
            query = "SELECT step, global_step, timestamp FROM steps ORDER BY step"
            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)
            rows = cursor.fetchall()

            steps = []
            global_steps = []
            timestamps = []
            values = []

            for row in rows:
                steps.append(row[0])
                global_steps.append(row[1])
                timestamps.append(
                    int(row[2] / 1000) if row[2] else None
                )  # ms to seconds

                # The requested column becomes the "value"
                if column == "step":
                    values.append(row[0])
                elif column == "global_step":
                    values.append(row[1])
                else:  # timestamp
                    values.append(int(row[2] / 1000) if row[2] else None)

            return {
                "steps": steps,
                "global_steps": global_steps,
                "timestamps": timestamps,
                "values": values,
            }
        finally:
            conn.close()

    def get_available_media_names(self) -> list[str]:
        """Get list of available media names

        Returns:
            List of media names
        """
        if not self.sqlite_db.exists():
            return []

        conn = self._get_sqlite_connection()
        try:
            cursor = conn.execute("SELECT DISTINCT name FROM media ORDER BY name")
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to query media: {e}")
            return []
        finally:
            conn.close()

    def get_media_entries(
        self, name: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get media entries for a media log name

        Args:
            name: Media log name
            limit: Optional limit

        Returns:
            List of media entries (filename derived as {media_hash}.{format})
        """
        if not self.sqlite_db.exists():
            return []

        conn = self._get_sqlite_connection()
        try:
            query = "SELECT * FROM media WHERE name = ?"
            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query, (name,))
            columns = [desc[0] for desc in cursor.description]

            # Convert to list of dicts and derive filename
            media_list = []
            for row in cursor.fetchall():
                media_dict = dict(zip(columns, row))
                # Derive filename from hash + format (v0.2.0+)
                media_dict["filename"] = (
                    f"{media_dict['media_hash']}.{media_dict['format']}"
                )
                media_list.append(media_dict)

            return media_list
        finally:
            conn.close()

    def get_available_table_names(self) -> list[str]:
        """Get list of available table names

        Returns:
            List of table names
        """
        if not self.sqlite_db.exists():
            return []

        conn = self._get_sqlite_connection()
        try:
            cursor = conn.execute("SELECT DISTINCT name FROM tables ORDER BY name")
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to query tables: {e}")
            return []
        finally:
            conn.close()

    def get_table_data(
        self, name: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get table data for a name

        Args:
            name: Table name
            limit: Optional limit

        Returns:
            List of table entries
        """
        if not self.sqlite_db.exists():
            return []

        conn = self._get_sqlite_connection()
        try:
            query = "SELECT * FROM tables WHERE name = ?"
            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query, (name,))
            columns = [desc[0] for desc in cursor.description]

            data = []
            for row in cursor.fetchall():
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

    def get_available_histogram_names(self) -> list[str]:
        """Get histogram names from ColumnVault DB files

        Returns:
            List of unique histogram names
        """
        histograms_dir = self.board_dir / "data" / "histograms"
        if not histograms_dir.exists():
            return []

        try:
            names = set()

            # Read from all histogram ColumnVault DB files
            for db_file in histograms_dir.glob("*.db"):
                cv = ColumnVault(str(db_file))
                # Read all name entries
                name_bytes_list = list(cv["name"])
                # Decode bytes to strings
                for name_bytes in name_bytes_list:
                    names.add(name_bytes.decode("utf-8"))

            return sorted(names)

        except Exception as e:
            logger.error(f"Failed to list histogram names: {e}")
            return []

    def get_histogram_data(
        self, name: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get histogram data

        Args:
            name: Histogram name
            limit: Optional limit

        Returns:
            List of histogram entries
        """
        histograms_dir = self.board_dir / "data" / "histograms"
        if not histograms_dir.exists():
            return []

        try:
            # Determine which file
            namespace = name.split("/")[0] if "/" in name else name.replace("/", "__")

            # Try both precisions
            for suffix in ["_i32", "_u8"]:
                db_file = histograms_dir / f"{namespace}{suffix}.db"
                if not db_file.exists():
                    continue

                cv = ColumnVault(str(db_file))

                # Read all data from columns
                steps = list(cv["step"])
                global_steps = list(cv["global_step"])
                names = list(cv["name"])
                counts_bytes_list = list(cv["counts"])
                mins = list(cv["min"])
                maxs = list(cv["max"])

                # Filter by name and build result
                result = []
                name_bytes = name.encode("utf-8")

                for i in range(len(steps)):
                    if names[i] != name_bytes:
                        continue

                    # Deserialize counts from fixed-size bytes
                    counts_bytes = counts_bytes_list[i]
                    if suffix == "_u8":
                        # uint8: 1 byte per bin
                        counts = list(
                            struct.unpack(f"{len(counts_bytes)}B", counts_bytes)
                        )
                    else:
                        # int32: 4 bytes per bin, little-endian
                        num_bins = len(counts_bytes) // 4
                        counts = list(struct.unpack(f"<{num_bins}i", counts_bytes))

                    min_val = float(mins[i])
                    max_val = float(maxs[i])
                    num_bins = len(counts)

                    # Reconstruct bin edges from min/max/num_bins
                    bin_edges = np.linspace(min_val, max_val, num_bins + 1).tolist()

                    result.append(
                        {
                            "step": int(steps[i]),
                            "global_step": (
                                int(global_steps[i]) if global_steps[i] else None
                            ),
                            "bins": bin_edges,  # Bin EDGES (K+1 values)
                            "counts": counts,  # Counts (K values)
                        }
                    )

                    # Apply limit
                    if limit and len(result) >= limit:
                        break

                if result:
                    return result

            return []

        except Exception as e:
            logger.error(f"Failed to read histogram '{name}': {e}")
            return []

    def get_media_file_path(self, filename: str) -> Path | None:
        """Get full path to media file (DEPRECATED - use get_media_data instead)

        Args:
            filename: Media filename

        Returns:
            Full path or None
        """
        media_path = self.media_dir / filename
        return media_path if media_path.exists() else None

    def get_media_data(self, filename: str) -> bytes | None:
        """Get media binary data from KVault

        Args:
            filename: Media filename in format {media_hash}.{format}

        Returns:
            Binary media data or None if not found
        """
        if self.media_kv is None:
            logger.warning(f"KVault not initialized, cannot retrieve media: {filename}")
            return None

        try:
            data = self.media_kv.get(filename)
            return data
        except Exception as e:
            logger.error(f"Failed to read media from KVault: {filename}, error: {e}")
            return None

    def get_media_by_id(self, media_id: int) -> dict[str, Any] | None:
        """Get media metadata by ID from SQLite metadata DB

        NEW in v0.2.0: Resolves <media id=123> tags to media metadata.
        Filename is derived as {media_hash}.{format}.

        Args:
            media_id: Media database ID (SQLite auto-increment ID)

        Returns:
            Media metadata dict with derived 'filename' field, or None if not found
        """
        if not self.sqlite_db.exists():
            return None

        conn = self._get_sqlite_connection()

        try:
            query = "SELECT * FROM media WHERE id = ? LIMIT 1"
            cursor = conn.execute(query, (media_id,))
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()

            if row:
                # Convert row tuple to dict using column names
                media_dict = dict(zip(columns, row))
                # Derive filename from hash + format
                media_dict["filename"] = (
                    f"{media_dict['media_hash']}.{media_dict['format']}"
                )
                return media_dict
            return None

        finally:
            conn.close()

    def get_summary(self) -> dict[str, Any]:
        """Get board summary

        Returns:
            Summary with counts and available data
        """
        metadata = self.get_metadata()

        # Count from ColumnVault (sum rows from all metric files)
        metrics_count = 0
        if self.metrics_dir.exists():
            try:
                for db_file in self.metrics_dir.glob("*.db"):
                    cv = ColumnVault(str(db_file))
                    metrics_count += len(cv["step"])
            except Exception as e:
                logger.warning(f"Failed to count metrics: {e}")

        # Count from SQLite
        media_count = 0
        tables_count = 0

        if self.sqlite_db.exists():
            try:
                conn = self._get_sqlite_connection()
                cursor = conn.execute("SELECT COUNT(*) FROM media")
                media_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM tables")
                tables_count = cursor.fetchone()[0]

                conn.close()
            except Exception as e:
                logger.warning(f"Failed to count metadata: {e}")

        return {
            "metadata": metadata,
            "metrics_count": metrics_count,
            "media_count": media_count,
            "tables_count": tables_count,
            "histograms_count": len(self.get_available_histogram_names()),
            "available_metrics": self.get_available_metrics(),
            "available_media": self.get_available_media_names(),
            "available_tables": self.get_available_table_names(),
            "available_histograms": self.get_available_histogram_names(),
        }
