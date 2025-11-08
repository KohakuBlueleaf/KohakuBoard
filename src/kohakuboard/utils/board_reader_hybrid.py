"""Board reader for hybrid storage backend (Three-tier SQLite)

Reads from three specialized SQLite implementations:
1. KohakuVault ColumnVault - Metrics/histograms (blob-based columnar)
2. KohakuVault KVault - Media blobs (K-V table with B+Tree index)
3. Standard SQLite - Metadata (traditional relational tables)

All powered by KohakuVault for efficient data access.
"""

import io
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
        self.tensors_dir = self.board_dir / "data" / "tensors"

        # Validate
        if not self.board_dir.exists():
            raise FileNotFoundError(f"Board directory not found: {board_dir}")

        # Initialize KVault storage (read mode)
        self.media_kv = None
        if self.media_kv_path.exists():
            self.media_kv = KVault(str(self.media_kv_path))

        self._tensor_vaults: dict[str, KVault] = {}

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

    def _get_tensor_vault(self, kv_file: str) -> KVault:
        """Get or open tensor KVault by sanitized filename (without extension)."""
        if kv_file in self._tensor_vaults:
            return self._tensor_vaults[kv_file]

        db_path = self.tensors_dir / f"{kv_file}.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Tensor vault not found: {db_path}")

        kv = KVault(str(db_path))
        self._tensor_vaults[kv_file] = kv
        return kv

    def _load_tensor_blob(self, kv_file: str, kv_key: str) -> bytes:
        """Load raw tensor bytes from KVault."""
        kv = self._get_tensor_vault(kv_file)
        return kv.get(kv_key)

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

    def get_available_tensor_names(self) -> list[str]:
        """Get list of tensor log names from SQLite metadata."""
        if not self.sqlite_db.exists():
            return []

        try:
            conn = self._get_sqlite_connection()
            cursor = conn.execute("SELECT DISTINCT name FROM tensors ORDER BY name ASC")
            names = [row[0] for row in cursor.fetchall()]
            conn.close()
            return names
        except Exception as exc:
            logger.error(f"Failed to list tensor names: {exc}")
            return []

    def get_tensor_data(
        self, name: str, include_payload: bool = True
    ) -> list[dict[str, Any]]:
        """Retrieve tensor metadata (and optionally payload) for a given log name."""
        if not self.sqlite_db.exists():
            return []

        try:
            conn = self._get_sqlite_connection()
            cursor = conn.execute(
                """
                SELECT step, global_step, namespace, kv_file, kv_key, dtype,
                       shape, size_bytes, metadata
                FROM tensors
                WHERE name = ?
                ORDER BY step ASC
                """,
                (name,),
            )
            rows = cursor.fetchall()
            conn.close()
        except Exception as exc:
            logger.error(f"Failed to read tensors for {name}: {exc}")
            return []

        results: list[dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row[8]) if row[8] else {}
            shape = json.loads(row[6]) if row[6] else []
            entry = {
                "step": int(row[0]),
                "global_step": int(row[1]) if row[1] is not None else None,
                "namespace": row[2],
                "kv_file": row[3],
                "kv_key": row[4],
                "dtype": row[5],
                "shape": shape,
                "size_bytes": int(row[7]) if row[7] is not None else None,
                "metadata": metadata,
            }

            if include_payload:
                payload = self._load_tensor_blob(row[3], row[4])
                entry["payload"] = payload

            results.append(entry)

        return results

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
        """Get histogram names, including KDE logs exposed via histogram API."""
        histograms_dir = self.board_dir / "data" / "histograms"
        names: set[str] = set()

        if histograms_dir.exists():
            try:
                for db_file in histograms_dir.glob("*.db"):
                    cv = ColumnVault(str(db_file))
                    name_bytes_list = list(cv["name"])
                    for name_bytes in name_bytes_list:
                        names.add(name_bytes.decode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to list histogram names: {e}")

        # Always merge with KDE log names so summary exposes them in both lists.
        names.update(self.get_available_kernel_density_names())
        return sorted(names)

    def get_available_kernel_density_names(self) -> list[str]:
        """Get kernel density log names from metadata."""
        if not self.sqlite_db.exists():
            return []

        try:
            conn = self._get_sqlite_connection()
            cursor = conn.execute(
                "SELECT DISTINCT name FROM kernel_density ORDER BY name ASC"
            )
            names = [row[0] for row in cursor.fetchall()]
            conn.close()
        except Exception as exc:
            logger.error(f"Failed to list KDE names: {exc}")
            names = []

        return names

    def get_histogram_data(
        self,
        name: str,
        limit: int | None = None,
        bins: int | None = None,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get histogram-like data for both classic histograms and KDE logs."""
        hist_result = self._get_columnvault_histograms(name, limit=limit)
        if hist_result:
            return hist_result
        return self._get_kernel_density_histograms(
            name,
            limit=limit,
            target_bins=bins,
            override_range=(range_min, range_max),
        )

    def _get_columnvault_histograms(
        self, name: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        histograms_dir = self.board_dir / "data" / "histograms"
        if not histograms_dir.exists():
            return []

        namespace = name.split("/")[0] if "/" in name else name.replace("/", "__")
        name_bytes = name.encode("utf-8")

        def iter_candidate_files():
            for db_file in sorted(histograms_dir.glob("*.db")):
                stem = db_file.stem
                parts = stem.split("_")
                precision = None
                file_namespace = None

                if len(parts) >= 3 and parts[-1] in ("i32", "u8"):
                    precision = parts[-1]
                    file_namespace = "_".join(parts[:-2])
                elif len(parts) >= 2 and parts[-1] in ("i32", "u8"):
                    precision = parts[-1]
                    file_namespace = "_".join(parts[:-1])

                if (
                    precision is None
                    or file_namespace is None
                    or file_namespace != namespace
                ):
                    continue

                yield db_file, precision

        try:
            for db_file, precision in iter_candidate_files():
                if not db_file.exists():
                    continue

                cv = ColumnVault(str(db_file))
                steps = list(cv["step"])
                global_steps = list(cv["global_step"])
                names = list(cv["name"])
                counts_bytes_list = list(cv["counts"])
                mins = list(cv["min"])
                maxs = list(cv["max"])

                result: list[dict[str, Any]] = []

                for idx in range(len(steps)):
                    if names[idx] != name_bytes:
                        continue

                    counts_bytes = counts_bytes_list[idx]
                    if precision == "u8":
                        counts = list(
                            struct.unpack(f"{len(counts_bytes)}B", counts_bytes)
                        )
                    else:
                        num_bins = len(counts_bytes) // 4
                        counts = list(struct.unpack(f"<{num_bins}i", counts_bytes))

                    min_val = float(mins[idx])
                    max_val = float(maxs[idx])
                    num_bins = len(counts)
                    bin_edges = np.linspace(min_val, max_val, num_bins + 1).tolist()

                    result.append(
                        {
                            "step": int(steps[idx]),
                            "global_step": (
                                int(global_steps[idx]) if global_steps[idx] else None
                            ),
                            "bins": bin_edges,
                            "counts": counts,
                        }
                    )

                    if limit and len(result) >= limit:
                        break

                if result:
                    return result

            return []
        except Exception as exc:
            logger.error(f"Failed to read histogram '{name}': {exc}")
            return []

    def _get_kernel_density_histograms(
        self,
        name: str,
        limit: int | None = None,
        target_bins: int | None = None,
        override_range: tuple[float | None, float | None] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.sqlite_db.exists():
            return []

        try:
            conn = self._get_sqlite_connection()
            cursor = conn.execute(
                """
                SELECT step, global_step, kv_file, kv_key, kernel, bandwidth,
                       sample_count, range_min, range_max, num_points, metadata
                FROM kernel_density
                WHERE name = ?
                ORDER BY step ASC
                """,
                (name,),
            )
            rows = cursor.fetchall()
            conn.close()
        except Exception as exc:
            logger.error(f"Failed to read KDE metadata for '{name}': {exc}")
            return []

        if not rows:
            return []

        range_candidates_min: list[float] = []
        range_candidates_max: list[float] = []
        num_points_list: list[int] = []
        parsed_rows: list[dict[str, Any]] = []

        for row in rows:
            meta = json.loads(row[10]) if row[10] else {}
            entry = {
                "step": int(row[0]),
                "global_step": int(row[1]) if row[1] is not None else None,
                "kv_file": row[2],
                "kv_key": row[3],
                "kernel": row[4],
                "bandwidth": row[5],
                "sample_count": int(row[6]) if row[6] is not None else None,
                "range_min": float(row[7]) if row[7] is not None else None,
                "range_max": float(row[8]) if row[8] is not None else None,
                "num_points": int(row[9]) if row[9] is not None else None,
                "metadata": meta,
            }

            try:
                blob = self._load_tensor_blob(entry["kv_file"], entry["kv_key"])
            except Exception as exc:
                logger.error(
                    f"Failed to load KDE payload {entry['kv_file']}[{entry['kv_key']}]: {exc}"
                )
                continue

            try:
                with np.load(io.BytesIO(blob)) as data:
                    grid = np.asarray(data["grid"], dtype=np.float32)
                    density = np.asarray(data["density"], dtype=np.float32)
            except Exception as exc:
                logger.error(f"Invalid KDE payload for '{name}': {exc}")
                continue

            if grid.ndim != 1 or density.ndim != 1 or grid.size != density.size:
                logger.warning(
                    f"Skipping malformed KDE entry for '{name}' (shape mismatch)"
                )
                continue

            entry["grid"] = grid
            entry["density"] = density

            range_candidates_min.append(
                entry["range_min"]
                if entry["range_min"] is not None
                else float(grid.min())
            )
            range_candidates_max.append(
                entry["range_max"]
                if entry["range_max"] is not None
                else float(grid.max())
            )

            if entry["num_points"] is None:
                entry["num_points"] = int(grid.size)
            num_points_list.append(entry["num_points"])

            parsed_rows.append(entry)

        if not parsed_rows:
            return []

        canonical_min = min(range_candidates_min)
        canonical_max = max(range_candidates_max)
        if canonical_min == canonical_max:
            canonical_min -= 0.005
            canonical_max += 0.005

        if override_range:
            o_min, o_max = override_range
            if o_min is not None:
                canonical_min = float(o_min)
            if o_max is not None:
                canonical_max = float(o_max)
            if canonical_min >= canonical_max:
                canonical_max = canonical_min + 1.0

        if num_points_list:
            base_points = max(min(min(num_points_list), 256), 32)
        else:
            base_points = 128

        if target_bins is not None and target_bins >= 8:
            canonical_bins = int(target_bins)
        else:
            canonical_bins = max(base_points - 1, 32)
        bins = np.linspace(canonical_min, canonical_max, canonical_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        widths = np.diff(bins)

        results: list[dict[str, Any]] = []

        for entry in parsed_rows:
            grid = entry["grid"]
            density = entry["density"]

            range_min_entry = (
                float(entry["range_min"])
                if entry["range_min"] is not None
                else float(grid.min())
            )
            range_max_entry = (
                float(entry["range_max"])
                if entry["range_max"] is not None
                else float(grid.max())
            )
            if range_min_entry > range_max_entry:
                range_min_entry, range_max_entry = range_max_entry, range_min_entry

            interpolated = np.interp(
                centers,
                grid,
                density,
                left=0.0,
                right=0.0,
            )
            # Clamp any bins that fall outside the recorded KDE range to zero to
            # avoid artificial plateaus when canonical ranges expand.
            tolerance = max((range_max_entry - range_min_entry) * 1e-6, 1e-9)
            outside_mask = (centers + tolerance) < range_min_entry
            outside_mask |= (centers - tolerance) > range_max_entry
            interpolated[outside_mask] = 0.0

            multiplier = entry["sample_count"] if entry["sample_count"] else 1.0
            counts = interpolated * widths * multiplier
            counts = np.clip(counts, a_min=0.0, a_max=None)
            counts = counts.astype(float).tolist()

            results.append(
                {
                    "step": entry["step"],
                    "global_step": entry["global_step"],
                    "bins": bins.tolist(),
                    "counts": counts,
                    "precision": "kde",
                    "kernel": entry["kernel"],
                    "bandwidth": entry["bandwidth"],
                    "sample_count": entry["sample_count"],
                    "range_min": entry["range_min"],
                    "range_max": entry["range_max"],
                }
            )

            if limit and len(results) >= limit:
                break

        return results

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
        tensor_count = 0
        kde_count = 0

        if self.sqlite_db.exists():
            try:
                conn = self._get_sqlite_connection()
                cursor = conn.execute("SELECT COUNT(*) FROM media")
                media_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM tables")
                tables_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM tensors")
                tensor_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM kernel_density")
                kde_count = cursor.fetchone()[0]

                conn.close()
            except Exception as e:
                logger.warning(f"Failed to count metadata: {e}")

        return {
            "metadata": metadata,
            "metrics_count": metrics_count,
            "media_count": media_count,
            "tables_count": tables_count,
            "tensors_count": tensor_count,
            "kernel_density_count": kde_count,
            "histograms_count": len(self.get_available_histogram_names()),
            "available_metrics": self.get_available_metrics(),
            "available_media": self.get_available_media_names(),
            "available_tables": self.get_available_table_names(),
            "available_histograms": self.get_available_histogram_names(),
            "available_tensors": self.get_available_tensor_names(),
            "available_kernel_density": self.get_available_kernel_density_names(),
        }
