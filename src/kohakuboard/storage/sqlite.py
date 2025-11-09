"""SQLite-based storage for media and table metadata

Uses Python's built-in sqlite3 module for zero overhead and excellent multi-connection support.
Fixed schema - no dynamic columns needed.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from kohakuboard.logger import get_logger


class SQLiteMetadataStorage:
    """SQLite storage for media and table logs

    Benefits:
    - Built-in module (zero dependency overhead)
    - Excellent multi-connection support (WAL mode)
    - Auto-commit for simplicity
    - Fixed schema (media/tables don't need dynamic columns)
    """

    def __init__(self, base_dir: Path, logger=None):
        """Initialize SQLite metadata storage

        Args:
            base_dir: Base directory for database file
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

        self.db_file = base_dir / "metadata.db"

        # Use WAL mode for better concurrent access
        self.conn = sqlite3.connect(str(self.db_file), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes

        # Create tables
        self._init_tables()

    def _init_tables(self):
        """Initialize database tables"""
        # Steps table (global step/timestamp tracking)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                step INTEGER PRIMARY KEY,
                global_step INTEGER,
                timestamp INTEGER
            )
        """
        )

        # Media table (v0.2.0+ with content-addressable storage)
        # Filename is derived as {media_hash}.{format}, not stored
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_hash TEXT NOT NULL,
                format TEXT NOT NULL,
                step INTEGER NOT NULL,
                global_step INTEGER,
                name TEXT NOT NULL,
                caption TEXT,
                type TEXT NOT NULL,
                size_bytes INTEGER,
                width INTEGER,
                height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(media_hash, format)
            )
        """
        )

        # Indices for fast lookup
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_media_id ON media(id)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_media_hash ON media(media_hash)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_media_name ON media(name)")

        # Tables table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tables (
                step INTEGER NOT NULL,
                global_step INTEGER,
                name TEXT NOT NULL,
                columns TEXT,
                column_types TEXT,
                rows TEXT
            )
        """
        )

        self.conn.commit()

        # Buffers for batching (NOTE: media is no longer batched as of v0.2.0)
        self.step_buffer: list[tuple] = []
        self.table_buffer: list[tuple] = []

        # Ensure tensors table exists
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tensors (
                step INTEGER NOT NULL,
                global_step INTEGER,
                name TEXT NOT NULL,
                namespace TEXT NOT NULL,
                kv_file TEXT NOT NULL,
                kv_key TEXT NOT NULL,
                dtype TEXT,
                shape TEXT,
                size_bytes INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tensors_name ON tensors(name)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tensors_step ON tensors(step)"
        )

        # Kernel density metadata table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kernel_density (
                step INTEGER NOT NULL,
                global_step INTEGER,
                name TEXT NOT NULL,
                kv_file TEXT NOT NULL,
                kv_key TEXT NOT NULL,
                kernel TEXT,
                bandwidth TEXT,
                sample_count INTEGER,
                range_min REAL,
                range_max REAL,
                num_points INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kde_name ON kernel_density(name)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kde_step ON kernel_density(step)"
        )

        self.conn.commit()

        # Flush thresholds
        self.step_flush_threshold = 1000
        self.table_flush_threshold = 100

    def append_step_info(
        self,
        step: int,
        global_step: int | None,
        timestamp: int,
    ):
        """Record step/global_step/timestamp mapping (batched)

        Args:
            step: Step number
            global_step: Global step
            timestamp: Unix timestamp (milliseconds)
        """
        self.step_buffer.append((step, global_step, timestamp))

        # Don't auto-flush - writer will call flush() periodically

    def _flush_steps(self):
        """Flush steps buffer"""
        if not self.step_buffer:
            return

        # Bulk INSERT OR REPLACE
        self.conn.executemany(
            "INSERT OR REPLACE INTO steps (step, global_step, timestamp) VALUES (?, ?, ?)",
            self.step_buffer,
        )
        self.conn.commit()
        self.logger.debug(f"Flushed {len(self.step_buffer)} step records to SQLite")
        self.step_buffer.clear()

    def append_media(
        self,
        step: int,
        global_step: int | None,
        name: str,
        media_list: list[dict[str, Any]],
        caption: str | None = None,
    ) -> list[int]:
        """Append media log entry with deduplication (immediate insert, no batching)

        NEW in v0.2.0: Returns list of media IDs for reference in tables.
        Media inserts are no longer batched because we need immediate IDs.
        Filename is derived as {media_hash}.{format}, not stored in DB.

        Args:
            step: Auto-increment step
            global_step: Explicit global step
            name: Media log name
            media_list: List of media metadata dicts (from media_handler.process_media())
            caption: Optional caption

        Returns:
            List of media IDs (SQLite auto-increment IDs)
        """
        media_ids = []
        cursor = self.conn.cursor()

        for media_meta in media_list:
            media_hash = media_meta["media_hash"]
            format_ext = media_meta["format"]

            # Check if media already exists by (hash, format) - deduplication at DB level
            cursor.execute(
                "SELECT id FROM media WHERE media_hash = ? AND format = ?",
                (media_hash, format_ext),
            )
            existing = cursor.fetchone()

            if existing:
                # Media already in DB, reuse existing ID
                media_id = existing[0]
                self.logger.debug(
                    f"Reusing existing media ID {media_id} for {media_hash}.{format_ext}"
                )
            else:
                # Insert new media entry (filename derived from hash + format)
                cursor.execute(
                    """
                    INSERT INTO media (
                        media_hash, format, step, global_step, name, caption,
                        type, size_bytes, width, height
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        media_hash,
                        format_ext,
                        step,
                        global_step,
                        name,
                        caption or "",
                        media_meta["type"],
                        media_meta["size_bytes"],
                        media_meta.get("width"),
                        media_meta.get("height"),
                    ),
                )
                media_id = cursor.lastrowid
                self.logger.debug(
                    f"Inserted new media ID {media_id} for {media_hash}.{format_ext}"
                )

            media_ids.append(media_id)

        self.conn.commit()
        return media_ids

    def _flush_media(self):
        """Flush media buffer (DEPRECATED in v0.2.0 - media no longer batched)

        Media inserts are now immediate because we need to return IDs for table references.
        This method is kept for compatibility but does nothing.
        """
        pass

    def append_table(
        self,
        step: int,
        global_step: int | None,
        name: str,
        table_data: dict[str, Any],
    ):
        """Append table log entry (batched)

        Args:
            step: Auto-increment step
            global_step: Explicit global step
            name: Table log name
            table_data: Table dict with columns, column_types, rows
        """
        row = (
            step,
            global_step,
            name,
            json.dumps(table_data["columns"]),
            json.dumps(table_data["column_types"]),
            json.dumps(table_data["rows"]),
        )
        self.table_buffer.append(row)

        # Don't auto-flush - writer will call flush() periodically

    def _flush_tables(self):
        """Flush tables buffer"""
        if not self.table_buffer:
            return

        self.conn.executemany(
            "INSERT INTO tables (step, global_step, name, columns, column_types, rows) VALUES (?, ?, ?, ?, ?, ?)",
            self.table_buffer,
        )
        self.conn.commit()
        self.logger.debug(f"Flushed {len(self.table_buffer)} table rows to SQLite")
        self.table_buffer.clear()

    def append_tensor_metadata(
        self,
        step: int,
        global_step: int | None,
        name: str,
        namespace: str,
        kv_file: str,
        kv_key: str,
        tensor_meta: dict[str, Any],
        size_bytes: int,
    ):
        """Insert tensor metadata row."""
        metadata_json = json.dumps(tensor_meta.get("metadata", {}))
        shape_json = json.dumps(tensor_meta.get("shape", []))
        dtype = tensor_meta.get("dtype")

        self.conn.execute(
            """
            INSERT INTO tensors (
                step, global_step, name, namespace, kv_file, kv_key, dtype,
                shape, size_bytes, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                step,
                global_step,
                name,
                namespace,
                kv_file,
                kv_key,
                dtype,
                shape_json,
                size_bytes,
                metadata_json,
            ),
        )
        self.conn.commit()

    def append_kernel_density_metadata(
        self,
        step: int,
        global_step: int | None,
        name: str,
        kv_file: str,
        kv_key: str,
        kde_meta: dict[str, Any],
        num_points: int,
    ):
        """Insert kernel density metadata row."""
        metadata_json = json.dumps(kde_meta.get("metadata", {}))
        bandwidth = kde_meta.get("bandwidth")

        if bandwidth is not None and not isinstance(bandwidth, str):
            bandwidth_str = f"{bandwidth}"
        else:
            bandwidth_str = bandwidth

        self.conn.execute(
            """
            INSERT INTO kernel_density (
                step, global_step, name, kv_file, kv_key, kernel, bandwidth,
                sample_count, range_min, range_max, num_points, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                step,
                global_step,
                name,
                kv_file,
                kv_key,
                kde_meta.get("kernel"),
                bandwidth_str,
                kde_meta.get("sample_count"),
                kde_meta.get("range_min"),
                kde_meta.get("range_max"),
                num_points,
                metadata_json,
            ),
        )
        self.conn.commit()

    def fetch_steps_range(self, start_step: int, end_step: int) -> list[dict[str, Any]]:
        """Fetch steps within range inclusive."""
        cursor = self.conn.execute(
            """
            SELECT step, global_step, timestamp
            FROM steps
            WHERE step >= ? AND step <= ?
            ORDER BY step ASC
            """,
            (start_step, end_step),
        )
        return [
            {
                "step": int(row[0]),
                "global_step": int(row[1]) if row[1] is not None else None,
                "timestamp": int(row[2]) if row[2] is not None else None,
            }
            for row in cursor.fetchall()
        ]

    def fetch_media_range(self, start_step: int, end_step: int) -> list[dict[str, Any]]:
        """Fetch media metadata within range inclusive."""
        cursor = self.conn.execute(
            """
            SELECT id, media_hash, format, step, global_step, name,
                   caption, type, width, height, size_bytes
            FROM media
            WHERE step >= ? AND step <= ?
            ORDER BY step ASC
            """,
            (start_step, end_step),
        )
        return [
            {
                "id": int(row[0]),
                "media_hash": row[1],
                "format": row[2],
                "step": int(row[3]),
                "global_step": int(row[4]) if row[4] is not None else None,
                "name": row[5],
                "caption": row[6],
                "type": row[7],
                "width": int(row[8]) if row[8] is not None else None,
                "height": int(row[9]) if row[9] is not None else None,
                "size_bytes": int(row[10]) if row[10] is not None else None,
            }
            for row in cursor.fetchall()
        ]

    def fetch_tables_range(
        self, start_step: int, end_step: int
    ) -> list[dict[str, Any]]:
        """Fetch tables within step range inclusive."""
        cursor = self.conn.execute(
            """
            SELECT step, global_step, name, columns, column_types, rows
            FROM tables
            WHERE step >= ? AND step <= ?
            ORDER BY step ASC
            """,
            (start_step, end_step),
        )
        tables: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            tables.append(
                {
                    "step": int(row[0]),
                    "global_step": int(row[1]) if row[1] is not None else None,
                    "name": row[2],
                    "columns": json.loads(row[3]) if row[3] else [],
                    "column_types": json.loads(row[4]) if row[4] else [],
                    "rows": json.loads(row[5]) if row[5] else [],
                }
            )
        return tables

    def fetch_kernel_density_range(
        self, start_step: int, end_step: int
    ) -> list[dict[str, Any]]:
        """Fetch kernel density metadata within step range inclusive."""
        cursor = self.conn.execute(
            """
            SELECT step, global_step, name, kv_file, kv_key, kernel, bandwidth,
                   sample_count, range_min, range_max, num_points, metadata
            FROM kernel_density
            WHERE step >= ? AND step <= ?
            ORDER BY step ASC
            """,
            (start_step, end_step),
        )
        entries: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            meta = json.loads(row[11]) if row[11] else {}
            entries.append(
                {
                    "step": int(row[0]),
                    "global_step": int(row[1]) if row[1] is not None else None,
                    "name": row[2],
                    "kv_file": row[3],
                    "kv_key": row[4],
                    "kernel": row[5],
                    "bandwidth": row[6],
                    "sample_count": int(row[7]) if row[7] is not None else None,
                    "range_min": float(row[8]) if row[8] is not None else None,
                    "range_max": float(row[9]) if row[9] is not None else None,
                    "num_points": int(row[10]) if row[10] is not None else None,
                    "metadata": meta,
                }
            )
        return entries

    def fetch_tensors_since(self, last_rowid: int) -> list[dict[str, Any]]:
        """Fetch tensor metadata rows with rowid greater than last_rowid."""
        cursor = self.conn.execute(
            """
            SELECT rowid, step, global_step, name, namespace, kv_file, kv_key,
                   dtype, shape, size_bytes, metadata
            FROM tensors
            WHERE rowid > ?
            ORDER BY rowid ASC
            """,
            (last_rowid,),
        )
        results: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            results.append(
                {
                    "rowid": int(row[0]),
                    "step": int(row[1]),
                    "global_step": int(row[2]) if row[2] is not None else None,
                    "name": row[3],
                    "namespace": row[4],
                    "kv_file": row[5],
                    "kv_key": row[6],
                    "dtype": row[7],
                    "shape": json.loads(row[8]) if row[8] else [],
                    "size_bytes": int(row[9]) if row[9] is not None else None,
                    "metadata": json.loads(row[10]) if row[10] else {},
                }
            )
        return results

    def fetch_kernel_density_since(self, last_rowid: int) -> list[dict[str, Any]]:
        """Fetch kernel density rows with rowid greater than last_rowid."""
        cursor = self.conn.execute(
            """
            SELECT rowid, step, global_step, name, kv_file, kv_key, kernel, bandwidth,
                   sample_count, range_min, range_max, num_points, metadata
            FROM kernel_density
            WHERE rowid > ?
            ORDER BY rowid ASC
            """,
            (last_rowid,),
        )
        entries: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            meta = json.loads(row[12]) if row[12] else {}
            entries.append(
                {
                    "rowid": int(row[0]),
                    "step": int(row[1]),
                    "global_step": int(row[2]) if row[2] is not None else None,
                    "name": row[3],
                    "kv_file": row[4],
                    "kv_key": row[5],
                    "kernel": row[6],
                    "bandwidth": row[7],
                    "sample_count": int(row[8]) if row[8] is not None else None,
                    "range_min": float(row[9]) if row[9] is not None else None,
                    "range_max": float(row[10]) if row[10] is not None else None,
                    "num_points": int(row[11]) if row[11] is not None else None,
                    "metadata": meta,
                }
            )
        return entries

    def list_tensor_names(self) -> list[str]:
        """Return sorted tensor log names."""
        cursor = self.conn.execute(
            "SELECT DISTINCT name FROM tensors ORDER BY name ASC"
        )
        return [row[0] for row in cursor.fetchall()]

    def fetch_tensors_by_name(self, name: str) -> list[dict[str, Any]]:
        """Fetch tensor metadata rows for a given log name ordered by step."""
        cursor = self.conn.execute(
            """
            SELECT step, global_step, namespace, kv_file, kv_key,
                   dtype, shape, size_bytes, metadata
            FROM tensors
            WHERE name = ?
            ORDER BY step ASC
            """,
            (name,),
        )
        results: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            results.append(
                {
                    "step": int(row[0]),
                    "global_step": int(row[1]) if row[1] is not None else None,
                    "namespace": row[2],
                    "kv_file": row[3],
                    "kv_key": row[4],
                    "dtype": row[5],
                    "shape": json.loads(row[6]) if row[6] else [],
                    "size_bytes": int(row[7]) if row[7] is not None else None,
                    "metadata": json.loads(row[8]) if row[8] else {},
                }
            )
        return results

    def list_kernel_density_names(self) -> list[str]:
        """Return sorted KDE log names."""
        cursor = self.conn.execute(
            "SELECT DISTINCT name FROM kernel_density ORDER BY name ASC"
        )
        return [row[0] for row in cursor.fetchall()]

    def fetch_kernel_density_by_name(self, name: str) -> list[dict[str, Any]]:
        """Fetch KDE metadata rows for a given log name ordered by step."""
        cursor = self.conn.execute(
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
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
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
                    "metadata": json.loads(row[10]) if row[10] else {},
                }
            )
        return results

    def fetch_media_info_by_hashes(
        self, hashes: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch media info (format/size) by hash values."""
        hashes = list(hashes)
        if not hashes:
            return {}

        placeholders = ",".join("?" * len(hashes))
        cursor = self.conn.execute(
            f"""
            SELECT media_hash, format, size_bytes
            FROM media
            WHERE media_hash IN ({placeholders})
            """,
            hashes,
        )
        info: dict[str, dict[str, Any]] = {}
        for media_hash, fmt, size in cursor.fetchall():
            info[media_hash] = {
                "format": fmt,
                "size_bytes": int(size) if size is not None else None,
            }
        return info

    def get_latest_step(self) -> dict[str, Any] | None:
        """Return latest step record."""
        cursor = self.conn.execute(
            "SELECT step, global_step, timestamp FROM steps ORDER BY step DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "step": int(row[0]),
            "global_step": int(row[1]) if row[1] is not None else None,
            "timestamp": int(row[2]) if row[2] is not None else None,
        }

    def get_latest_step_value(self) -> int | None:
        """Return latest step number or None."""
        cursor = self.conn.execute("SELECT MAX(step) FROM steps")
        row = cursor.fetchone()
        if row and row[0] is not None:
            return int(row[0])
        return None

    def flush_all(self):
        """Flush all buffers"""
        self._flush_steps()
        self._flush_media()
        self._flush_tables()
        self.logger.debug("Flushed all SQLite buffers")

    def close(self):
        """Close database connection - flush first"""
        self.flush_all()
        if self.conn:
            self.conn.close()
            self.logger.debug("SQLite metadata storage closed")
