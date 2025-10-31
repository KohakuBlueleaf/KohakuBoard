"""SQLite-based storage for media and table metadata

Uses Python's built-in sqlite3 module for zero overhead and excellent multi-connection support.
Fixed schema - no dynamic columns needed.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from kohakuboard.logger import get_logger


class SQLiteMetadataStorage:
    """SQLite storage for media and table logs

    Benefits:
    - Built-in module (zero dependency overhead)
    - Excellent multi-connection support (WAL mode)
    - Auto-commit for simplicity
    - Fixed schema (media/tables don't need dynamic columns)
    """

    def __init__(self, base_dir: Path):
        """Initialize SQLite metadata storage

        Args:
            base_dir: Base directory for database file
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Setup file-only logger for storage
        log_file = base_dir.parent / "storage.log"
        self.logger = get_logger("STORAGE", file_only=True, log_file=log_file)

        self.db_file = base_dir / "metadata.db"

        # Use WAL mode for better concurrent access
        self.conn = sqlite3.connect(str(self.db_file))
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
        self.step_buffer: List[tuple] = []
        self.table_buffer: List[tuple] = []

        # Flush thresholds
        self.step_flush_threshold = 1000
        self.table_flush_threshold = 100

    def append_step_info(
        self,
        step: int,
        global_step: Optional[int],
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
        global_step: Optional[int],
        name: str,
        media_list: List[Dict[str, Any]],
        caption: Optional[str] = None,
    ) -> List[int]:
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
        global_step: Optional[int],
        name: str,
        table_data: Dict[str, Any],
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
