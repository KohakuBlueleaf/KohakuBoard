"""SQLite-based storage for media and table metadata

Uses Python's built-in sqlite3 module for zero overhead and excellent multi-connection support.
Fixed schema - no dynamic columns needed.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


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

        # Media table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS media (
                step INTEGER NOT NULL,
                global_step INTEGER,
                name TEXT NOT NULL,
                caption TEXT,
                media_id TEXT,
                type TEXT,
                filename TEXT,
                path TEXT,
                size_bytes INTEGER,
                format TEXT,
                width INTEGER,
                height INTEGER
            )
        """
        )

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

    def __init__(self, base_dir: Path):
        """Initialize SQLite metadata storage

        Args:
            base_dir: Base directory for database file
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.db_file = base_dir / "metadata.db"

        # Use WAL mode for better concurrent access
        self.conn = sqlite3.connect(str(self.db_file))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes

        # Create tables
        self._init_tables()

        # Buffers for batching (AGGRESSIVE)
        self.step_buffer: List[tuple] = []
        self.media_buffer: List[tuple] = []
        self.table_buffer: List[tuple] = []

        # Flush thresholds - batch aggressively since SQLite is fast
        self.step_flush_threshold = 1000
        self.media_flush_threshold = 100
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
        logger.debug(f"Flushed {len(self.step_buffer)} step records to SQLite")
        self.step_buffer.clear()

    def append_media(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        media_list: List[Dict[str, Any]],
        caption: Optional[str] = None,
    ):
        """Append media log entry (batched)

        Args:
            step: Auto-increment step
            global_step: Explicit global step
            name: Media log name
            media_list: List of media metadata dicts
            caption: Optional caption
        """
        for media_meta in media_list:
            row = (
                step,
                global_step,
                name,
                caption or "",
                media_meta.get("media_id", ""),
                media_meta.get("type", ""),
                media_meta.get("filename", ""),
                media_meta.get("path", ""),
                media_meta.get("size_bytes", 0),
                media_meta.get("format", ""),
                media_meta.get("width"),
                media_meta.get("height"),
            )
            self.media_buffer.append(row)

        # Don't auto-flush - writer will call flush() periodically

    def _flush_media(self):
        """Flush media buffer"""
        if not self.media_buffer:
            return

        self.conn.executemany(
            """
            INSERT INTO media (
                step, global_step, name, caption, media_id, type,
                filename, path, size_bytes, format, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            self.media_buffer,
        )
        self.conn.commit()
        logger.debug(f"Flushed {len(self.media_buffer)} media rows to SQLite")
        self.media_buffer.clear()

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
        logger.debug(f"Flushed {len(self.table_buffer)} table rows to SQLite")
        self.table_buffer.clear()

    def flush_all(self):
        """Flush all buffers"""
        self._flush_steps()
        self._flush_media()
        self._flush_tables()
        logger.debug("Flushed all SQLite buffers")

    def close(self):
        """Close database connection - flush first"""
        self.flush_all()
        if self.conn:
            self.conn.close()
            logger.debug("SQLite metadata storage closed")
