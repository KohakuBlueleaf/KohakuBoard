"""
SQLite-based KV storage for efficient binary blob management.

Optimized for write-once, read-many media blobs with:
- Large page size for blob efficiency
- WAL mode for concurrent readers
- Memory-mapped I/O for fast access
- Incremental BLOB I/O for memory efficiency
"""

import sqlite3
from pathlib import Path
from typing import Optional

from kohakuboard.logger import get_logger


class SQLiteKVStorage:
    """
    SQLite-based key-value storage optimized for media blobs.

    Uses incremental BLOB I/O (blobopen) for memory-efficient streaming.
    All WAL/SHM files are contained in the same directory as the DB file.

    File location: {board_dir}/media/blobs.db
    Table schema: (key TEXT PRIMARY KEY, value BLOB, size INTEGER)
    """

    def __init__(self, db_path: Path, readonly: bool = False):
        """
        Initialize SQLite KV storage.

        Args:
            db_path: Path to the SQLite database file
            readonly: Open database in read-only mode
        """
        self.db_path = Path(db_path)
        self.readonly = readonly
        self.conn: Optional[sqlite3.Connection] = None

        # Initialize file-only logger (no stdout output)
        log_file = self.db_path.parent.parent / "logs" / "sqlite_kv.log"
        self.logger = get_logger("SQLITE_KV", file_only=True, log_file=log_file)

        # Create parent directory if needed
        if not readonly:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._open()

    def _open(self):
        """Open the SQLite database with optimized PRAGMAs."""
        try:
            # Open connection
            uri = f"file:{self.db_path}{'?mode=ro' if self.readonly else ''}"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

            if not self.readonly:
                # Set optimized PRAGMAs for media blob storage
                cursor = self.conn.cursor()

                # Page size: 16KB for large blobs (must be set before any tables)
                cursor.execute("PRAGMA page_size = 16384")

                # WAL mode: Enables concurrent readers
                cursor.execute("PRAGMA journal_mode = WAL")

                # Synchronous: NORMAL for balance (FULL for max durability)
                cursor.execute("PRAGMA synchronous = NORMAL")

                # WAL autocheckpoint: Checkpoint every 1000 pages (~16MB)
                cursor.execute("PRAGMA wal_autocheckpoint = 1000")

                # Memory-mapped I/O: 256MB for fast reads
                cursor.execute("PRAGMA mmap_size = 268435456")

                # Page cache: 64MB (negative = KB)
                cursor.execute("PRAGMA cache_size = -65536")

                # Temp storage: Use memory for temp tables
                cursor.execute("PRAGMA temp_store = MEMORY")

                # Auto-vacuum: Incremental to reclaim space
                cursor.execute("PRAGMA auto_vacuum = INCREMENTAL")

                # Create table with simple schema
                # Note: Must have rowid for blobopen() to work
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS kv_store (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        value BLOB NOT NULL,
                        size INTEGER NOT NULL
                    )
                """)

                # Create index on key for fast lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_kv_key ON kv_store(key)
                """)

                self.conn.commit()
                self.logger.debug(f"Opened SQLite KV storage at {self.db_path}")
            else:
                self.logger.debug(f"Opened SQLite KV storage (readonly) at {self.db_path}")

        except Exception as e:
            self.logger.error(f"Failed to open SQLite KV at {self.db_path}: {e}")
            raise

    def put(self, key: str, data: bytes) -> bool:
        """
        Store data with the given key using incremental BLOB I/O.

        Args:
            key: Storage key
            data: Binary data

        Returns:
            True if data was written, False if key already exists
        """
        if self.conn is None:
            raise RuntimeError("SQLite connection is not open")

        if self.readonly:
            raise RuntimeError("Cannot write in readonly mode")

        try:
            cursor = self.conn.cursor()

            # Check if key exists (for deduplication)
            cursor.execute("SELECT 1 FROM kv_store WHERE key = ? LIMIT 1", (key,))
            if cursor.fetchone() is not None:
                self.logger.debug(f"Key already exists: {key}")
                return False

            data_size = len(data)

            # Insert with zeroblob to allocate space
            cursor.execute(
                "INSERT INTO kv_store (key, value, size) VALUES (?, zeroblob(?), ?)",
                (key, data_size, data_size)
            )

            # Get id for the inserted row (this is the rowid)
            row_id = cursor.lastrowid

            # Open blob for incremental write
            blob = self.conn.blobopen("kv_store", "value", row_id, readonly=False)
            try:
                # Write data in chunks
                chunk_size = 1024 * 1024  # 1MB chunks
                offset = 0
                while offset < data_size:
                    chunk = data[offset:offset + chunk_size]
                    blob.write(chunk)
                    offset += chunk_size
            finally:
                blob.close()

            self.conn.commit()
            self.logger.debug(f"Wrote {data_size} bytes to key: {key}")
            return True

        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to write key {key}: {e}")
            raise

    def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve data by key using incremental BLOB I/O.

        Args:
            key: Storage key

        Returns:
            Binary data, or None if key doesn't exist
        """
        if self.conn is None:
            raise RuntimeError("SQLite connection is not open")

        try:
            cursor = self.conn.cursor()

            # Get id and size for the key
            cursor.execute(
                "SELECT id, size FROM kv_store WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            row_id, size = row

            # Open blob for incremental read
            blob = self.conn.blobopen("kv_store", "value", row_id, readonly=True)
            try:
                # Read entire blob
                data = blob.read()
                self.logger.debug(f"Retrieved {len(data)} bytes from key: {key}")
                return data
            finally:
                blob.close()

        except Exception as e:
            self.logger.error(f"Failed to read key {key}: {e}")
            return None

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the database.

        Args:
            key: Storage key

        Returns:
            True if key exists, False otherwise
        """
        if self.conn is None:
            raise RuntimeError("SQLite connection is not open")

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM kv_store WHERE key = ? LIMIT 1", (key,))
            return cursor.fetchone() is not None
        except Exception as e:
            self.logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    def get_size(self, key: str) -> Optional[int]:
        """
        Get size of data for a key.

        Args:
            key: Storage key

        Returns:
            Size in bytes, or None if key doesn't exist
        """
        if self.conn is None:
            raise RuntimeError("SQLite connection is not open")

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT size FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            self.logger.error(f"Failed to get size for key {key}: {e}")
            return None

    def close(self):
        """Close the SQLite connection."""
        if self.conn is not None:
            try:
                # Run checkpoint to merge WAL into main DB
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as e:
                self.logger.warning(f"WAL checkpoint failed: {e}")

            self.conn.close()
            self.conn = None
            self.logger.debug(f"Closed SQLite KV storage at {self.db_path}")

    def sync(self, force: bool = True):
        """
        Flush changes to disk.

        Args:
            force: Force synchronous flush (runs WAL checkpoint)
        """
        if self.conn is not None and force:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                self.logger.debug("Synced SQLite KV storage to disk")
            except Exception as e:
                self.logger.warning(f"Sync failed: {e}")

    def vacuum_incremental(self, pages: int = 100):
        """
        Reclaim unused space incrementally.

        Args:
            pages: Number of pages to free (default: 100)
        """
        if self.conn is not None and not self.readonly:
            try:
                self.conn.execute(f"PRAGMA incremental_vacuum({pages})")
                self.logger.debug(f"Incremental vacuum freed up to {pages} pages")
            except Exception as e:
                self.logger.warning(f"Incremental vacuum failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
