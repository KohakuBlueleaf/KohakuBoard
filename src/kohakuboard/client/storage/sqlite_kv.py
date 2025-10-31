"""
KohakuVault-based KV storage for efficient binary blob management.

Wrapper around KohakuVault for media blob storage.
KohakuVault provides:
- Rust backend for high performance
- Streaming operations for large files
- Write-back caching for bulk operations
- No size limits
"""

from pathlib import Path
from typing import Optional

from kohakuvault import KVault
from kohakuboard.logger import get_logger


class SQLiteKVStorage:
    """
    KohakuVault-based key-value storage for media blobs.

    Wrapper around KohakuVault (Rust-based SQLite KV store).
    Provides simple dict-like interface with efficient streaming.

    File location: {board_dir}/media/blobs.db
    """

    def __init__(self, db_path: Path, readonly: bool = False):
        """
        Initialize KohakuVault storage.

        Args:
            db_path: Path to the SQLite database file
            readonly: Open database in read-only mode (not used, kept for compatibility)
        """
        self.db_path = Path(db_path)
        self.readonly = readonly

        # Initialize file-only logger (no stdout output)
        log_file = self.db_path.parent.parent / "logs" / "sqlite_kv.log"
        self.logger = get_logger("SQLITE_KV", file_only=True, log_file=log_file)

        # Create parent directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize KohakuVault
        self.vault = KVault(str(self.db_path))
        self.logger.debug(f"Opened KohakuVault at {self.db_path}")

    def put(self, key: str, data: bytes) -> bool:
        """
        Store data with the given key.

        Args:
            key: Storage key
            data: Binary data

        Returns:
            True if data was written, False if key already exists
        """
        try:
            # Check if key exists (for deduplication)
            if key in self.vault:
                self.logger.debug(f"Key already exists: {key}")
                return False

            # Store data using KohakuVault
            self.vault[key] = data
            self.logger.debug(f"Wrote {len(data)} bytes to key: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write key {key}: {e}")
            raise

    def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve data by key.

        Args:
            key: Storage key

        Returns:
            Binary data, or None if key doesn't exist
        """
        try:
            # Get data using KohakuVault
            data = self.vault.get(key)
            if data is not None:
                self.logger.debug(f"Retrieved {len(data)} bytes from key: {key}")
            return data

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
        try:
            return key in self.vault
        except Exception as e:
            self.logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    def close(self):
        """Close the KohakuVault (flushes any cached data)."""
        try:
            # KohakuVault doesn't have explicit close, but let's flush cache if enabled
            if hasattr(self.vault, 'flush_cache'):
                self.vault.flush_cache()
            self.logger.debug(f"Closed KohakuVault at {self.db_path}")
        except Exception as e:
            self.logger.warning(f"Error during KohakuVault close: {e}")

    def sync(self, force: bool = True):
        """
        Flush changes to disk.

        Args:
            force: Force synchronous flush
        """
        try:
            # Flush cache if it's enabled
            if hasattr(self.vault, 'flush_cache'):
                self.vault.flush_cache()
            self.logger.debug("Synced KohakuVault to disk")
        except Exception as e:
            self.logger.warning(f"Sync failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
