"""Deletion service - delete data from boards"""

import sqlite3
from pathlib import Path

from kohakuvault import ColumnVault, KVault


class DeletionService:
    """Service for deleting data from boards"""

    @staticmethod
    def delete_metric_by_step_range(board_path: Path, metric: str, start_step: int, end_step: int) -> int:
        """Delete metric values in step range

        Args:
            board_path: Path to board
            metric: Metric name
            start_step: Start step (inclusive)
            end_step: End step (inclusive)

        Returns:
            Number of rows deleted
        """
        metric_escaped = metric.replace("/", "__")
        metric_db = board_path / "data" / "metrics" / f"{metric_escaped}.db"

        if not metric_db.exists():
            return 0

        # Read current data
        cv = ColumnVault(str(metric_db))
        steps = list(cv["step"])
        global_steps = list(cv["global_step"])
        timestamps = list(cv["timestamp"])
        values = list(cv["value"])

        # Find indices to keep (inverse of delete range)
        indices_to_keep = [
            i for i, step in enumerate(steps)
            if not (start_step <= step <= end_step)
        ]

        deleted_count = len(steps) - len(indices_to_keep)

        if deleted_count == 0:
            return 0

        # Recreate file with filtered data
        import os
        temp_db = metric_db.with_suffix('.db.tmp')

        # Create new DB with kept data
        cv_new = ColumnVault(str(temp_db))
        cv_new.create_column("step", "i64")
        cv_new.create_column("global_step", "i64")
        cv_new.create_column("timestamp", "i64")
        cv_new.create_column("value", "f64")

        # Write kept data
        if indices_to_keep:
            cv_new["step"].extend([steps[i] for i in indices_to_keep])
            cv_new["global_step"].extend([global_steps[i] for i in indices_to_keep])
            cv_new["timestamp"].extend([timestamps[i] for i in indices_to_keep])
            cv_new["value"].extend([values[i] for i in indices_to_keep])

        cv_new.close()
        cv.close()

        # Replace original with new
        os.replace(temp_db, metric_db)

        return deleted_count

    @staticmethod
    def delete_entire_metric(board_path: Path, metric: str) -> bool:
        """Delete entire metric file

        Args:
            board_path: Path to board
            metric: Metric name

        Returns:
            True if deleted
        """
        metric_escaped = metric.replace("/", "__")
        metric_db = board_path / "data" / "metrics" / f"{metric_escaped}.db"

        if metric_db.exists():
            metric_db.unlink()
            return True

        return False

    @staticmethod
    def delete_media_entry(board_path: Path, media_hash: str, media_format: str) -> bool:
        """Delete a specific media entry and its binary

        Args:
            board_path: Path to board
            media_hash: Media hash
            media_format: Media format (png, jpg, etc.)

        Returns:
            True if deleted
        """
        # Delete from KVault
        kv_path = board_path / "media" / "blobs.db"
        if kv_path.exists():
            kv = KVault(str(kv_path))
            filename = f"{media_hash}.{media_format}"
            if filename in kv:
                del kv[filename]

        # Delete metadata from SQLite
        db_path = board_path / "data" / "metadata.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            conn.execute("DELETE FROM media WHERE media_hash = ?", (media_hash,))
            conn.commit()
            conn.close()

        return True

    @staticmethod
    def delete_media_log(board_path: Path, media_name: str) -> int:
        """Delete all media entries for a media log name

        Args:
            board_path: Path to board
            media_name: Media log name

        Returns:
            Number of entries deleted
        """
        db_path = board_path / "data" / "metadata.db"
        if not db_path.exists():
            return 0

        conn = sqlite3.connect(str(db_path))

        # Get media hashes to delete from KVault
        cursor = conn.execute("SELECT media_hash, format FROM media WHERE name = ?", (media_name,))
        media_files = cursor.fetchall()

        # Delete from metadata
        cursor = conn.execute("DELETE FROM media WHERE name = ?", (media_name,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        # Delete binary files from KVault
        kv_path = board_path / "media" / "blobs.db"
        if kv_path.exists() and media_files:
            kv = KVault(str(kv_path))
            for media_hash, fmt in media_files:
                filename = f"{media_hash}.{fmt}"
                if filename in kv:
                    del kv[filename]
            kv.close()

        return deleted_count
