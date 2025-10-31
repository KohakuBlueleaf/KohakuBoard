"""Sync worker for incremental remote synchronization

Standalone thread that periodically reads from local storage and syncs
new data to remote server using JSON payloads.
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import orjson
import requests
from loguru import logger

try:
    from lance.dataset import LanceDataset

    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False
    logger.warning("Lance not available, scalar and histogram sync will be skipped")


class SyncWorker:
    """Background worker for incremental sync to remote server

    Periodically checks local SQLite/Lance storage for new data and syncs
    to remote server. Runs in a separate thread and doesn't block logging.

    Features:
    - Periodic polling of local storage
    - Incremental sync (only new data since last sync)
    - Retry queue with exponential backoff
    - Media deduplication (hash-based)
    - Non-blocking (local logging continues on sync failure)
    """

    def __init__(
        self,
        board_dir: Path,
        remote_url: str,
        remote_token: str,
        project: str,
        run_id: str,
        sync_interval: int = 10,
    ):
        """Initialize sync worker

        Args:
            board_dir: Path to board directory
            remote_url: Remote server base URL (e.g., https://board.example.com)
            remote_token: Authentication token
            project: Project name on remote server
            run_id: Run ID
            sync_interval: Sync check interval in seconds (default: 10)
        """
        self.board_dir = Path(board_dir)
        self.remote_url = remote_url.rstrip("/")
        self.remote_token = remote_token
        self.project = project
        self.run_id = run_id
        self.sync_interval = sync_interval

        # Paths
        self.state_file = self.board_dir / "sync_state.json"
        self.sqlite_db = self.board_dir / "data" / "metadata.db"
        self.metrics_dir = self.board_dir / "data" / "metrics"
        self.histograms_dir = self.board_dir / "data" / "histograms"
        self.media_dir = self.board_dir / "media"

        # Sync state
        self.state = self._load_state()

        # Retry queue
        self.retry_queue: List[Dict[str, Any]] = []
        self.max_retries = 5
        self.backoff_base = 2

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.running = False

        # HTTP session
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {remote_token}"
        self.session.headers["Content-Type"] = "application/json"

        # Setup dedicated logger for sync worker (write to file, not stdout)
        self._setup_logger()

        self.logger.info(
            f"SyncWorker initialized: {project}/{run_id} -> {remote_url} "
            f"(interval: {sync_interval}s)"
        )

    def _setup_logger(self):
        """Setup dedicated logger for sync worker that writes to file ONLY"""
        from kohakuboard.logger import get_logger

        log_dir = self.board_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "sync_worker.log"

        # Get file-only logger instance
        self.logger = get_logger("SYNC", file_only=True, log_file=log_file)

    def start(self):
        """Start sync worker thread"""
        if self.running:
            self.logger.warning("SyncWorker already running")
            return

        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sync_loop, daemon=False)
        self.thread.start()
        self.logger.info("SyncWorker thread started")

    def stop(self, timeout: float = 30.0):
        """Stop sync worker thread gracefully

        Args:
            timeout: Max time to wait for thread to finish (seconds)
        """
        if not self.running:
            return

        self.logger.info("Stopping SyncWorker...")
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

            if self.thread.is_alive():
                self.logger.warning("SyncWorker thread did not stop within timeout")
            else:
                self.logger.info("SyncWorker stopped")

        self.running = False

    def _sync_loop(self):
        """Main sync loop (runs in background thread)"""
        self.logger.info("SyncWorker loop started")

        while not self.stop_event.is_set():
            try:
                # Process retry queue first
                self._process_retry_queue()

                # Sync new data
                self._sync_new_data()

            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")

            # Wait for next interval or stop event
            self.stop_event.wait(timeout=self.sync_interval)

        # Final sync before stopping
        try:
            self.logger.info("Final sync before stopping...")
            self._sync_new_data()
        except Exception as e:
            self.logger.error(f"Final sync failed: {e}")

        self.logger.info("SyncWorker loop exited")

    def _sync_new_data(self):
        """Collect and sync new data since last sync"""
        if not self.sqlite_db.exists():
            self.logger.debug("SQLite DB not found, skipping sync")
            return

        # Get latest step from local storage
        latest_step = self._get_latest_local_step()
        if latest_step is None:
            self.logger.debug("No data to sync yet")
            return

        last_synced_step = self.state.get("last_synced_step", -1)

        if latest_step <= last_synced_step:
            self.logger.debug(
                f"No new data (latest: {latest_step}, synced: {last_synced_step})"
            )
            return

        self.logger.info(
            f"Syncing steps {last_synced_step + 1} to {latest_step} "
            f"({latest_step - last_synced_step} new steps)"
        )

        try:
            # Collect new data
            payload = self._collect_sync_payload(last_synced_step + 1, latest_step)

            # Send log data
            response = self._sync_logs(payload)

            # Upload missing media
            missing_media = response.get("missing_media", [])
            if missing_media:
                self.logger.info(f"Uploading {len(missing_media)} missing media files")
                self._sync_media(missing_media)

            # Update state
            self.state["last_synced_step"] = latest_step
            self.state["last_sync_at"] = datetime.now(timezone.utc).isoformat()

            # Mark metadata as synced if it was included
            if payload.get("metadata"):
                self.state["metadata_synced"] = True

            # Update last synced log line
            if payload.get("log_lines"):
                self.state["last_synced_log_line"] = self.state.get(
                    "last_synced_log_line", 0
                ) + len(payload["log_lines"])

            self._save_state()

            self.logger.info(f"Sync completed successfully (step: {latest_step})")

        except Exception as e:
            self.logger.error(f"Sync failed: {e}")
            # Add to retry queue
            self.retry_queue.append(
                {
                    "payload": payload,
                    "attempts": 0,
                    "last_attempt": time.time(),
                    "created_at": time.time(),
                }
            )

    def _collect_sync_payload(self, start_step: int, end_step: int) -> Dict[str, Any]:
        """Collect new data from local storage

        Args:
            start_step: Start step (inclusive)
            end_step: End step (inclusive)

        Returns:
            Sync payload dict
        """
        payload = {
            "sync_range": {"start_step": start_step, "end_step": end_step},
            "steps": [],
            "scalars": {},
            "media": [],
            "tables": [],
            "histograms": [],
        }

        # Collect steps
        payload["steps"] = self._collect_steps(start_step, end_step)

        # Collect scalars from Lance
        payload["scalars"] = self._collect_scalars(start_step, end_step)

        # Collect media from SQLite
        payload["media"] = self._collect_media(start_step, end_step)

        # Collect tables from SQLite
        payload["tables"] = self._collect_tables(start_step, end_step)

        # Collect histograms from Lance
        payload["histograms"] = self._collect_histograms(start_step, end_step)

        # Collect metadata (only on first sync)
        if not self.state.get("metadata_synced", False):
            payload["metadata"] = self._collect_metadata()

        # Collect new log lines
        payload["log_lines"] = self._collect_log_lines()

        return payload

    def _collect_steps(self, start_step: int, end_step: int) -> List[Dict[str, Any]]:
        """Collect steps from SQLite"""
        conn = sqlite3.connect(str(self.sqlite_db))
        try:
            cursor = conn.execute(
                """
                SELECT step, global_step, timestamp
                FROM steps
                WHERE step >= ? AND step <= ?
                ORDER BY step ASC
                """,
                (start_step, end_step),
            )
            steps = [
                {"step": row[0], "global_step": row[1], "timestamp": row[2]}
                for row in cursor.fetchall()
            ]
            return steps
        finally:
            conn.close()

    def _collect_scalars(
        self, start_step: int, end_step: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect scalar metrics from Lance files"""
        scalars = {}

        if not LANCE_AVAILABLE:
            return scalars

        if not self.metrics_dir.exists():
            return scalars

        try:
            for lance_file in self.metrics_dir.glob("*.lance"):
                # Convert filename back to metric name (undo __ escaping)
                metric_name = lance_file.stem.replace("__", "/")

                # Read dataset
                ds = LanceDataset(str(lance_file))

                # Filter by step range
                table = ds.to_table(
                    filter=f"step >= {start_step} and step <= {end_step}"
                )

                if len(table) == 0:
                    continue

                # Convert to list of {step, value}
                steps = table["step"].to_pylist()
                values = table["value"].to_pylist()

                scalars[metric_name] = [
                    {"step": s, "value": v} for s, v in zip(steps, values)
                ]

        except Exception as e:
            self.logger.error(f"Failed to collect scalars: {e}")

        return scalars

    def _collect_media(self, start_step: int, end_step: int) -> List[Dict[str, Any]]:
        """Collect media metadata from SQLite"""
        conn = sqlite3.connect(str(self.sqlite_db))
        try:
            cursor = conn.execute(
                """
                SELECT id, media_hash, format, step, global_step, name,
                       caption, type, width, height, size_bytes
                FROM media
                WHERE step >= ? AND step <= ?
                ORDER BY step ASC
                """,
                (start_step, end_step),
            )

            media = []
            for row in cursor.fetchall():
                media.append(
                    {
                        "id": row[0],
                        "media_hash": row[1],
                        "format": row[2],
                        "step": row[3],
                        "global_step": row[4],
                        "name": row[5],
                        "caption": row[6],
                        "type": row[7],
                        "width": row[8],
                        "height": row[9],
                        "size_bytes": row[10],
                    }
                )
            return media

        finally:
            conn.close()

    def _collect_tables(self, start_step: int, end_step: int) -> List[Dict[str, Any]]:
        """Collect tables from SQLite"""
        conn = sqlite3.connect(str(self.sqlite_db))
        try:
            cursor = conn.execute(
                """
                SELECT step, global_step, name, columns, column_types, rows
                FROM tables
                WHERE step >= ? AND step <= ?
                ORDER BY step ASC
                """,
                (start_step, end_step),
            )

            tables = []
            for row in cursor.fetchall():
                tables.append(
                    {
                        "step": row[0],
                        "global_step": row[1],
                        "name": row[2],
                        "columns": json.loads(row[3]) if row[3] else [],
                        "column_types": json.loads(row[4]) if row[4] else [],
                        "rows": json.loads(row[5]) if row[5] else [],
                    }
                )
            return tables

        finally:
            conn.close()

    def _collect_histograms(
        self, start_step: int, end_step: int
    ) -> List[Dict[str, Any]]:
        """Collect histograms from Lance files"""
        histograms = []

        if not LANCE_AVAILABLE:
            return histograms

        if not self.histograms_dir.exists():
            return histograms

        try:
            for lance_file in self.histograms_dir.glob("*.lance"):
                # Read dataset
                ds = LanceDataset(str(lance_file))

                # Filter by step range
                table = ds.to_table(
                    filter=f"step >= {start_step} and step <= {end_step}"
                )

                if len(table) == 0:
                    continue

                # Convert to list of dicts
                for i in range(len(table)):
                    counts = table["counts"][i].as_py()
                    min_val = float(table["min"][i].as_py())
                    max_val = float(table["max"][i].as_py())
                    num_bins = len(counts)

                    # Reconstruct bin edges
                    bin_edges = np.linspace(min_val, max_val, num_bins + 1).tolist()

                    # Determine precision from filename
                    precision = "i32" if "_i32" in lance_file.stem else "u8"

                    histograms.append(
                        {
                            "step": int(table["step"][i].as_py()),
                            "global_step": (
                                int(table["global_step"][i].as_py())
                                if table["global_step"][i].as_py()
                                else None
                            ),
                            "name": table["name"][i].as_py(),
                            "bins": bin_edges,
                            "counts": counts,
                            "precision": precision,
                        }
                    )

        except Exception as e:
            self.logger.error(f"Failed to collect histograms: {e}")

        return histograms

    def _collect_metadata(self) -> Optional[Dict[str, Any]]:
        """Collect metadata.json

        Returns:
            Metadata dict or None if file doesn't exist
        """
        metadata_file = self.board_dir / "metadata.json"
        if not metadata_file.exists():
            self.logger.debug("metadata.json not found")
            return None

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            self.logger.debug("Collected metadata.json")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to read metadata.json: {e}")
            return None

    def _collect_log_lines(self) -> List[str]:
        """Collect new log lines from output.log

        Returns:
            List of new log lines (since last sync)
        """
        log_file = self.board_dir / "logs" / "output.log"
        if not log_file.exists():
            return []

        last_synced_line = self.state.get("last_synced_log_line", 0)

        try:
            with open(log_file, "r") as f:
                all_lines = f.readlines()

            # Get new lines (strip newlines since we add them back on server)
            new_lines = [line.rstrip("\n") for line in all_lines[last_synced_line:]]

            if new_lines:
                self.logger.debug(
                    f"Collected {len(new_lines)} new log lines "
                    f"(from line {last_synced_line} to {last_synced_line + len(new_lines)})"
                )

            return new_lines
        except Exception as e:
            self.logger.error(f"Failed to read output.log: {e}")
            return []

    def _sync_logs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send log data to remote server

        Args:
            payload: Sync payload

        Returns:
            Response dict with missing_media list

        Raises:
            requests.RequestException: On HTTP error
        """
        url = f"{self.remote_url}/api/projects/{self.project}/runs/{self.run_id}/log"

        # Serialize with orjson (faster than json.dumps)
        json_bytes = orjson.dumps(payload)

        response = self.session.post(
            url,
            data=json_bytes,
            timeout=60,
        )

        response.raise_for_status()
        return response.json()

    def _sync_media(self, missing_hashes: List[str]):
        """Upload missing media files to remote server

        Args:
            missing_hashes: List of media hashes to upload
        """
        if not missing_hashes:
            return

        url = f"{self.remote_url}/api/projects/{self.project}/runs/{self.run_id}/media"

        # Collect files to upload
        files_to_upload = []
        for media_hash in missing_hashes:
            # Find file with this hash (may have different extensions)
            matching_files = list(self.media_dir.glob(f"{media_hash}.*"))

            if not matching_files:
                self.logger.warning(f"Media file not found for hash: {media_hash}")
                continue

            # Use first match
            media_file = matching_files[0]
            files_to_upload.append(media_file)

        if not files_to_upload:
            self.logger.warning("No media files to upload")
            return

        # Prepare multipart upload
        files = []
        handles = []

        try:
            for media_file in files_to_upload:
                handle = open(media_file, "rb")
                handles.append(handle)
                files.append(
                    ("files", (media_file.name, handle, "application/octet-stream"))
                )

            # Upload with custom headers (remove JSON Content-Type for multipart)
            headers = {"Authorization": self.session.headers.get("Authorization")}
            # Don't include Content-Type - let requests set it for multipart/form-data

            response = requests.post(
                url,
                files=files,
                headers=headers,
                timeout=300,
            )

            response.raise_for_status()
            self.logger.info(f"Uploaded {len(files_to_upload)} media files")

        finally:
            # Close all handles
            for handle in handles:
                handle.close()

    def _get_latest_local_step(self) -> Optional[int]:
        """Get latest step from local SQLite

        Returns:
            Latest step number or None if no data
        """
        conn = sqlite3.connect(str(self.sqlite_db))
        try:
            cursor = conn.execute("SELECT MAX(step) FROM steps")
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None
        finally:
            conn.close()

    def _process_retry_queue(self):
        """Process retry queue with exponential backoff"""
        for retry_item in self.retry_queue[:]:
            # Check max retries
            if retry_item["attempts"] >= self.max_retries:
                self.logger.error(
                    f"Max retries ({self.max_retries}) exceeded for sync payload, "
                    f"giving up (created at: {retry_item['created_at']})"
                )
                self.retry_queue.remove(retry_item)
                continue

            # Check if ready to retry (exponential backoff)
            backoff = self._calc_backoff(retry_item["attempts"])
            if time.time() - retry_item["last_attempt"] < backoff:
                continue  # Not ready yet

            # Retry
            self.logger.info(
                f"Retrying sync (attempt {retry_item['attempts'] + 1}/{self.max_retries})"
            )

            try:
                payload = retry_item["payload"]
                response = self._sync_logs(payload)

                # Upload missing media
                missing_media = response.get("missing_media", [])
                if missing_media:
                    self._sync_media(missing_media)

                # Success - remove from queue
                self.retry_queue.remove(retry_item)
                self.logger.info("Retry successful")

            except Exception as e:
                self.logger.warning(f"Retry failed: {e}")
                retry_item["attempts"] += 1
                retry_item["last_attempt"] = time.time()

    def _calc_backoff(self, attempts: int) -> float:
        """Calculate exponential backoff delay

        Args:
            attempts: Number of attempts so far

        Returns:
            Delay in seconds (2, 4, 8, 16, 32, ...)
        """
        return self.backoff_base**attempts

    def _load_state(self) -> Dict[str, Any]:
        """Load sync state from JSON file

        Returns:
            State dict with defaults
        """
        if not self.state_file.exists():
            return {
                "last_synced_step": -1,
                "last_synced_log_line": 0,
                "metadata_synced": False,
                "last_sync_at": None,
                "remote_url": self.remote_url,
                "project": self.project,
                "run_id": self.run_id,
            }

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                # Add new fields if missing (for backward compatibility)
                state.setdefault("last_synced_log_line", 0)
                state.setdefault("metadata_synced", False)
                return state
        except Exception as e:
            self.logger.warning(f"Failed to load sync state: {e}, using defaults")
            return {
                "last_synced_step": -1,
                "last_synced_log_line": 0,
                "metadata_synced": False,
                "last_sync_at": None,
                "remote_url": self.remote_url,
                "project": self.project,
                "run_id": self.run_id,
            }

    def _save_state(self):
        """Save sync state to JSON file"""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save sync state: {e}")

    def force_sync(self):
        """Force an immediate sync (useful for testing or manual triggers)"""
        self.logger.info("Force sync triggered")
        try:
            self._sync_new_data()
        except Exception as e:
            self.logger.error(f"Force sync failed: {e}")
