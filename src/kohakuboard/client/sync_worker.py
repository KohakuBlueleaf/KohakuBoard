"""Sync worker for incremental remote synchronization

Standalone thread that periodically reads from local storage and syncs
new data to remote server using JSON payloads.
"""

import hashlib
import io
import json
import sqlite3
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import orjson
import requests

from kohakuvault import ColumnVault, KVault


class SyncWorker:
    """Background worker for incremental sync to remote server

    Periodically checks local storage (three-tier SQLite architecture) for new data
    and syncs to remote server. Runs in a separate thread and doesn't block logging.

    Reads from:
    1. KohakuVault ColumnVault - Metrics (blob-based columnar)
    2. KohakuVault KVault - Media blobs (K-V table with B+Tree index)
    3. Standard SQLite - Metadata (traditional tables)

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
        storage: Any = None,
        storage_lock: Optional[threading.RLock] = None,
        media_kv: Optional[KVault] = None,
        memory_mode: bool = False,
        log_buffers: Optional[dict[str, io.StringIO]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Initialize sync worker

        Args:
            board_dir: Path to board directory
            remote_url: Remote server base URL (e.g., https://board.example.com)
            remote_token: Authentication token
            project: Project name on remote server
            run_id: Run ID
            sync_interval: Sync check interval in seconds (default: 10)
            storage: Optional shared HybridStorage instance (from writer process)
            storage_lock: Optional threading lock coordinating storage access
            media_kv: Optional shared KVault instance for media data
            memory_mode: Whether writer is operating in memory-only mode
            log_buffers: Optional dict of in-memory log buffers for stdout/stderr
            metadata: Optional metadata snapshot for memory mode
        """
        self.board_dir = Path(board_dir)
        self.remote_url = remote_url.rstrip("/")
        self.remote_token = remote_token
        self.project = project
        self.run_id = run_id
        self.sync_interval = sync_interval
        self.storage = storage
        self.storage_lock = storage_lock or threading.RLock()
        self.memory_mode = memory_mode
        self._uses_shared_storage = storage is not None
        self.initial_metadata = deepcopy(metadata) if metadata else None
        self._memory_state: Optional[dict[str, Any]] = None
        self.log_buffers = log_buffers

        # Paths
        self.state_file = (
            self.board_dir / "sync_state.json" if not self.memory_mode else None
        )
        self.sqlite_db = self.board_dir / "data" / "metadata.db"
        self.metrics_dir = self.board_dir / "data" / "metrics"
        self.histograms_dir = self.board_dir / "data" / "histograms"
        self.media_dir = self.board_dir / "media"
        self.media_kv_path = self.board_dir / "media" / "blobs.db"

        # Initialize KVault storage for reading media (shared if provided)
        self.media_kv = media_kv
        self._owns_media_kv = False
        if self.media_kv is None and self.media_kv_path.exists():
            self.media_kv = KVault(str(self.media_kv_path))
            self._owns_media_kv = True

        # Sync state
        self.state = self._load_state()

        # Retry queue
        self.retry_queue: list[dict[str, Any]] = []
        self.max_retries = 5
        self.backoff_base = 2

        # Thread control
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False

        # HTTP session
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {remote_token}"
        self.session.headers["Content-Type"] = "application/json"

        # Setup dedicated logger for sync worker (write to file, not stdout)
        self._setup_logger()

        storage_mode = "shared" if self._uses_shared_storage else "filesystem"
        self.logger.info(
            f"SyncWorker initialized: {project}/{run_id} -> {remote_url} "
            f"(interval: {sync_interval}s, storage={storage_mode})"
        )

    def _setup_logger(self):
        """Setup dedicated logger for sync worker that writes to file ONLY"""
        from kohakuboard.logger import get_logger

        if self.memory_mode:
            self.logger = get_logger("SYNC", drop=True)
            return

        log_dir = self.board_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "sync_worker.log"
        self.logger = get_logger("SYNC", file_only=True, log_file=log_file)

    def start(self):
        """Start sync worker thread"""
        if self.running:
            self.logger.warning("SyncWorker already running")
            return

        self.running = True
        self.stop_event.clear()

        # Perform initial sync immediately to create remote board
        self._initial_sync()

        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
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

        # Clean up SQLite KV connection if owned
        if self._owns_media_kv and self.media_kv is not None:
            try:
                self.logger.debug("Closing SQLite KV connection...")
                self.media_kv.close()
                self.media_kv = None
                self.logger.debug("SQLite KV connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing SQLite KV: {e}")

        self.running = False

    def _initial_sync(self):
        """Perform initial sync to create remote board immediately

        This uploads metadata.json to create the board on the remote server
        as soon as the Board is initialized, without waiting for first log.
        """
        try:
            # Check if metadata already synced
            if self.state.get("metadata_synced", False):
                self.logger.debug("Metadata already synced, skipping initial sync")
                return

            # Collect metadata
            metadata = self._collect_metadata()
            if not metadata:
                self.logger.warning("No metadata.json found, skipping initial sync")
                return

            # Create minimal payload with just metadata (no data yet)
            payload = {
                "sync_range": {"start_step": 0, "end_step": -1},
                "steps": [],
                "scalars": {},
                "media": [],
                "tables": [],
                "histograms": [],
                "metadata": metadata,
                "log_files": {},
            }

            # Send to remote server
            self.logger.info("Performing initial sync to create remote board...")
            response = self._sync_logs(payload)

            # Mark metadata as synced
            self.state["metadata_synced"] = True
            self.state["last_sync_at"] = datetime.now(timezone.utc).isoformat()
            self._save_state()

            self.logger.info("Initial sync completed - remote board created")

        except Exception as e:
            self.logger.warning(f"Initial sync failed (will retry later): {e}")
            # Don't raise - sync worker should still start even if initial sync fails

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
        if not self._uses_shared_storage and not self.sqlite_db.exists():
            self.logger.debug("SQLite DB not found, skipping sync")
            return

        # Determine latest step from local storage
        if self._uses_shared_storage:
            with self.storage_lock:
                self.storage.flush_all()
                latest_info = self.storage.metadata_storage.get_latest_step()
                latest_step = latest_info["step"] if latest_info else None
        else:
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
            start_step = last_synced_step + 1
            end_step = latest_step

            # Collect new data
            if self._uses_shared_storage:
                with self.storage_lock:
                    payload = self._collect_sync_payload_from_storage(
                        start_step, end_step
                    )
                if not self.state.get("metadata_synced", False):
                    payload["metadata"] = self._collect_metadata()
            else:
                payload = self._collect_sync_payload(start_step, end_step)

            # Collect log hashes outside the storage lock
            if "log_hashes" not in payload:
                payload["log_hashes"] = self._collect_log_hashes()

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

            # Upload changed log files (compare hashes)
            # Server always returns log_hashes (even if empty dict on first sync)
            if "log_hashes" in response:
                self._upload_changed_log_files(
                    payload.get("log_hashes", {}), response.get("log_hashes", {})
                )

            if self.memory_mode and self._uses_shared_storage:
                with self.storage_lock:
                    prune_fn = getattr(self.storage, "prune_up_to", None)
                    if callable(prune_fn):
                        prune_fn(latest_step)

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

    def _collect_sync_payload(self, start_step: int, end_step: int) -> dict[str, Any]:
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

        # Collect scalars from ColumnVault
        payload["scalars"] = self._collect_scalars(start_step, end_step)

        # Collect media from SQLite
        payload["media"] = self._collect_media(start_step, end_step)

        # Collect tables from SQLite
        payload["tables"] = self._collect_tables(start_step, end_step)

        # Collect histograms from ColumnVault
        payload["histograms"] = self._collect_histograms(start_step, end_step)

        # Collect metadata (only on first sync)
        if not self.state.get("metadata_synced", False):
            payload["metadata"] = self._collect_metadata()

        # Collect log file hashes for change detection
        payload["log_hashes"] = self._collect_log_hashes()

        return payload

    def _collect_sync_payload_from_storage(
        self, start_step: int, end_step: int
    ) -> dict[str, Any]:
        """Collect sync payload using shared storage (lock must be held)."""
        if not self.storage:
            raise RuntimeError("Shared storage not available")

        steps = self.storage.metadata_storage.fetch_steps_range(start_step, end_step)
        scalars = self.storage.metrics_storage.collect_metrics_range(
            start_step, end_step
        )
        media = self.storage.metadata_storage.fetch_media_range(start_step, end_step)
        tables = self.storage.metadata_storage.fetch_tables_range(start_step, end_step)
        histograms = self.storage.histogram_storage.collect_histograms_range(
            start_step, end_step
        )

        return {
            "sync_range": {"start_step": start_step, "end_step": end_step},
            "steps": steps,
            "scalars": scalars,
            "media": media,
            "tables": tables,
            "histograms": histograms,
        }

    def _collect_steps(self, start_step: int, end_step: int) -> list[dict[str, Any]]:
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
    ) -> dict[str, list[dict[str, Any]]]:
        """Collect scalar metrics from ColumnVault DB files"""
        scalars = {}

        if not self.metrics_dir.exists():
            return scalars

        try:
            for db_file in self.metrics_dir.glob("*.db"):
                # Convert filename back to metric name (undo __ escaping)
                metric_name = db_file.stem.replace("__", "/")

                # Read ColumnVault dataset
                cv = ColumnVault(str(db_file))

                # Read all data
                all_steps = list(cv["step"])
                all_values = list(cv["value"])

                # Filter by step range
                filtered_data = [
                    {"step": s, "value": v}
                    for s, v in zip(all_steps, all_values)
                    if start_step <= s <= end_step
                ]

                if not filtered_data:
                    continue

                scalars[metric_name] = filtered_data

        except Exception as e:
            self.logger.error(f"Failed to collect scalars: {e}")

        return scalars

    def _collect_media(self, start_step: int, end_step: int) -> list[dict[str, Any]]:
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

    def _collect_tables(self, start_step: int, end_step: int) -> list[dict[str, Any]]:
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
    ) -> list[dict[str, Any]]:
        """Collect histograms from ColumnVault DB files"""
        histograms = []

        if not self.histograms_dir.exists():
            return histograms

        try:
            import struct

            for db_file in self.histograms_dir.glob("*.db"):
                parts = db_file.stem.split("_")
                if parts and parts[-1] in ("u8", "i32"):
                    precision = parts[-1]
                elif db_file.stem.endswith("_i32"):
                    precision = "i32"
                else:
                    precision = "u8"

                expected_size = None
                if len(parts) >= 3 and parts[-2].isdigit():
                    bin_count = int(parts[-2])
                    expected_size = bin_count if precision == "u8" else bin_count * 4

                try:
                    cv = ColumnVault(str(db_file))
                    all_steps = list(cv["step"])
                    all_global_steps = list(cv["global_step"])
                    all_names = list(cv["name"])
                    all_counts_bytes = list(cv["counts"])
                    all_mins = list(cv["min"])
                    all_maxs = list(cv["max"])
                except Exception as exc:
                    self.logger.error(f"Failed to read histogram DB {db_file}: {exc}")
                    continue
                finally:
                    try:
                        cv.close()
                    except Exception:
                        pass

                iterator = zip(
                    all_steps,
                    all_global_steps,
                    all_names,
                    all_counts_bytes,
                    all_mins,
                    all_maxs,
                )

                for (
                    step,
                    global_step,
                    raw_name,
                    counts_bytes,
                    min_val,
                    max_val,
                ) in iterator:
                    if not (start_step <= step <= end_step):
                        continue

                    if expected_size and len(counts_bytes) != expected_size:
                        self.logger.warning(
                            f"Counts size mismatch for {db_file.name}: "
                            f"expected {expected_size}, got {len(counts_bytes)}"
                        )

                    if precision == "u8":
                        counts = list(
                            struct.unpack(f"{len(counts_bytes)}B", counts_bytes)
                        )
                    else:
                        num_bins = len(counts_bytes) // 4
                        if num_bins == 0:
                            continue
                        counts = list(struct.unpack(f"<{num_bins}i", counts_bytes))

                    if not counts:
                        continue

                    bins = np.linspace(min_val, max_val, len(counts) + 1).tolist()
                    name = (
                        raw_name.decode("utf-8")
                        if isinstance(raw_name, (bytes, bytearray))
                        else str(raw_name)
                    )

                    histograms.append(
                        {
                            "step": int(step),
                            "global_step": (
                                int(global_step) if global_step is not None else None
                            ),
                            "name": name,
                            "bins": bins,
                            "counts": counts,
                            "precision": precision,
                        }
                    )

        except Exception as e:
            self.logger.error(f"Failed to collect histograms: {e}")

        return histograms

    def _collect_metadata(self) -> dict[str, Any] | None:
        """Collect metadata.json

        Returns:
            Metadata dict or None if file doesn't exist
        """
        metadata_file = self.board_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                self.logger.debug("Collected metadata.json")
                return metadata
            except Exception as e:
                self.logger.error(f"Failed to read metadata.json: {e}")

        if self.initial_metadata:
            self.logger.debug("Using in-memory metadata snapshot")
            return deepcopy(self.initial_metadata)

        self.logger.debug("metadata.json not found and no in-memory metadata")
        return None

    def _collect_log_hashes(self) -> dict[str, str]:
        """Collect hashes of ALL log files for change detection

        Returns:
            Dict of {filename: hash}
        """
        if self.memory_mode:
            if not self.log_buffers:
                return {}

            hashes = {}
            with self.storage_lock:
                for log_name, buffer in self.log_buffers.items():
                    content = buffer.getvalue().encode("utf-8")
                    if content:
                        hashes[log_name] = hashlib.sha256(content).hexdigest()
            return hashes

        logs_dir = self.board_dir / "logs"
        if not logs_dir.exists():
            return {}

        log_hashes = {}

        # All log files to sync
        log_names = [
            "output.log",
            "board.log",
            "writer.log",
            "sync_worker.log",
            "storage.log",
        ]

        for log_name in log_names:
            log_path = logs_dir / log_name
            if not log_path.exists():
                continue

            try:
                # Read file and calculate hash
                with open(log_path, "rb") as f:
                    content_bytes = f.read()

                file_hash = hashlib.sha256(content_bytes).hexdigest()
                log_hashes[log_name] = file_hash

            except Exception as e:
                self.logger.error(f"Failed to hash {log_name}: {e}")

        return log_hashes

    def _upload_changed_log_files(
        self, local_hashes: dict[str, str], remote_hashes: dict[str, str]
    ):
        """Upload log files that have changed

        Args:
            local_hashes: Local log file hashes
            remote_hashes: Remote log file hashes from server (may be empty on first sync)
        """
        if self.memory_mode:
            if not self.log_buffers:
                return

            with self.storage_lock:
                for log_name, local_hash in local_hashes.items():
                    remote_hash = remote_hashes.get(log_name)
                    if remote_hash and local_hash == remote_hash:
                        continue

                    buffer = self.log_buffers.get(log_name)
                    if buffer is None:
                        continue

                    content_bytes = buffer.getvalue().encode("utf-8")
                    if not content_bytes:
                        continue

                    try:
                        self._upload_memory_log(log_name, content_bytes)
                        self.state[f"log_hash_{log_name}"] = local_hash
                    except Exception as e:
                        self.logger.error(
                            f"Failed to upload in-memory log {log_name}: {e}"
                        )
            return

        logs_dir = self.board_dir / "logs"

        for log_name, local_hash in local_hashes.items():
            remote_hash = remote_hashes.get(log_name)

            if remote_hash and local_hash == remote_hash:
                # File exists on server and hash matches - skip
                self.logger.debug(f"Log file unchanged: {log_name}")
                continue

            # File changed or new or server doesn't have it - upload
            log_path = logs_dir / log_name

            if not log_path.exists():
                self.logger.warning(f"Log file disappeared: {log_name}")
                continue

            try:
                with open(log_path, "rb") as f:
                    content_bytes = f.read()

                # Upload via binary endpoint
                url = f"{self.remote_url}/api/projects/{self.project}/runs/{self.run_id}/logs/{log_name}"

                response = self.session.put(
                    url,
                    data=content_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=60,  # Longer timeout for large log files
                )

                response.raise_for_status()

                # Update local state
                self.state[f"log_hash_{log_name}"] = local_hash

                if remote_hash:
                    self.logger.info(
                        f"Updated log file: {log_name} ({len(content_bytes)} bytes)"
                    )
                else:
                    self.logger.info(
                        f"Uploaded new log file: {log_name} ({len(content_bytes)} bytes)"
                    )

            except Exception as e:
                self.logger.error(f"Failed to upload {log_name}: {e}")

    def _upload_memory_log(self, log_name: str, content_bytes: bytes) -> None:
        url = f"{self.remote_url}/api/projects/{self.project}/runs/{self.run_id}/logs/{log_name}"
        response = self.session.put(
            url,
            data=content_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=60,
        )
        response.raise_for_status()
        self.logger.info(
            f"Uploaded in-memory log file: {log_name} ({len(content_bytes)} bytes)"
        )

    def _sync_logs(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    def _sync_media(self, missing_hashes: list[str]):
        """Upload missing media files to remote server (from SQLite KV)

        Args:
            missing_hashes: List of media hashes to upload
        """
        if not missing_hashes:
            return

        if self.media_kv is None:
            if self.media_kv_path.exists() and not self._uses_shared_storage:
                self.media_kv = KVault(str(self.media_kv_path))
                self._owns_media_kv = True
            else:
                self.logger.warning("KVault not available, cannot upload media")
                return

        url = f"{self.remote_url}/api/projects/{self.project}/runs/{self.run_id}/media"

        files_data: list[tuple[str, bytes]] = []

        if self._uses_shared_storage:
            with self.storage_lock:
                media_info = self.storage.metadata_storage.fetch_media_info_by_hashes(
                    missing_hashes
                )

                for media_hash in missing_hashes:
                    if media_hash not in media_info:
                        self.logger.warning(
                            f"Media metadata not found for hash: {media_hash}"
                        )
                        continue

                    fmt = media_info[media_hash]["format"]
                    key = f"{media_hash}.{fmt}"
                    data = self.media_kv.get(key)
                    if data is None:
                        self.logger.warning(f"Media data not found in KVault: {key}")
                        continue

                    files_data.append((key, data))
        else:
            # Query SQLite metadata to get format for each hash
            conn = sqlite3.connect(str(self.sqlite_db))
            media_info = {}
            try:
                placeholders = ",".join("?" * len(missing_hashes))
                cursor = conn.execute(
                    f"SELECT DISTINCT media_hash, format FROM media WHERE media_hash IN ({placeholders})",
                    missing_hashes,
                )
                for row in cursor.fetchall():
                    media_hash, format = row
                    media_info[media_hash] = format
            finally:
                conn.close()

            for media_hash in missing_hashes:
                if media_hash not in media_info:
                    self.logger.warning(
                        f"Media metadata not found for hash: {media_hash}"
                    )
                    continue

                fmt = media_info[media_hash]
                key = f"{media_hash}.{fmt}"
                data = self.media_kv.get(key)
                if data is None:
                    self.logger.warning(f"Media data not found in KVault: {key}")
                    continue

                files_data.append((key, data))

        if not files_data:
            self.logger.warning("No media data to upload")
            return

        # Prepare multipart upload with in-memory files
        import io

        files = []
        try:
            for filename, data in files_data:
                # Create in-memory file object
                file_obj = io.BytesIO(data)
                file_obj.seek(0)  # Ensure we're at the start
                files.append(
                    ("files", (filename, file_obj, "application/octet-stream"))
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
            self.logger.info(f"Uploaded {len(files_data)} media files from KVault")

        except Exception as e:
            self.logger.error(f"Failed to upload media: {e}")
            raise

    def _get_latest_local_step(self) -> int | None:
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

    def _load_state(self) -> dict[str, Any]:
        """Load sync state from JSON file

        Returns:
            State dict with defaults
        """
        if self.memory_mode or self.state_file is None:
            if self._memory_state is None:
                self._memory_state = {
                    "last_synced_step": -1,
                    "metadata_synced": False,
                    "last_sync_at": None,
                    "remote_url": self.remote_url,
                    "project": self.project,
                    "run_id": self.run_id,
                }
            return self._memory_state

        if not self.state_file.exists():
            return {
                "last_synced_step": -1,
                "metadata_synced": False,
                "last_sync_at": None,
                "remote_url": self.remote_url,
                "project": self.project,
                "run_id": self.run_id,
                # Log file hashes stored as: log_hash_<filename>
            }

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                # Add new fields if missing (for backward compatibility)
                # Log file hashes are managed separately per file
                state.setdefault("metadata_synced", False)
                return state
        except Exception as e:
            self.logger.warning(f"Failed to load sync state: {e}, using defaults")
            return {
                "last_synced_step": -1,
                "metadata_synced": False,
                "last_sync_at": None,
                "remote_url": self.remote_url,
                "project": self.project,
                "run_id": self.run_id,
                # Log file hashes stored as: log_hash_<filename>
            }

    def _save_state(self):
        """Save sync state to JSON file"""
        if self.memory_mode or self.state_file is None:
            self._memory_state = self.state.copy()
            return

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
