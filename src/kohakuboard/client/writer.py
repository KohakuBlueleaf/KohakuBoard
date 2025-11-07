"""Background writer process for non-blocking logging

v0.2.0+ ARCHITECTURE:
- Single writer process with multiprocessing queue
- Three-tier SQLite storage architecture:
  1. KohakuVault ColumnVault - Metrics/histograms (blob-based columnar)
  2. KohakuVault KVault - Media blobs (K-V table with B+Tree index)
  3. Standard SQLite - Metadata (traditional relational tables)
- Immediate media inserts (for ID return)
- Batched metric/table writes
"""

import io
import threading
import time
from copy import deepcopy
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty
from typing import Any, Optional

import numpy as np
from kohakuvault import KVault

from kohakuboard.client.sync_worker import SyncWorker
from kohakuboard.client.types.kernel_density import KernelDensity
from kohakuboard.client.types.media_handler import MediaHandler
from kohakuboard.storage.hybrid import HybridStorage
from kohakuboard.storage.memory import MemoryHybridStorage
from kohakuboard.logger import get_logger


class LogWriter:
    """Background process that handles all disk I/O operations"""

    def __init__(
        self,
        board_dir: Path,
        queue: Any,
        stop_event: Any,
        sync_config: Optional[dict[str, Any]] = None,
        memory_mode: bool = False,
        release_queue: Any | None = None,
    ):
        """Initialize log writer

        Args:
            board_dir: Board directory path
            queue: Queue for receiving log messages (mp.Queue)
            stop_event: Event to signal shutdown (mp.Event)
            sync_config: Optional configuration for remote sync worker
            memory_mode: Enable in-memory storage mode
        """
        self.board_dir = board_dir
        self.queue = queue
        self.stop_event = stop_event
        self.logger = None  # Will be set by writer_process_main
        self.storage_lock = threading.RLock()
        self.memory_mode = memory_mode
        self._pending_memory_warning = memory_mode and not sync_config
        self._sync_worker_init_error: Optional[Exception] = None
        self.release_queue = release_queue

        # Optional metadata payload (used in memory mode when metadata.json unavailable)
        self.initial_metadata = (
            deepcopy(sync_config.get("metadata"))
            if sync_config and "metadata" in sync_config
            else None
        )

        # Initialize storage backend (disk or in-memory)
        if memory_mode:
            storage_logger = get_logger("MEMORY_STORAGE", drop=True)
            self.storage = MemoryHybridStorage(logger=storage_logger)
            self.media_kv = KVault(":memory:")
            media_logger = get_logger("MEMORY_MEDIA", drop=True)
            self.log_buffers = {"output.log": io.StringIO()}
        else:
            self.storage = HybridStorage(board_dir / "data")
            media_kv_path = board_dir / "media" / "blobs.db"
            self.media_kv = KVault(str(media_kv_path))
            media_logger = None
            self.log_buffers = None

        # Initialize media handler with storage-backed KV
        self.media_handler = MediaHandler(
            board_dir / "media", self.media_kv, logger=media_logger
        )

        # Optional sync worker (started in run() once logger is ready)
        self.sync_worker: Optional[SyncWorker] = None
        if sync_config:
            try:
                remote_url = sync_config["remote_url"]
                remote_token = sync_config["remote_token"]
                project = sync_config.get("project", "local")
                run_id = sync_config["run_id"]
                sync_interval = sync_config.get("sync_interval", 10)

                self.sync_worker = SyncWorker(
                    board_dir=board_dir,
                    remote_url=remote_url,
                    remote_token=remote_token,
                    project=project,
                    run_id=run_id,
                    sync_interval=sync_interval,
                    storage=self.storage,
                    storage_lock=self.storage_lock,
                    media_kv=self.media_kv,
                    memory_mode=memory_mode,
                    log_buffers=self.log_buffers,
                    metadata=self.initial_metadata,
                )
            except Exception as exc:
                self._sync_worker_init_error = exc
                self.sync_worker = None

        # Statistics
        self.messages_processed = 0
        self.last_flush_time = time.time()
        # Best-effort flush: flush frequently for online sync
        self.auto_flush_interval = 0.25 if memory_mode else 5.0

    def _notify_shared_memory_release(self, *names: str):
        """Tell the parent process it can release the creation handles."""
        if not self.release_queue:
            return

        for name in names:
            if not name:
                continue
            try:
                self.release_queue.put({"name": name})
            except Exception as exc:
                if self.logger:
                    self.logger.debug(
                        f"Failed to enqueue SharedMemory release for '{name}': {exc}"
                    )

    def run(self):
        """Main loop - adaptive batching with exponential backoff"""
        self.logger.info(f"LogWriter started for {self.board_dir}")

        if self._pending_memory_warning:
            self.logger.warning(
                "Memory mode enabled without remote sync worker. "
                "Logs will remain in-memory and will be lost after shutdown."
            )
            self._pending_memory_warning = False

        if self._sync_worker_init_error is not None:
            self.logger.error(
                f"Failed to initialize sync worker: {self._sync_worker_init_error}"
            )
            self._sync_worker_init_error = None

        if self.sync_worker:
            try:
                self.sync_worker.start()
                self.logger.info(
                    "Sync worker thread started within writer process "
                    f"(interval: {self.sync_worker.sync_interval}s)"
                )
            except Exception as exc:
                self.logger.error(f"Could not start sync worker: {exc}")
                self.sync_worker = None

        # Adaptive sleep parameters
        min_period = 0.5  # 0.5s minimum sleep
        max_period = 5.0  # 5s maximum sleep
        consecutive_empty = 0  # Track consecutive empty queue reads

        try:
            while not self.stop_event.is_set():
                try:
                    # Process ALL available messages in queue
                    batch_count = 0
                    batch_start = time.time()

                    # Drain queue completely (up to 10k to allow stop_event check)
                    while batch_count < 10000 and not self.stop_event.is_set():
                        try:
                            message = self.queue.get_nowait()
                            self._process_message(message)
                            self.messages_processed += 1
                            batch_count += 1
                        except Empty:
                            break

                    # Flush immediately after processing ANY messages
                    if batch_count > 0:
                        with self.storage_lock:
                            self.storage.flush_all()
                        batch_time = time.time() - batch_start
                        self.logger.debug(
                            f"Processed and flushed {batch_count} messages in {batch_time*1000:.1f}ms"
                        )
                        self.last_flush_time = time.time()

                        # Reset backoff counter - we had work to do
                        consecutive_empty = 0
                    else:
                        # Queue empty - increase backoff
                        consecutive_empty += 1

                        if (
                            self.auto_flush_interval
                            and time.time() - self.last_flush_time
                            >= self.auto_flush_interval
                        ):
                            self._auto_flush()
                            consecutive_empty = 0

                    # Adaptive sleep: min_period * 2^k, capped at max_period
                    sleep_time = min(min_period * (2**consecutive_empty), max_period)

                    # Sleep in small chunks to allow responsive shutdown
                    # Instead of sleep(1.0), sleep(0.1) × 10 times and check stop_event
                    slept = 0.0
                    while slept < sleep_time and not self.stop_event.is_set():
                        time.sleep(0.05)  # Sleep in 50ms chunks
                        slept += 0.05

                except KeyboardInterrupt:
                    # Received interrupt in worker - DRAIN QUEUE FIRST
                    self.logger.warning(
                        "Writer received interrupt, draining queue before stopping..."
                    )
                    # Continue processing until stop_event is set by main process

                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")

            # Stop event is set - drain remaining queue
            self.logger.info("Stop event detected, draining remaining queue...")
            self._final_flush()

        except KeyboardInterrupt:
            self.logger.warning("Writer interrupted during shutdown, forcing exit...")
        except Exception as e:
            self.logger.error(f"Fatal error in LogWriter: {e}")
            raise

    def _process_message(self, message: dict):
        """Process a single log message

        Args:
            message: Log message dict with 'type' and data fields
        """
        msg_type = message.get("type")

        if msg_type == "scalar":
            self._handle_scalar(message)
        elif msg_type == "media":
            self._handle_media(message)
        elif msg_type == "table":
            self._handle_table(message)
        elif msg_type == "histogram":
            self._handle_histogram(message)
        elif msg_type == "tensor":
            self._handle_tensor(message)
        elif msg_type == "kernel_density":
            self._handle_kernel_density(message)
        elif msg_type == "batch":
            self._handle_batch(message)
        elif msg_type == "log":
            self._handle_log(message)
        elif msg_type == "flush":
            self._handle_flush()
        elif msg_type == "stop":
            # Poison pill - stop processing and exit
            self.logger.info("Received stop message, exiting immediately")
            self.stop_event.set()
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")

    def _handle_scalar(self, message: dict):
        """Handle scalar metric logging"""
        step = message["step"]
        global_step = message.get("global_step")
        metrics = message["metrics"]
        timestamp = message.get("timestamp")

        with self.storage_lock:
            self.storage.append_metrics(step, global_step, metrics, timestamp)

    def _handle_media(self, message: dict):
        """Handle direct media logging (images/video/audio)"""
        step = message["step"]
        global_step = message.get("global_step")
        name = message["name"]
        caption = message.get("caption")
        media_type = message.get("media_type", "image")
        media_data = message["media_data"]

        try:
            with self.storage_lock:
                # Process single media (saves to disk)
                media_meta = self.media_handler.process_media(
                    media_data, name, step, media_type
                )

                # Store metadata in database (returns media_id list)
                media_ids = self.storage.append_media(
                    step, global_step, name, [media_meta], caption
                )

            self.logger.debug(
                f"Logged media '{name}' at step {step} (ID: {media_ids[0]})"
            )

        except Exception as e:
            self.logger.error(f"Error handling media '{name}': {e}")
            raise

    def _handle_table(self, message: dict):
        """Handle table logging with embedded media"""
        step = message["step"]
        global_step = message.get("global_step")
        name = message["name"]
        table_data = message["table_data"]

        # Process any media objects in the table
        try:
            with self.storage_lock:
                media_objects = table_data.pop("media_objects", {})
                if media_objects:
                    for row_idx, col_dict in media_objects.items():
                        for col_idx, media_obj in col_dict.items():
                            try:
                                # Process media (saves to disk)
                                media_meta = self.media_handler.process_media(
                                    media_obj.data,
                                    f"{name}_r{row_idx}_c{col_idx}",
                                    step,
                                    media_type=media_obj.media_type,
                                )

                                media_ids = self.storage.append_media(
                                    step,
                                    global_step,
                                    f"{name}_r{row_idx}_c{col_idx}",
                                    [media_meta],
                                    caption=media_obj.caption,
                                )

                                # Replace placeholder with database ID tag
                                media_id = media_ids[0]
                                table_data["rows"][int(row_idx)][
                                    int(col_idx)
                                ] = f"<media id={media_id}>"

                                self.logger.debug(
                                    f"Logged table media at ({row_idx},{col_idx}), ID: {media_id}"
                                )

                            except Exception as e:
                                self.logger.error(
                                    f"Error processing table media at ({row_idx},{col_idx}): {e}"
                                )
                                table_data["rows"][int(row_idx)][
                                    int(col_idx)
                                ] = "<media error>"

                # Store table with media ID references
                self.storage.append_table(step, global_step, name, table_data)
                self.logger.debug(f"Logged table '{name}' at step {step}")
        except Exception as e:
            self.logger.error(f"Error storing table: {e}")
            raise

    def _handle_histogram(self, message: dict):
        """Handle histogram logging with SharedMemory support"""
        step = message["step"]
        global_step = message.get("global_step")
        name = message["name"]
        precomputed = message.get("precomputed", False)

        # Check if data is in SharedMemory
        if "shared_memory" in message:
            shm_info = message["shared_memory"]

            if precomputed:
                # Precomputed bins/counts from SharedMemory
                bins_shm = None
                counts_shm = None
                release_names: list[str] = []
                try:
                    # Attach to bins SharedMemory
                    bins_shm = shared_memory.SharedMemory(name=shm_info["bins_name"])
                    bins_dtype = np.dtype(shm_info["bins_dtype"])
                    bins_array = np.ndarray(
                        shm_info["bins_shape"], dtype=bins_dtype, buffer=bins_shm.buf
                    )
                    bins = bins_array.copy().tolist()  # Copy to local memory

                    # Attach to counts SharedMemory
                    counts_shm = shared_memory.SharedMemory(
                        name=shm_info["counts_name"]
                    )
                    counts_dtype = np.dtype(shm_info["counts_dtype"])
                    counts_array = np.ndarray(
                        shm_info["counts_shape"],
                        dtype=counts_dtype,
                        buffer=counts_shm.buf,
                    )
                    counts = counts_array.copy().tolist()  # Copy to local memory
                    release_names = [
                        shm_info.get("bins_name"),
                        shm_info.get("counts_name"),
                    ]

                    precision = message.get("precision", "exact")

                    # Store histogram
                    with self.storage_lock:
                        self.storage.append_histogram(
                            step,
                            global_step,
                            name,
                            None,
                            bins=bins,
                            counts=counts,
                            precision=precision,
                        )

                finally:
                    self._notify_shared_memory_release(*release_names)
                    # Clean up SharedMemory
                    if bins_shm:
                        try:
                            bins_shm.close()
                            bins_shm.unlink()
                        except Exception as e:
                            self.logger.warning(
                                f"Error cleaning up bins SharedMemory: {e}"
                            )

                    if counts_shm:
                        try:
                            counts_shm.close()
                            counts_shm.unlink()
                        except Exception as e:
                            self.logger.warning(
                                f"Error cleaning up counts SharedMemory: {e}"
                            )

            else:
                # Raw values from SharedMemory
                values_shm = None
                release_name = shm_info.get("values_name")
                try:
                    # Attach to values SharedMemory
                    values_shm = shared_memory.SharedMemory(
                        name=shm_info["values_name"]
                    )
                    values_dtype = np.dtype(shm_info["values_dtype"])
                    values_array = np.ndarray(
                        shm_info["values_shape"],
                        dtype=values_dtype,
                        buffer=values_shm.buf,
                    )
                    values = values_array.copy().tolist()  # Copy to local memory

                    num_bins = message.get("num_bins", 64)
                    precision = message.get("precision", "compact")

                    # Store histogram
                    with self.storage_lock:
                        self.storage.append_histogram(
                            step, global_step, name, values, num_bins, precision
                        )

                finally:
                    self._notify_shared_memory_release(release_name)
                    # Clean up SharedMemory
                    if values_shm:
                        try:
                            values_shm.close()
                            values_shm.unlink()
                        except Exception as e:
                            self.logger.warning(
                                f"Error cleaning up values SharedMemory: {e}"
                            )

        else:
            # Legacy path (no SharedMemory) - for backward compatibility
            if precomputed:
                bins = message["bins"]
                counts = message["counts"]
                precision = message.get("precision", "exact")

                with self.storage_lock:
                    self.storage.append_histogram(
                        step,
                        global_step,
                        name,
                        None,
                        bins=bins,
                        counts=counts,
                        precision=precision,
                    )
            else:
                values = message["values"]
                num_bins = message.get("num_bins", 64)
                precision = message.get("precision", "compact")

                with self.storage_lock:
                    self.storage.append_histogram(
                        step, global_step, name, values, num_bins, precision
                    )

    def _handle_tensor(self, message: dict):
        """Handle tensor logging."""
        step = message["step"]
        global_step = message.get("global_step")
        name = message["name"]
        tensor_meta = message.get("tensor_meta", {})

        payload_bytes = None
        shm = None
        release_name = None

        try:
            if "shared_memory" in message:
                shm_info = message["shared_memory"]
                release_name = shm_info.get("values_name")
                shm = shared_memory.SharedMemory(name=shm_info["values_name"])
                dtype = np.dtype(shm_info["values_dtype"])
                array = np.ndarray(
                    shm_info["values_shape"], dtype=dtype, buffer=shm.buf
                )
                local_array = array.copy()
            else:
                values = message.get("values")
                if values is None:
                    self.logger.warning("Tensor message missing values payload")
                    return
                local_array = np.array(values)

            buffer = io.BytesIO()
            np.save(buffer, local_array, allow_pickle=False)
            payload_bytes = buffer.getvalue()

            with self.storage_lock:
                self.storage.append_tensor(
                    step,
                    global_step,
                    name,
                    payload_bytes,
                    tensor_meta,
                )
        finally:
            self._notify_shared_memory_release(release_name)
            if shm:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as exc:
                    self.logger.warning(f"Error cleaning tensor SharedMemory: {exc}")

    def _handle_kernel_density(self, message: dict):
        """Handle kernel density logging."""
        step = message["step"]
        global_step = message.get("global_step")
        name = message["name"]
        kde_mode = message.get("kde_mode", "precomputed")
        kde_meta = message.get("kde_meta", {}) if kde_mode != "raw" else {}

        grid_shm = None
        density_shm = None
        raw_shm = None

        release_names: list[str] = []

        try:
            if kde_mode == "raw":
                shm_info = message.get("shared_memory") or {}
                raw_name = shm_info.get("raw_name")
                if not raw_name:
                    self.logger.warning("Raw KDE message missing shared memory info")
                    return

                raw_shm = shared_memory.SharedMemory(name=raw_name)
                raw_dtype = np.dtype(shm_info["raw_dtype"])
                raw_array = np.ndarray(
                    shm_info["raw_shape"], dtype=raw_dtype, buffer=raw_shm.buf
                ).copy()
                release_names = [raw_name]

                kde_config = message.get("kde_config") or {}
                kde_obj = KernelDensity(raw_values=raw_array, **kde_config)
                grid_array, density_array = kde_obj.to_arrays()
                kde_meta = kde_obj.summary()
            elif "shared_memory" in message:
                shm_info = message["shared_memory"]

                grid_shm = shared_memory.SharedMemory(name=shm_info["grid_name"])
                grid_dtype = np.dtype(shm_info["grid_dtype"])
                grid_array = np.ndarray(
                    shm_info["grid_shape"], dtype=grid_dtype, buffer=grid_shm.buf
                ).copy()

                density_shm = shared_memory.SharedMemory(name=shm_info["density_name"])
                density_dtype = np.dtype(shm_info["density_dtype"])
                density_array = np.ndarray(
                    shm_info["density_shape"],
                    dtype=density_dtype,
                    buffer=density_shm.buf,
                ).copy()
                release_names = [
                    shm_info.get("grid_name"),
                    shm_info.get("density_name"),
                ]
            else:
                grid_array = np.array(message.get("grid"), dtype=np.float32)
                density_array = np.array(message.get("density"), dtype=np.float32)

            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                grid=grid_array.astype(np.float32, copy=False),
                density=density_array.astype(np.float32, copy=False),
            )
            payload_bytes = buffer.getvalue()

            with self.storage_lock:
                self.storage.append_kernel_density(
                    step,
                    global_step,
                    name,
                    payload_bytes,
                    kde_meta,
                )
        finally:
            self._notify_shared_memory_release(*release_names)
            if raw_shm:
                try:
                    raw_shm.close()
                    raw_shm.unlink()
                except Exception as exc:
                    self.logger.warning(f"Error cleaning KDE raw SharedMemory: {exc}")
            if grid_shm:
                try:
                    grid_shm.close()
                    grid_shm.unlink()
                except Exception as exc:
                    self.logger.warning(f"Error cleaning KDE grid SharedMemory: {exc}")

            if density_shm:
                try:
                    density_shm.close()
                    density_shm.unlink()
                except Exception as exc:
                    self.logger.warning(
                        f"Error cleaning KDE density SharedMemory: {exc}"
                    )

    def _handle_log(self, message: dict):
        """Handle stdout/stderr chunks in memory mode."""
        if not self.log_buffers:
            return

        stream = message.get("stream", "stdout")
        data = message.get("data", "")
        if not data:
            return

        with self.storage_lock:
            buffer = self.log_buffers.setdefault("output.log", io.StringIO())
            if stream == "stderr":
                buffer.write("[STDERR] ")
            buffer.write(data)

    def _handle_batch(self, message: dict):
        """Handle batched message containing multiple types

        This processes scalars, media, tables, and histograms from a single message.
        All items share the same step and global_step.
        """
        step = message["step"]
        global_step = message.get("global_step")
        timestamp = message.get("timestamp")

        # Scalars
        if "scalars" in message:
            self._handle_scalar(
                {
                    "type": "scalar",
                    "step": step,
                    "global_step": global_step,
                    "metrics": message["scalars"],
                    "timestamp": timestamp,
                }
            )

        # Media
        if "media" in message:
            for name, media_data in message["media"].items():
                media_msg = {
                    "type": "media",
                    "step": step,
                    "global_step": global_step,
                    "name": name,
                    "media_type": media_data.get("media_type", "image"),
                    "media_data": media_data["media_data"],
                    "caption": media_data.get("caption"),
                }
                self._handle_media(media_msg)

        # Tables
        if "tables" in message:
            for name, table_data in message["tables"].items():
                table_msg = {
                    "type": "table",
                    "step": step,
                    "global_step": global_step,
                    "name": name,
                    "table_data": table_data,
                }
                self._handle_table(table_msg)

        # Histograms
        if "histograms" in message:
            for name, hist_data in message["histograms"].items():
                hist_msg = {
                    "type": "histogram",
                    "step": step,
                    "global_step": global_step,
                    "name": name,
                }
                hist_msg.update(hist_data)

                # Ensure compatibility with _handle_histogram expectations
                hist_msg["precomputed"] = hist_data.get("computed", False)
                self._handle_histogram(hist_msg)

    def _handle_flush(self):
        """Handle explicit flush request"""
        with self.storage_lock:
            self.storage.flush_all()
        self.last_flush_time = time.time()
        self.logger.debug("Explicit flush completed")

    def _auto_flush(self):
        """Periodic auto-flush"""
        with self.storage_lock:
            self.storage.flush_all()
        self.last_flush_time = time.time()
        self.logger.debug(
            f"Auto-flush completed ({self.messages_processed} messages processed)"
        )

    def _stop_sync_worker(self):
        """Stop sync worker thread gracefully"""
        if not self.sync_worker:
            return

        try:
            self.logger.info("Stopping sync worker...")
            self.sync_worker.stop(timeout=30)
        except Exception as exc:
            self.logger.error(f"Error stopping sync worker: {exc}")
        finally:
            self.sync_worker = None

    def _final_flush(self):
        """Final flush on shutdown - drain ALL remaining messages"""
        try:
            remaining = 0
            batch_size = 0
            batch_start = time.time()
            last_log_time = 0.0

            while True:
                try:
                    message = self.queue.get(timeout=0.1)
                except Empty:
                    break
                except ValueError:
                    break
                except KeyboardInterrupt:
                    self.logger.warning(
                        f"Final drain interrupted after {remaining} messages!"
                    )
                    self.logger.warning(
                        "Press Ctrl+C again in main process for force exit"
                    )
                    continue
                except Exception as exc:
                    self.logger.error(
                        f"Error retrieving message during final drain: {exc}"
                    )
                    break

                try:
                    self._process_message(message)
                    remaining += 1
                    batch_size += 1
                except Exception as exc:
                    self.logger.error(
                        f"Error during final drain at message {remaining}: {exc}"
                    )
                    continue

                if batch_size >= 5000:
                    with self.storage_lock:
                        self.storage.flush_all()
                    self.logger.debug(
                        f"Final drain flushed {batch_size} messages in "
                        f"{(time.time() - batch_start)*1000:.1f}ms"
                    )
                    batch_size = 0
                    batch_start = time.time()
                elif time.time() - last_log_time >= 1.0:
                    self.logger.debug(
                        f"Final drain progress: {remaining} messages processed..."
                    )
                    last_log_time = time.time()

            # Flush all buffers
            self.logger.info(f"Flushing all buffers ({remaining} messages drained)...")
            with self.storage_lock:
                self.storage.flush_all()

            # Ensure sync worker completes final upload before closing storage
            self._stop_sync_worker()

            # Close SQLite KV storage and hybrid storage while holding lock
            with self.storage_lock:
                if hasattr(self, "media_kv") and self.media_kv is not None:
                    try:
                        self.media_kv.checkpoint()
                    except Exception as e:
                        self.logger.warning(f"KVault checkpoint failed: {e}")
                    self.media_kv.close()
                    self.media_kv = None

                if hasattr(self.storage, "close"):
                    self.storage.close()

            try:
                self.queue.close()
            except Exception as exc:
                self.logger.debug(f"Error closing queue in writer: {exc}")
            try:
                self.queue.cancel_join_thread()
            except Exception:
                pass
            finally:
                self.queue = None

            self.logger.info(
                f"LogWriter stopped. Processed {self.messages_processed} total messages "
                f"({remaining} from final queue drain)"
            )

        except Exception as e:
            self.logger.error(f"Error during final flush: {e}")


def writer_process_main(
    board_dir: Path,
    queue: Any,
    stop_event: Any,
    sync_config: Optional[dict[str, Any]] = None,
    memory_mode: bool = False,
    release_queue: Any | None = None,
):
    """Entry point for writer process

    Args:
        board_dir: Board directory
        queue: Message queue (mp.Queue)
        stop_event: Stop event (mp.Event)
        sync_config: Optional sync configuration dict
        memory_mode: Enable in-memory storage mode
        release_queue: Optional queue for SharedMemory release notifications
    """
    # Configure logger for this process (file only, no stdout)
    from kohakuboard.logger import get_logger

    if memory_mode:
        logger = get_logger("WRITER", drop=True)
    else:
        log_file = board_dir / "logs" / "writer.log"
        logger = get_logger("WRITER", file_only=True, log_file=log_file)

    # Create and run writer
    writer = LogWriter(
        board_dir,
        queue,
        stop_event,
        sync_config=sync_config,
        memory_mode=memory_mode,
        release_queue=release_queue,
    )
    writer.logger = logger
    writer.run()
