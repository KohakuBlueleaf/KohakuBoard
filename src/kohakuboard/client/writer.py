"""Background writer process for non-blocking logging

v0.2.0+ ARCHITECTURE:
- Single writer process with multiprocessing queue
- SQLite + Lance storage backends only
- Immediate media inserts (for ID return)
- Batched metric/table writes
"""

import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any

from kohakuboard.client.types.media_handler import MediaHandler
from kohakuboard.client.storage.hybrid import HybridStorage
from kohakuboard.client.storage.sqlite import SQLiteMetadataStorage


class LogWriter:
    """Background process that handles all disk I/O operations"""

    def __init__(
        self, board_dir: Path, queue: Any, stop_event: Any, backend: str = "duckdb"
    ):
        """Initialize log writer

        Args:
            board_dir: Board directory path
            queue: Queue for receiving log messages (mp.Queue)
            stop_event: Event to signal shutdown (mp.Event)
            backend: Storage backend ("duckdb" or "parquet")
        """
        self.board_dir = board_dir
        self.queue = queue
        self.stop_event = stop_event
        self.backend = backend
        self.logger = None  # Will be set by writer_process_main

        # Initialize storage backend based on selection (v0.2.0+: sqlite or hybrid only)
        if backend == "hybrid":
            self.storage = HybridStorage(board_dir / "data")
        elif backend == "sqlite":
            from kohakuboard.client.storage.sqlite import SQLiteMetadataStorage

            self.storage = SQLiteMetadataStorage(board_dir / "data")
        else:
            raise ValueError(
                f"Unsupported backend: '{backend}'. "
                f"Only 'sqlite' and 'hybrid' are supported in v0.2.0+."
            )

        self.media_handler = MediaHandler(board_dir / "media")

        # Statistics
        self.messages_processed = 0
        self.last_flush_time = time.time()
        # Best-effort flush: flush frequently for online sync
        self.auto_flush_interval = 5  # Auto-flush every 5 seconds (aggressive)

    def run(self):
        """Main loop - adaptive batching with exponential backoff"""
        self.logger.info(f"LogWriter started for {self.board_dir}")

        # Adaptive sleep parameters
        min_period = 0.01  # 10ms minimum sleep
        max_period = 1.0  # 1s maximum sleep
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

                        # Adaptive sleep: min_period * 2^k, capped at max_period
                        sleep_time = min(
                            min_period * (2**consecutive_empty), max_period
                        )

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
        elif msg_type == "batch":
            self._handle_batch(message)
        elif msg_type == "flush":
            self._handle_flush()
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")

    def _handle_scalar(self, message: dict):
        """Handle scalar metric logging"""
        step = message["step"]
        global_step = message.get("global_step")
        metrics = message["metrics"]
        timestamp = message.get("timestamp")

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

                        # **FIX (v0.2.0): Store metadata in database**
                        # This was missing in v0.1 - table media was never registered in DB
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
                        # Replace with error marker
                        table_data["rows"][int(row_idx)][int(col_idx)] = "<media error>"

        # Store table with media ID references
        try:
            self.storage.append_table(step, global_step, name, table_data)
            self.logger.debug(f"Logged table '{name}' at step {step}")
        except Exception as e:
            self.logger.error(f"Error storing table: {e}")
            raise

    def _handle_histogram(self, message: dict):
        """Handle histogram logging"""
        step = message["step"]
        global_step = message.get("global_step")
        name = message["name"]
        precomputed = message.get("precomputed", False)

        if precomputed:
            # Precomputed bins/counts - store directly
            bins = message["bins"]
            counts = message["counts"]
            precision = message.get("precision", "exact")

            # For precomputed, we need to store bins/counts directly
            # The storage layer needs to handle this appropriately
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
            # Raw values - compute histogram in storage layer
            values = message["values"]
            num_bins = message.get("num_bins", 64)
            precision = message.get("precision", "compact")

            self.storage.append_histogram(
                step, global_step, name, values, num_bins, precision
            )

    def _handle_batch(self, message: dict):
        """Handle batched message containing multiple types

        This processes scalars, media, tables, and histograms from a single message.
        All items share the same step and global_step.
        """
        step = message["step"]
        global_step = message.get("global_step")
        timestamp = message.get("timestamp")

        # Process scalars if present
        if "scalars" in message:
            scalars = message["scalars"]
            self.storage.append_metrics(step, global_step, scalars, timestamp)

        # Process media if present
        if "media" in message:
            media_dict = message["media"]
            for name, media_data in media_dict.items():
                media_type = media_data.get("media_type", "image")
                media_obj_data = media_data["media_data"]
                caption = media_data.get("caption")

                # Process media
                media_meta = self.media_handler.process_media(
                    media_obj_data, name, step, media_type
                )
                media_list = [media_meta]

                # Store metadata
                self.storage.append_media(step, global_step, name, media_list, caption)

        # Process tables if present
        if "tables" in message:
            tables_dict = message["tables"]
            for name, table_data in tables_dict.items():
                # Process any media objects in the table
                media_objects = table_data.pop("media_objects", {})
                if media_objects:
                    # Process each media object and save to disk
                    for row_idx, col_dict in media_objects.items():
                        for col_idx, media_obj in col_dict.items():
                            # Save media to disk
                            media_meta = self.media_handler.process_media(
                                media_obj.data,
                                f"{name}_r{row_idx}_c{col_idx}",
                                step,
                                media_type=media_obj.media_type,
                            )
                            # Replace placeholder with media reference
                            media_id = media_meta["media_id"]
                            table_data["rows"][int(row_idx)][
                                int(col_idx)
                            ] = f"<media id={media_id}>"

                self.storage.append_table(step, global_step, name, table_data)

        # Process histograms if present
        if "histograms" in message:
            histograms_dict = message["histograms"]
            for name, hist_data in histograms_dict.items():
                computed = hist_data.get("computed", False)

                if computed:
                    # Precomputed bins/counts
                    bins = hist_data["bins"]
                    counts = hist_data["counts"]
                    precision = hist_data.get("precision", "exact")

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
                    # Raw values - compute in storage layer
                    values = hist_data["values"]
                    num_bins = hist_data.get("num_bins", 64)
                    precision = hist_data.get("precision", "compact")

                    self.storage.append_histogram(
                        step, global_step, name, values, num_bins, precision
                    )

    def _handle_flush(self):
        """Handle explicit flush request"""
        self.storage.flush_all()
        self.last_flush_time = time.time()
        self.logger.debug("Explicit flush completed")

    def _auto_flush(self):
        """Periodic auto-flush"""
        self.storage.flush_all()
        self.last_flush_time = time.time()
        self.logger.debug(
            f"Auto-flush completed ({self.messages_processed} messages processed)"
        )

    def _final_flush(self):
        """Final flush on shutdown - drain ALL remaining messages"""
        try:
            # Process ALL remaining messages in queue (no arbitrary limit!)
            remaining = 0
            last_log_count = 0

            while not self.queue.empty():
                try:
                    message = self.queue.get_nowait()
                    self._process_message(message)
                    remaining += 1

                    # Log progress every 1000 messages
                    if remaining - last_log_count >= 1000:
                        self.logger.info(
                            f"Final drain progress: {remaining} messages processed..."
                        )
                        last_log_count = remaining

                except Empty:
                    break
                except KeyboardInterrupt:
                    self.logger.warning(
                        f"Final drain interrupted after {remaining} messages!"
                    )
                    self.logger.warning(
                        "Press Ctrl+C again in main process for force exit"
                    )
                    # Don't break - let main process handle force exit
                except Exception as e:
                    self.logger.error(
                        f"Error during final drain at message {remaining}: {e}"
                    )
                    # Continue processing other messages

            # Flush all buffers
            self.logger.info(f"Flushing all buffers ({remaining} messages drained)...")
            self.storage.flush_all()

            # Close storage backend if it has a close method
            if hasattr(self.storage, "close"):
                self.storage.close()

            self.logger.info(
                f"LogWriter stopped. Processed {self.messages_processed} total messages "
                f"({remaining} from final queue drain)"
            )

        except Exception as e:
            self.logger.error(f"Error during final flush: {e}")


def writer_process_main(
    board_dir: Path, queue: Any, stop_event: Any, backend: str = "duckdb"
):
    """Entry point for writer process

    Args:
        board_dir: Board directory
        queue: Message queue (mp.Queue)
        stop_event: Stop event (mp.Event)
        backend: Storage backend ("duckdb" or "parquet")
    """
    # Configure logger for this process (file only, no stdout)
    from kohakuboard.logger import get_logger

    log_file = board_dir / "logs" / "writer.log"
    logger = get_logger("WRITER", file_only=True, log_file=log_file)

    # Create and run writer
    writer = LogWriter(board_dir, queue, stop_event, backend)
    writer.logger = logger
    writer.run()
