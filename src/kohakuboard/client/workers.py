"""Individual worker processes for each storage type

Architecture:
- One process per storage type (metrics, media, tables, histograms)
- Each reads from its own queue
- Batching: count-based (threshold) OR time-based (interval)
- No GIL contention between storage types
"""

import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty
from typing import Any

from kohakuboard.logger import get_logger

from kohakuboard.storage.histogram import HistogramStorage
from kohakuboard.storage.lance import LanceMetricsStorage
from kohakuboard.storage.sqlite import SQLiteMetadataStorage

logger = get_logger("WORKER")


def metrics_worker_main(board_dir: Path, queue: Any, stop_event: Any):
    """Worker process for scalar metrics (Lance storage)

    Args:
        board_dir: Board directory
        queue: Message queue (mp.Queue)
        stop_event: Stop event (mp.Event)
    """
    # Configure logger
    logger.remove()
    logger.add(
        board_dir / "logs" / "metrics_worker.log", rotation="10 MB", level="DEBUG"
    )

    storage = LanceMetricsStorage(board_dir / "data")

    # Batching config
    batch_threshold = 1000  # Flush after 1000 messages
    batch_interval = 2.0  # OR flush after 2 seconds

    logger.info("Metrics worker started")

    try:
        while not stop_event.is_set():
            # STEP 1: Drain ALL messages from queue
            messages = []
            while True:
                try:
                    messages.append(queue.get_nowait())
                except Empty:
                    break

            # STEP 2: Process all messages
            if messages:
                batch_start = time.time()
                for message in messages:
                    storage.append_metrics(
                        message["step"],
                        message.get("global_step"),
                        message["metrics"],
                        message.get("timestamp"),
                    )

                # STEP 3: Flush once
                storage.flush()
                batch_time = time.time() - batch_start
                logger.debug(
                    f"Processed and flushed {len(messages)} metrics in {batch_time*1000:.1f}ms"
                )
            else:
                # No messages - sleep
                time.sleep(0.01)

        # Final flush
        logger.info("Metrics worker shutting down, draining queue...")
        final_count = 0
        while not queue.empty():
            try:
                message = queue.get_nowait()
                storage.append_metrics(
                    message["step"],
                    message.get("global_step"),
                    message["metrics"],
                    message.get("timestamp"),
                )
                final_count += 1
            except Empty:
                break

        storage.flush()
        storage.close()
        logger.info(f"Metrics worker stopped (drained {final_count}, flushed all)")

    except Exception as e:
        logger.error(f"Metrics worker error: {e}")
        raise


def metadata_worker_main(
    board_dir: Path, media_queue: Any, tables_queue: Any, stop_event: Any
):
    """Worker process for media and tables (SQLite storage)

    Args:
        board_dir: Board directory
        media_queue: Media queue (mp.Queue)
        tables_queue: Tables queue (mp.Queue)
        stop_event: Stop event (mp.Event)
    """
    logger.remove()
    logger.add(
        board_dir / "logs" / "metadata_worker.log", rotation="10 MB", level="DEBUG"
    )

    from kohakuboard.client.types.media_handler import MediaHandler

    storage = SQLiteMetadataStorage(board_dir / "data")
    media_handler = MediaHandler(board_dir / "media")

    batch_threshold = 100
    batch_interval = 2.0

    logger.info("Metadata worker started")

    try:
        while not stop_event.is_set():
            # STEP 1: Drain ALL messages from both queues
            media_messages = []
            table_messages = []

            while True:
                try:
                    media_messages.append(media_queue.get_nowait())
                except Empty:
                    break

            while True:
                try:
                    table_messages.append(tables_queue.get_nowait())
                except Empty:
                    break

            # STEP 2: Process all messages
            total = 0
            if media_messages or table_messages:
                batch_start = time.time()

                for message in media_messages:
                    if "images" in message:
                        images = message["images"]
                        media_list = media_handler.process_images(
                            images, message["name"], message["step"]
                        )
                    else:
                        media_type = message.get("media_type", "image")
                        media_data = message["media_data"]
                        media_meta = media_handler.process_media(
                            media_data, message["name"], message["step"], media_type
                        )
                        media_list = [media_meta]

                    storage.append_media(
                        message["step"],
                        message.get("global_step"),
                        message["name"],
                        media_list,
                        message.get("caption"),
                    )

                for message in table_messages:
                    storage.append_table(
                        message["step"],
                        message.get("global_step"),
                        message["name"],
                        message["table_data"],
                    )

                # STEP 3: Flush once
                storage.flush_all()
                total = len(media_messages) + len(table_messages)
                batch_time = time.time() - batch_start
                logger.debug(
                    f"Processed and flushed {total} metadata entries ({len(media_messages)} media, {len(table_messages)} tables) in {batch_time*1000:.1f}ms"
                )
            else:
                # No messages - sleep
                time.sleep(0.01)

        # Final flush
        logger.info("Metadata worker shutting down...")
        storage.flush_all()
        storage.close()
        logger.info("Metadata worker stopped")

    except Exception as e:
        logger.error(f"Metadata worker error: {e}")
        raise


def histogram_worker_main(board_dir: Path, queue: Any, stop_event: Any):
    """Worker process for histograms (Lance storage)

    Args:
        board_dir: Board directory
        queue: Histogram queue (mp.Queue)
        stop_event: Stop event (mp.Event)
    """
    logger.remove()
    logger.add(
        board_dir / "logs" / "histogram_worker.log", rotation="10 MB", level="DEBUG"
    )

    storage = HistogramStorage(board_dir / "data", num_bins=64)

    batch_threshold = 500  # Total histograms across all names
    batch_interval = 2.0

    logger.info("Histogram worker started")

    try:
        while not stop_event.is_set():
            # Check queue size before draining
            try:
                queue_size = queue.qsize()
            except:
                queue_size = 0

            # STEP 1: Drain ALL messages from queue
            messages = []
            drain_start = time.time()
            while True:
                try:
                    messages.append(queue.get_nowait())
                except Empty:
                    break
            drain_time = time.time() - drain_start

            # STEP 2: Process all messages (add to storage buffers)
            if messages:
                batch_start = time.time()
                for message in messages:
                    storage.append_histogram(
                        message["step"],
                        message.get("global_step"),
                        message["name"],
                        message["values"],
                        message.get("num_bins", 64),
                        message.get("precision", "exact"),
                    )

                # STEP 3: Flush ALL buffers once
                storage.flush()
                batch_time = time.time() - batch_start
                logger.info(
                    f"Drained {len(messages)}/{queue_size} from queue in {drain_time*1000:.1f}ms, processed+flushed in {batch_time*1000:.1f}ms"
                )
            else:
                # No messages - sleep
                time.sleep(0.01)

        # Final flush
        logger.info("Histogram worker shutting down, draining queue...")
        drained = 0
        while not queue.empty():
            try:
                message = queue.get_nowait()
                storage.append_histogram(
                    message["step"],
                    message.get("global_step"),
                    message["name"],
                    message["values"],
                    message.get("num_bins", 64),
                    message.get("precision", "exact"),
                )
                drained += 1
            except Empty:
                break

        storage.flush()
        storage.close()
        logger.info(f"Histogram worker stopped (drained {drained}, flushed all)")

    except Exception as e:
        logger.error(f"Histogram worker error: {e}")
        raise
