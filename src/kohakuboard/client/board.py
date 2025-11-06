"""Main Board class for non-blocking experiment logging"""

import atexit
import json
import multiprocessing as mp
import signal
import sys
import time
import uuid
import weakref
from copy import deepcopy
from datetime import datetime, timezone
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from kohakuboard.client.capture import MemoryOutputCapture, OutputCapture
from kohakuboard.client.types import (
    Histogram,
    KernelDensity,
    Media,
    Table,
    TensorLog,
)
from kohakuboard.client.types.media import is_media
from kohakuboard.client.writer import writer_process_main

# Get logger for Board
from kohakuboard.logger import get_logger


# Global weakref registry - doesn't prevent GC
_active_boards_weakrefs = []
_cleanup_registered = False


def _cleanup_all_boards():
    """Cleanup all boards at exit - uses weakrefs so doesn't prevent GC"""
    for board_ref in _active_boards_weakrefs:
        board = board_ref()
        if board is not None and hasattr(board, "finish"):
            try:
                board.finish()
            except:
                pass


class Board:
    """Board - Main interface for logging ML experiments

    Features:
        - Non-blocking logging using background process
        - Automatic step tracking + explicit global_step
        - Parquet-based storage for efficient queries
        - Media (images) and table logging
        - stdout/stderr capture to log file

    Example:
        >>> # Create board - auto-finishes on program exit via atexit
        >>> board = Board(name="resnet_training", config={"lr": 0.001})
        >>>
        >>> for epoch in range(100):
        ...     board.step()  # Explicit step increment
        ...
        ...     for batch_idx, (data, target) in enumerate(train_loader):
        ...         loss = train_step(data, target)
        ...
        ...         # Log scalars (non-blocking)
        ...         board.log(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        ...
        ...         # Log images occasionally
        ...         if batch_idx % 100 == 0:
        ...             board.log_images("predictions", images[:8], caption="Model predictions")
        ...
        ...     # Log table at end of epoch
        ...     board.log_table("metrics", [
        ...         {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        ...     ])
        ...
        >>> # board.finish() called automatically on program exit
    """

    def __init__(
        self,
        name: str,
        board_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        capture_output: bool = True,
        remote_url: Optional[str] = None,
        remote_token: Optional[str] = None,
        remote_project: Optional[str] = None,
        sync_enabled: bool = False,
        sync_interval: int = 10,
        memory_mode: bool = False,
    ):
        """Create a new Board for logging

        Args:
            name: Human-readable name for this board
            board_id: Unique ID (auto-generated if not provided)
            config: Configuration dict for this run (hyperparameters, etc.)
            base_dir: Base directory for boards (default: ./kohakuboard)
            capture_output: Whether to capture stdout/stderr to log file
            remote_url: Remote server base URL for sync (e.g., https://board.example.com)
            remote_token: Authentication token for remote server
            remote_project: Project name on remote server (default: "local")
            sync_enabled: Whether to enable real-time sync to remote server
            sync_interval: Sync check interval in seconds (default: 10)
            memory_mode: Store data in memory-only mode (requires remote sync to persist)
        """

        # Board metadata
        self.name = name
        self.board_id = board_id or self._generate_id()
        self.config = config or {}
        self.created_at = datetime.now(timezone.utc)
        self.memory_mode = memory_mode

        # Setup directories
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "kohakuboard"
        self.board_dir = self.base_dir / self.board_id

        self.board_dir.mkdir(parents=True, exist_ok=True)
        (self.board_dir / "data").mkdir(exist_ok=True)
        (self.board_dir / "media").mkdir(exist_ok=True)
        (self.board_dir / "logs").mkdir(exist_ok=True)

        # Step tracking
        # _step increments on EVERY log/media/table call (auto-increment)
        # _global_step is set explicitly via step() method
        self._step = -1  # Start at -1, will be 0 on first log
        self._global_step: int = 0  # Start at 0, NOT None!

        # Shutdown tracking
        self._is_finishing = False  # Prevent re-entrant finish() calls
        self._interrupt_count = 0  # Track Ctrl+C presses for force exit

        self.queue = mp.Queue(maxsize=50000)
        self.stop_event = mp.Event()

        # Track active SharedMemory blocks for cleanup
        self._shared_memory_blocks = []

        # Setup board logger FIRST (stdout + file)
        self.logger = get_logger("BOARD")

        # Add file handler to board logger (in addition to stdout)
        # Note: loguru is global, so this adds a handler that ALL loggers will use
        # But we filter it to only log BOARD messages to board.log
        self._log_handler_id = None
        if not self.memory_mode:
            self._log_handler_id = self.logger._logger.add(
                self.board_dir / "logs" / "board.log",
                format="<cyan>{time:HH:mm:ss.SSS}</cyan> | <fg #FF00CD>{extra[api_name]: <8}</fg #FF00CD> | <level>{level: <8}</level> | {message}",
                level="DEBUG",
                rotation="10 MB",
                retention="7 days",
                colorize=False,  # No ANSI color codes in log files
                filter=lambda record: record["extra"].get("api_name") == "BOARD",
            )

        # Save board metadata (may be in-memory only)
        self._save_metadata()

        # Prepare sync configuration (handled inside writer process)
        self._sync_config: dict[str, Any] | None = None
        if sync_enabled and remote_url and remote_token:
            self._sync_config = {
                "remote_url": remote_url,
                "remote_token": remote_token,
                "project": remote_project or "local",
                "run_id": self.board_id,
                "sync_interval": sync_interval,
                "metadata": deepcopy(self._metadata),
            }
            if self.memory_mode:
                # Memory mode requires aggressive sync intervals
                self._sync_config["sync_interval"] = min(
                    self._sync_config["sync_interval"], 0.25
                )
        elif sync_enabled:
            self.logger.warning(
                "Sync enabled but remote_url or remote_token missing; sync worker will not start"
            )

        if self.memory_mode and not self._sync_config:
            self.logger.warning(
                "Memory mode enabled without remote sync configuration; "
                "logs will exist only in memory for this process."
            )

        # Start single writer process
        self.writer_process = mp.Process(
            target=writer_process_main,
            args=(
                self.board_dir,
                self.queue,
                self.stop_event,
                self._sync_config,
                self.memory_mode,
            ),
            daemon=False,
        )
        self.writer_process.start()

        # Output capture
        self.capture_output = capture_output
        if self.capture_output:
            if self.memory_mode:
                self.output_capture = MemoryOutputCapture(self._enqueue_log_chunk)
            else:
                self.output_capture = OutputCapture(
                    self.board_dir / "logs" / "output.log"
                )
            self.output_capture.start()
        else:
            self.output_capture = None

        # Register in global weakref list (doesn't prevent GC)
        global _active_boards_weakrefs, _cleanup_registered
        _active_boards_weakrefs.append(weakref.ref(self))

        # Register global atexit handler once
        if not _cleanup_registered:
            atexit.register(_cleanup_all_boards)
            _cleanup_registered = True

        # Register signal handlers with weakref (doesn't prevent GC)
        self._register_signal_handlers()

        self.logger.info(f"Board created: {self.name} (ID: {self.board_id})")
        self.logger.info(f"Board directory: {self.board_dir}")

    def log(
        self,
        auto_step: bool = True,
        **metrics: Union[int, float, Media, Table, Histogram],
    ):
        """Unified logging method supporting all data types (non-blocking)

        Supports scalars, Media, Table, and Histogram objects in a single call.
        All values logged in one call share the same step, eliminating step inflation.

        Args:
            auto_step: Whether to auto-increment step (default: True)
            **metrics: Metric name-value pairs
                - Scalars: Python numbers, single-item tensors, numpy scalars
                - Media: Media objects (images/video/audio)
                - Table: Table objects (tabular data)
                - Histogram: Histogram objects (distributions)

        Examples:
            >>> # Scalars only
            >>> board.log(loss=0.5, accuracy=0.95, lr=0.001)

            >>> # Mixed types - all share same step
            >>> board.log(
            ...     loss=0.5,
            ...     sample_img=Media(image_array),
            ...     results=Table(data),
            ...     gradients=Histogram(grad_values)
            ... )

            >>> # Multiple histograms without step inflation
            >>> board.log(
            ...     grad_layer1=Histogram(grad1),
            ...     grad_layer2=Histogram(grad2),
            ...     grad_layer3=Histogram(grad3),
            ...     auto_step=True  # All get same step
            ... )

            >>> # Control step manually
            >>> board.log(loss=0.5, auto_step=False)  # No step increment
        """
        # Increment step if auto_step is enabled
        if auto_step:
            self._step += 1

        # Categorize values by type
        scalars = {}
        media_logs: list[tuple[str, Media]] = []
        table_logs: list[tuple[str, Table]] = []
        histogram_logs: list[tuple[str, Histogram]] = []
        tensor_logs: list[tuple[str, TensorLog]] = []
        kde_logs: list[tuple[str, KernelDensity]] = []

        for key, value in metrics.items():
            if isinstance(value, TensorLog):
                tensor_logs.append((key, value))
            elif isinstance(value, KernelDensity):
                kde_logs.append((key, value))
            elif isinstance(value, Histogram):
                # Histogram object
                histogram_logs.append((key, value))
            elif isinstance(value, Table):
                # Table object
                table_logs.append((key, value))
            elif is_media(value):
                # Media object
                media_logs.append((key, value))
            else:
                # Scalar - convert to Python number
                scalars[key] = self._to_python_number(value)

        # Use batched message if we have multiple types
        multi_category = (
            sum(
                [
                    bool(scalars),
                    bool(media_logs),
                    bool(table_logs),
                    bool(histogram_logs),
                    bool(tensor_logs),
                    bool(kde_logs),
                ]
            )
            > 1
        )

        if multi_category and not (tensor_logs or kde_logs):
            # Send single batched message
            self._send_batch_message(scalars, media_logs, table_logs, histogram_logs)
        else:
            # Send individual messages (backward compatible)
            if scalars:
                self._send_scalar_message(scalars)
            if media_logs:
                for media_name, media_obj in media_logs:
                    self._send_media_message(media_name, media_obj)
            if table_logs:
                for table_name, table_obj in table_logs:
                    self._send_table_message(table_name, table_obj)
            if histogram_logs:
                for hist_name, hist_obj in histogram_logs:
                    self._send_histogram_message(hist_name, hist_obj)
            if tensor_logs:
                for tensor_name, tensor_obj in tensor_logs:
                    self._send_tensor_message(tensor_name, tensor_obj)
            if kde_logs:
                for kde_name, kde_obj in kde_logs:
                    self._send_kernel_density_message(kde_name, kde_obj)

    def _send_scalar_message(self, scalars: Dict[str, Union[int, float]]):
        """Send scalar metrics message"""
        message = {
            "type": "scalar",
            "step": self._step,
            "global_step": self._global_step,
            "metrics": scalars,
            "timestamp": datetime.now(timezone.utc),
        }
        self.queue.put(message)

    def _send_media_message(self, name: str, media_obj: Media):
        """Send single media message"""
        message = {
            "type": "media",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "media_type": media_obj.media_type,
            "media_data": media_obj.data,
            "caption": media_obj.caption,
        }
        self.queue.put(message)

    def _send_table_message(self, name: str, table_obj: Table):
        """Send single table message"""
        message = {
            "type": "table",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "table_data": table_obj.to_dict(),
        }
        self.queue.put(message)

    def _create_shared_memory(self, array: np.ndarray, name_prefix: str) -> tuple:
        """Create SharedMemory from numpy array

        Args:
            array: numpy array to store in shared memory
            name_prefix: prefix for shared memory name

        Returns:
            tuple of (shm_name, size, dtype_str)
        """
        # Create unique name for this shared memory block
        shm_name = f"{name_prefix}_{uuid.uuid4().hex[:16]}"

        # Ensure array is contiguous
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        # Create shared memory block
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes, name=shm_name)

        # Copy data to shared memory
        shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shm_array[:] = array[:]

        # Track this block for cleanup (keep reference to prevent premature GC)
        self._shared_memory_blocks.append(shm)

        return shm_name, array.nbytes, str(array.dtype)

    def _send_histogram_message(self, name: str, hist_obj: Histogram):
        """Send single histogram message using SharedMemory for data transfer"""
        hist_data = hist_obj.to_dict()

        if hist_data.get("computed", False):
            # Precomputed histogram - send bins/counts via SharedMemory
            bins_array = np.array(hist_data["bins"], dtype=np.float32)
            counts_array = np.array(hist_data["counts"], dtype=np.int32)

            bins_shm_name, bins_size, bins_dtype = self._create_shared_memory(
                bins_array, "hist_bins"
            )
            counts_shm_name, counts_size, counts_dtype = self._create_shared_memory(
                counts_array, "hist_counts"
            )

            message = {
                "type": "histogram",
                "step": self._step,
                "global_step": self._global_step,
                "name": name,
                "precision": hist_data["precision"],
                "precomputed": True,
                "shared_memory": {
                    "bins_name": bins_shm_name,
                    "bins_size": bins_size,
                    "bins_dtype": bins_dtype,
                    "bins_shape": bins_array.shape,
                    "counts_name": counts_shm_name,
                    "counts_size": counts_size,
                    "counts_dtype": counts_dtype,
                    "counts_shape": counts_array.shape,
                },
            }
        else:
            # Raw values - send to writer for computation via SharedMemory
            values_array = np.array(hist_data["values"], dtype=np.float32)

            values_shm_name, values_size, values_dtype = self._create_shared_memory(
                values_array, "hist_values"
            )

            message = {
                "type": "histogram",
                "step": self._step,
                "global_step": self._global_step,
                "name": name,
                "num_bins": hist_data["num_bins"],
                "precision": hist_data["precision"],
                "precomputed": False,
                "shared_memory": {
                    "values_name": values_shm_name,
                    "values_size": values_size,
                    "values_dtype": values_dtype,
                    "values_shape": values_array.shape,
                },
            }
        self.queue.put(message)

    def _send_tensor_message(self, name: str, tensor_obj: TensorLog):
        """Send tensor payload via SharedMemory."""
        tensor_array = tensor_obj.to_numpy()
        tensor_meta = tensor_obj.summary()

        shm_name, shm_size, shm_dtype = self._create_shared_memory(
            tensor_array, "tensor"
        )

        message = {
            "type": "tensor",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "tensor_meta": tensor_meta,
            "shared_memory": {
                "values_name": shm_name,
                "values_size": shm_size,
                "values_dtype": shm_dtype,
                "values_shape": tensor_array.shape,
            },
        }
        self.queue.put(message)

    def _send_kernel_density_message(self, name: str, kde_obj: KernelDensity):
        """Send kernel density coefficients via SharedMemory."""
        grid_array, density_array = kde_obj.to_arrays()
        kde_meta = kde_obj.summary()

        grid_shm_name, grid_size, grid_dtype = self._create_shared_memory(
            grid_array, "kde_grid"
        )
        density_shm_name, density_size, density_dtype = self._create_shared_memory(
            density_array, "kde_density"
        )

        message = {
            "type": "kernel_density",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "kde_meta": kde_meta,
            "shared_memory": {
                "grid_name": grid_shm_name,
                "grid_size": grid_size,
                "grid_dtype": grid_dtype,
                "grid_shape": grid_array.shape,
                "density_name": density_shm_name,
                "density_size": density_size,
                "density_dtype": density_dtype,
                "density_shape": density_array.shape,
            },
        }
        self.queue.put(message)

    def _send_batch_message(
        self,
        scalars: Dict[str, Union[int, float]],
        media_logs: List[tuple],
        table_logs: List[tuple],
        histogram_logs: List[tuple],
    ):
        """Send batched message containing multiple types"""
        batch_data = {
            "type": "batch",
            "step": self._step,
            "global_step": self._global_step,
            "timestamp": datetime.now(timezone.utc),
        }

        # Add scalars
        if scalars:
            batch_data["scalars"] = scalars

        # Add media
        if media_logs:
            batch_data["media"] = {
                name: {
                    "media_type": media_obj.media_type,
                    "media_data": media_obj.data,
                    "caption": media_obj.caption,
                }
                for name, media_obj in media_logs
            }

        # Add tables
        if table_logs:
            batch_data["tables"] = {
                name: table_obj.to_dict() for name, table_obj in table_logs
            }

        # Add histograms
        if histogram_logs:
            batch_data["histograms"] = {}
            for name, hist_obj in histogram_logs:
                hist_data = hist_obj.to_dict()
                batch_data["histograms"][name] = hist_data

        self.queue.put(batch_data)

    def log_images(
        self,
        name: str,
        images: Union[Any, List[Any]],
        caption: Optional[str] = None,
    ):
        """Log images (non-blocking)

        Supports: PIL Images, numpy arrays, torch Tensors, file paths

        Args:
            name: Name for this media log
            images: Single image or list of images
            caption: Optional caption

        Example:
            >>> # Single image
            >>> board.log_images("prediction", pred_image)
            >>>
            >>> # Multiple images
            >>> board.log_images("samples", [img1, img2, img3], caption="Generated samples")
        """
        # Increment step (auto-increment on every log call)
        self._step += 1

        # Ensure list
        if not isinstance(images, (list, tuple)):
            images = [images]

        message = {
            "type": "media",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "images": images,
            "caption": caption,
        }
        self.queue.put(message)

    def log_video(
        self, name: str, video_path: Union[str, Path], caption: Optional[str] = None
    ):
        """Log video file (non-blocking)

        Args:
            name: Name for this video log
            video_path: Path to video file (mp4, avi, mov, mkv, webm, etc.)
            caption: Optional caption

        Example:
            >>> board.log_video("training_progress", "output.mp4", caption="Training visualization")
        """
        self._step += 1

        message = {
            "type": "media",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "media_type": "video",
            "media_data": video_path,
            "caption": caption,
        }
        self.queue.put(message)

    def log_audio(
        self, name: str, audio_path: Union[str, Path], caption: Optional[str] = None
    ):
        """Log audio file (non-blocking)

        Args:
            name: Name for this audio log
            audio_path: Path to audio file (mp3, wav, flac, ogg, etc.)
            caption: Optional caption

        Example:
            >>> board.log_audio("generated_speech", "output.wav", caption="TTS output")
        """
        self._step += 1

        message = {
            "type": "media",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "media_type": "audio",
            "media_data": audio_path,
            "caption": caption,
        }
        self.queue.put(message)

    def log_table(self, name: str, table: Union[Table, List[Dict[str, Any]]]):
        """Log table data (non-blocking)

        Args:
            name: Name for this table log
            table: Table object or list of dicts

        Example:
            >>> # From list of dicts
            >>> board.log_table("results", [
            ...     {"epoch": 1, "loss": 0.5, "acc": 0.9},
            ...     {"epoch": 2, "loss": 0.3, "acc": 0.95},
            ... ])
            >>>
            >>> # Using Table class
            >>> from kohakuboard.client import Table
            >>> table = Table([{"name": "Alice", "score": 95}])
            >>> board.log_table("scores", table)
        """
        # Increment step (auto-increment on every log call)
        self._step += 1

        # Convert to Table if needed
        if not isinstance(table, Table):
            table = Table(table)

        message = {
            "type": "table",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "table_data": table.to_dict(),
        }
        self.queue.put(message)

    def log_histogram(
        self,
        name: str,
        values: Union[List[float], Any],
        num_bins: int = 64,
        precision: str = "exact",
    ):
        """Log histogram data (non-blocking)

        Args:
            name: Name for this histogram log (supports namespace: "gradients/layer1")
            values: List of values or tensor to create histogram from
            num_bins: Number of bins for histogram (default: 64)
            precision: "exact" (int32, default) or "compact" (uint8, ~1% loss)

        Example:
            >>> # Log gradient histogram (compact)
            >>> grads = [p.grad.flatten().cpu().numpy() for p in model.parameters()]
            >>> board.log_histogram("gradients/all", np.concatenate(grads))
            >>>
            >>> # Log parameter histogram (exact counts)
            >>> params = model.fc1.weight.detach().cpu().numpy().flatten()
            >>> board.log_histogram("params/fc1_weight", params, precision="exact")
        """
        # Increment step (auto-increment on every log call)
        self._step += 1

        # Check queue size and warn if getting full
        try:
            queue_size = self.queue.qsize()
            if queue_size > 40000:
                self.logger.warning(
                    f"Queue size is {queue_size}/50000. Consider reducing logging frequency."
                )
        except NotImplementedError:
            pass  # qsize() not supported on all platforms

        # Convert tensor to list if needed
        if hasattr(values, "cpu"):  # PyTorch tensor
            values = values.detach().cpu().numpy().flatten().tolist()
        elif hasattr(values, "numpy"):  # NumPy array
            values = values.flatten().tolist()
        elif not isinstance(values, list):
            values = list(values)

        message = {
            "type": "histogram",
            "step": self._step,
            "global_step": self._global_step,
            "name": name,
            "values": values,
            "num_bins": num_bins,
            "precision": precision,
        }
        self.queue.put(message)

    def step(self, increment: int = 1):
        """Explicit step increment

        This sets the global_step. All logs between step() calls
        belong to the previous global_step value.

        Args:
            increment: Step increment (default: 1)

        Example:
            >>> for epoch in range(100):
            ...     board.step()  # global_step = 0, 1, 2, ...
            ...     for batch in train_loader:
            ...         loss = train_step(batch)
            ...         board.log(loss=loss)  # All batches share same global_step
        """
        self._global_step += increment

    def flush(self):
        """Flush all pending logs to disk (blocking)

        Normally logs are flushed automatically. Use this for
        critical checkpoints or before long-running operations.
        """
        # Send flush signal
        flush_msg = {"type": "flush"}
        self.queue.put(flush_msg)

    def log_tensor(
        self,
        name: str,
        values: Any,
        precision: Any | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Log full tensor payload (non-blocking).

        Args:
            name: Log name (supports namespaces via '/')
            values: Tensor/array-like payload
            precision: Optional dtype override during serialization
            metadata: Optional metadata dict stored alongside tensor
        """
        self._step += 1

        try:
            queue_size = self.queue.qsize()
            if queue_size > 40000:
                self.logger.warning(
                    f"Queue size is {queue_size}/50000. Consider reducing logging frequency."
                )
        except NotImplementedError:
            pass

        tensor_obj = TensorLog(values, precision=precision, metadata=metadata)
        self._send_tensor_message(name, tensor_obj)

    def log_kernel_density(
        self,
        name: str,
        grid: Any | None = None,
        density: Any | None = None,
        *,
        values: Any | None = None,
        num_points: int = 256,
        kernel: str = "gaussian",
        bandwidth: float | str | None = None,
        sample_count: int | None = None,
        range_min: float | None = None,
        range_max: float | None = None,
        percentile_clip: tuple[float, float] = (1.0, 99.0),
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Log kernel density data (non-blocking).

        Args:
            name: Log name (namespace supported via '/')
            grid: Optional precomputed grid positions (1D array-like)
            density: Optional precomputed density values aligned with `grid`
            values: Optional raw samples; if provided the KDE is computed automatically
            num_points: Number of evaluation points when computing from raw samples
            kernel: Kernel name (currently only 'gaussian' is supported)
            bandwidth: Bandwidth override (float) or "auto"/None for Scott's rule
            sample_count: Optional explicit sample count metadata
            range_min: Optional recommended minimum range
            range_max: Optional recommended maximum range
            percentile_clip: Percentile bounds (min, max) used when deriving range
            metadata: Optional metadata dict persisted alongside KDE
        """
        self._step += 1

        try:
            queue_size = self.queue.qsize()
            if queue_size > 40000:
                self.logger.warning(
                    f"Queue size is {queue_size}/50000. Consider reducing logging frequency."
                )
        except NotImplementedError:
            pass

        perc_min, perc_max = percentile_clip
        kde_obj = KernelDensity(
            raw_values=values,
            grid=grid,
            density=density,
            kernel=kernel,
            bandwidth=bandwidth,
            num_points=num_points,
            percentile_min=float(perc_min),
            percentile_max=float(perc_max),
            sample_count=sample_count,
            range_min=range_min,
            range_max=range_max,
            metadata=metadata,
        )
        self._send_kernel_density_message(name, kde_obj)

    def finish(self):
        """Finish logging and clean up

        Flushes all buffers, stops writer process, and releases resources.
        Called automatically on exit, Ctrl+C, or exceptions.
        """
        if not hasattr(self, "writer_process"):
            return  # Already finished

        if self._is_finishing:
            self.logger.debug("finish() already in progress, skipping re-entrant call")
            return

        self._is_finishing = True
        self.logger.info(f"Finishing board: {self.name}")

        # Check queue size
        try:
            queue_size = self.queue.qsize()
            self.logger.info(f"Queue size: {queue_size} messages")
        except:
            queue_size = 0

        # Stop output capture
        if self.output_capture:
            self.output_capture.stop()

        # Signal workers to stop (FIRST - let writer start draining)
        self.stop_event.set()
        self.logger.info("Stop event set, waiting for writer to drain queue...")

        # Give workers a moment to start draining
        time.sleep(0.5)

        # Poll queue to monitor draining (max 30 seconds)
        max_wait_time = 30
        start_time = time.time()
        last_size = queue_size

        while time.time() - start_time < max_wait_time:
            try:
                current_size = self.queue.qsize()

                if current_size == 0:
                    self.logger.info("Queue empty - waiting 1s for writer to finish...")
                    time.sleep(1)
                    break

                # Log progress
                if current_size != last_size:
                    self.logger.info(f"Queue: {current_size} remaining")
                    last_size = current_size

                time.sleep(0.5)
            except KeyboardInterrupt:
                self.logger.warning("Interrupted during drain - forcing shutdown")
                break
            except:
                break

        if time.time() - start_time >= max_wait_time:
            self.logger.error(f"Timeout after {max_wait_time}s - KILLING writer")
            if self.writer_process.is_alive():
                self.writer_process.kill()
            self.logger.error("Writer killed, exiting")
            delattr(self, "writer_process")
            sys.exit(1)

        # Wait for writer to exit
        self.logger.info("Waiting for writer process to exit...")
        self.writer_process.join(timeout=2)

        if self.writer_process.is_alive():
            self.logger.warning("Writer still alive after 2s, killing...")
            self.writer_process.kill()

        # Clean up SharedMemory blocks
        if hasattr(self, "_shared_memory_blocks"):
            self.logger.debug(
                f"Cleaning up {len(self._shared_memory_blocks)} SharedMemory blocks..."
            )
            for shm in self._shared_memory_blocks:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up SharedMemory: {e}")
            self._shared_memory_blocks.clear()

        self.logger.info(f"Board finished: {self.name}")

        # Remove to prevent double-call
        delattr(self, "writer_process")

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown using weakref

        Uses weakref to avoid circular references that prevent GC.
        Handles:
        - SIGINT (Ctrl+C)
        - SIGTERM (kill command)
        - Uncaught exceptions
        """
        # Use weakref to avoid keeping Board alive
        weak_self = weakref.ref(self)
        interrupt_count = [0]  # Use list to allow modification in closure

        def signal_handler(signum, frame):
            """Handle termination signals (double Ctrl+C for force exit)"""
            board = weak_self()
            if board is None:
                return  # Board was garbage collected

            sig_name = signal.Signals(signum).name
            interrupt_count[0] += 1

            if interrupt_count[0] == 1:
                board.logger.warning(
                    f"Received {sig_name}, shutting down gracefully..."
                )
                board.logger.warning(
                    "Press Ctrl+C again within 3 seconds to FORCE EXIT"
                )
                try:
                    board.finish()
                except Exception as e:
                    board.logger.error(f"Error during graceful shutdown: {e}")
                finally:
                    board.logger.info("Shutdown complete")
                    sys.exit(0)
            elif interrupt_count[0] == 2:
                board.logger.error(
                    "Second Ctrl+C - KILLING writer process (data will be lost!)"
                )
                if hasattr(board, "writer_process") and board.writer_process.is_alive():
                    board.writer_process.kill()
                    time.sleep(0.5)
                board.logger.error("Force exit")
                sys.exit(1)
            else:
                # Third+ interrupt - nuclear option
                import os

                os._exit(1)

        # Register signal handlers (Ctrl+C, kill)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Hook into sys.excepthook for uncaught exceptions using weakref
        original_excepthook = sys.excepthook

        def exception_handler(exc_type, exc_value, exc_traceback):
            """Handle uncaught exceptions"""
            board = weak_self()
            if board is not None:
                board.logger.error(
                    f"Uncaught exception: {exc_type.__name__}: {exc_value}"
                )
                board.logger.error("Attempting graceful shutdown...")
                try:
                    board.finish()
                except Exception as e:
                    board.logger.error(f"Error during exception cleanup: {e}")
            # Call original excepthook to print traceback
            original_excepthook(exc_type, exc_value, exc_traceback)

        sys.excepthook = exception_handler

    def _to_python_number(self, value: Any) -> Union[int, float]:
        """Convert various numeric types to Python number"""
        # Already Python number
        if isinstance(value, (int, float)):
            return value

        # Numpy
        if hasattr(value, "item"):
            return value.item()

        # Torch tensor
        if hasattr(value, "cpu"):
            return value.detach().cpu().item()

        raise ValueError(f"Cannot convert {type(value)} to number")

    def _generate_id(self) -> str:
        """Generate unique board ID"""
        # Use timestamp + random UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_id}"

    def _save_metadata(self):
        """Save board metadata to JSON"""
        metadata = {
            "name": self.name,
            "board_id": self.board_id,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
        }

        self._metadata = metadata

        if self.memory_mode:
            return

        metadata_file = self.board_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _enqueue_log_chunk(self, stream: str, text: str):
        """Send log chunk to writer process for in-memory storage."""
        try:
            self.queue.put_nowait({"type": "log", "stream": stream, "data": text})
        except Exception:
            pass

    def __del__(self):
        """Destructor - cleanup when board is garbage collected"""
        # Try to finish if not already finished
        if hasattr(self, "writer_process") and not self._is_finishing:
            try:
                self.finish()
            except:
                pass  # Silent fail in __del__

    def __repr__(self) -> str:
        return f"Board(name={self.name!r}, id={self.board_id!r})"

    def __enter__(self):
        print("Board Enter")
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Board exit")
        """Context manager cleanup"""
        self.flush()
        self.finish()
