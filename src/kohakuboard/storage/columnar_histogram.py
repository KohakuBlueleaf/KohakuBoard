"""Histogram storage using KohakuVault ColumnVault (grouped by namespace)

Strategy:
1. One ColumnVault DB per namespace:
   - params/layer1, params/layer2 → params_i32.db (if int32)
   - gradients/layer1, gradients/layer2 → gradients_i32.db
   - custom → custom_i32.db
2. Precision is per-file (suffix: _u8 or _i32)
3. Schema includes "name" field (full name with namespace)

Schema:
- step: i64
- global_step: i64
- name: bytes (variable-size string)
- counts: bytes:N (fixed-size, N=num_bins for u8, N=num_bins*4 for i32)
- min: f64
- max: f64

Fixed-size bytes optimization for counts reduces overhead!
"""

import struct
from pathlib import Path
from typing import Any

import numpy as np
from kohakuvault import ColumnVault

from kohakuboard.logger import get_logger


class ColumnVaultHistogramStorage:
    """Histogram storage with namespace-based grouping using ColumnVault"""

    def __init__(self, base_dir: Path, num_bins: int = 64, logger=None):
        """Initialize histogram storage

        Args:
            base_dir: Base directory
            num_bins: Number of bins (default: 64)
            logger: Optional logger instance (if None, creates file-only logger)
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Optional logger injection
        if logger is not None:
            self.logger = logger
        else:
            # Default: file-only logger (for client/writer)
            log_file = base_dir.parent / "logs" / "storage.log"
            self.logger = get_logger("STORAGE", file_only=True, log_file=log_file)

        self.histograms_dir = base_dir / "histograms"
        self.histograms_dir.mkdir(exist_ok=True)

        self.num_bins = num_bins

        # Cache of ColumnVault instances
        self.vaults: dict[str, ColumnVault] = {}

        # Buffers grouped by namespace + precision
        # Key: "{namespace}_{u8|i32}"
        self.buffers: dict[str, list[dict[str, Any]]] = {}

        # Fixed-size bytes for counts
        # uint8: 1 byte per bin → bytes:num_bins
        # int32: 4 bytes per bin → bytes:(num_bins*4)
        self.counts_size_u8 = num_bins
        self.counts_size_i32 = num_bins * 4

    def _get_or_create_vault(self, buffer_key: str) -> ColumnVault:
        """Get or create ColumnVault instance for a namespace+precision

        Args:
            buffer_key: Key like "gradients_i32" or "params_u8"

        Returns:
            ColumnVault instance
        """
        if buffer_key not in self.vaults:
            db_path = self.histograms_dir / f"{buffer_key}.db"

            # Check if database file exists before creating ColumnVault
            is_new_db = not db_path.exists()

            # Create ColumnVault (WAL is default in SQLite)
            suffix = buffer_key.split("_")[-1]
            if suffix == "u8":
                cv = ColumnVault(str(db_path))
                counts_dtype = f"bytes:{self.counts_size_u8}"
            else:
                cv = ColumnVault(str(db_path))
                counts_dtype = f"bytes:{self.counts_size_i32}"

            # Create columns only if database is new
            if is_new_db:
                # New database - create schema
                cv.create_column("step", "i64")
                cv.create_column("global_step", "i64")
                cv.create_column("name", "bytes")  # Variable-size for names
                cv.create_column("counts", counts_dtype)  # Fixed-size!
                cv.create_column("min", "f64")
                cv.create_column("max", "f64")

            self.vaults[buffer_key] = cv
            self.logger.debug(f"Opened ColumnVault for histograms: {buffer_key}")

        return self.vaults[buffer_key]

    def append_histogram(
        self,
        step: int,
        global_step: int | None,
        name: str,
        values: list[float] | None = None,
        num_bins: int = None,
        precision: str = "exact",
        bins: list[float] | None = None,
        counts: list[int] | None = None,
    ):
        """Append histogram

        Args:
            step: Step number
            global_step: Global step
            name: Histogram name (e.g., "gradients/layer1")
            values: Raw values (if not precomputed)
            num_bins: Ignored (uses self.num_bins)
            precision: "exact" (int32, default) or "compact" (uint8)
            bins: Precomputed bin edges (optional)
            counts: Precomputed bin counts (optional)
        """
        # Check if precomputed
        if bins is not None and counts is not None:
            # Precomputed histogram - use provided bins/counts
            bins_array = np.array(bins, dtype=np.float32)
            counts_array = np.array(counts, dtype=np.int32)

            # Use first and last bin edges as min/max
            p1 = float(bins_array[0])
            p99 = float(bins_array[-1])

            # Convert counts based on precision
            if precision == "compact":
                max_count = counts_array.max()
                final_counts = (
                    (counts_array / max_count * 255).astype(np.uint8)
                    if max_count > 0
                    else counts_array.astype(np.uint8)
                )
                suffix = "_u8"
            else:
                final_counts = counts_array.astype(np.int32)
                suffix = "_i32"
        else:
            # Compute histogram from raw values
            if not values:
                return

            values_array = np.array(values, dtype=np.float32)
            values_array = values_array[np.isfinite(values_array)]

            if len(values_array) == 0:
                return

            # Compute p1-p99 range
            p1 = float(np.percentile(values_array, 1))
            p99 = float(np.percentile(values_array, 99))

            if p99 - p1 < 1e-6:
                p1 = float(values_array.min())
                p99 = float(values_array.max())
                if p99 - p1 < 1e-6:
                    p1 -= 0.5
                    p99 += 0.5

            # Compute histogram
            counts_array, _ = np.histogram(
                values_array, bins=self.num_bins, range=(p1, p99)
            )

            # Convert based on precision
            if precision == "compact":
                max_count = counts_array.max()
                final_counts = (
                    (counts_array / max_count * 255).astype(np.uint8)
                    if max_count > 0
                    else counts_array.astype(np.uint8)
                )
                suffix = "_u8"
            else:
                final_counts = counts_array.astype(np.int32)
                suffix = "_i32"

        # Serialize counts to fixed-size bytes
        if suffix == "_u8":
            # uint8: 1 byte per bin
            counts_bytes = final_counts.tobytes()
        else:
            # int32: 4 bytes per bin, little-endian
            counts_bytes = final_counts.tobytes()

        # Extract namespace
        namespace = name.split("/")[0] if "/" in name else name.replace("/", "__")
        buffer_key = f"{namespace}{suffix}"

        # Initialize buffer
        if buffer_key not in self.buffers:
            self.buffers[buffer_key] = []

        # Add to buffer
        self.buffers[buffer_key].append(
            {
                "step": step,
                "global_step": global_step if global_step is not None else step,
                "name": name.encode("utf-8"),  # Encode string to bytes
                "counts": counts_bytes,
                "min": float(p1),
                "max": float(p99),
            }
        )

    def flush(self):
        """Flush all buffers to ColumnVault DBs"""
        if not self.buffers:
            return

        total_entries = sum(len(buf) for buf in self.buffers.values())
        total_files = len(self.buffers)

        for buffer_key, buffer in list(self.buffers.items()):
            if not buffer:
                continue

            try:
                cv = self._get_or_create_vault(buffer_key)

                # Extract columns for batch extend
                steps = [row["step"] for row in buffer]
                global_steps = [row["global_step"] for row in buffer]
                names = [row["name"] for row in buffer]
                counts_list = [row["counts"] for row in buffer]
                mins = [row["min"] for row in buffer]
                maxs = [row["max"] for row in buffer]

                # Batch append with .extend()
                cv["step"].extend(steps)
                cv["global_step"].extend(global_steps)
                cv["name"].extend(names)
                cv["counts"].extend(counts_list)
                cv["min"].extend(mins)
                cv["max"].extend(maxs)

                self.logger.debug(
                    f"Flushed {len(buffer)} histograms to {buffer_key}.db"
                )
                buffer.clear()

            except Exception as e:
                self.logger.error(f"Failed to flush {buffer_key}: {e}")

        self.logger.debug(
            f"Flushed {total_entries} histograms to {total_files} ColumnVault files"
        )

    def close(self):
        """Close storage - flush all remaining buffers and close vaults"""
        self.flush()

        # Close all ColumnVault instances
        for buffer_key, cv in self.vaults.items():
            try:
                cv.close()
            except:
                pass  # Ignore close errors

        self.vaults.clear()
        self.logger.debug("Histogram storage closed")
