"""Histogram storage using KohakuVault's ColumnVault (grouped by namespace)

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

from pathlib import Path
from typing import Any

import numpy as np
from kohakuvault import ColumnVault

from kohakuboard.logger import get_logger


class ColumnVaultHistogramStorage:
    """Histogram storage with namespace-based grouping using KohakuVault's ColumnVault

    Uses blob-based columnar storage with Rust-managed layout for efficient
    histogram data management.
    """

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

        # Cache of ColumnVault instances keyed by (namespace, bin_count, precision)
        self.vaults: dict[tuple[str, int, str], ColumnVault] = {}

        # Buffers grouped by namespace + bin_count + precision
        self.buffers: dict[tuple[str, int, str], list[dict[str, Any]]] = {}

        # Fixed-size bytes for counts
        # uint8: 1 byte per bin → bytes:num_bins
        # int32: 4 bytes per bin → bytes:(num_bins*4)
        self.counts_size_u8 = num_bins
        self.counts_size_i32 = num_bins * 4

    def _get_or_create_vault(
        self, namespace: str, bin_count: int, precision: str
    ) -> ColumnVault:
        """Get or create ColumnVault instance for namespace/bin_count/precision."""
        key = (namespace, bin_count, precision)
        if key not in self.vaults:
            filename = f"{namespace}_{bin_count}_{precision}.db"
            db_path = self.histograms_dir / filename
            legacy_path = self.histograms_dir / f"{namespace}_{precision}.db"

            if (
                not db_path.exists()
                and legacy_path.exists()
                and bin_count == self.num_bins
            ):
                db_path = legacy_path

            is_new_db = not db_path.exists()
            counts_size = bin_count if precision == "u8" else bin_count * 4
            counts_dtype = f"bytes:{counts_size}"

            cv = ColumnVault(str(db_path))

            if is_new_db:
                cv.create_column("step", "i64")
                cv.create_column("global_step", "i64")
                cv.create_column("name", "bytes")
                cv.create_column("counts", counts_dtype)
                cv.create_column("min", "f64")
                cv.create_column("max", "f64")
                self.logger.debug(
                    f"Created ColumnVault for histograms: {db_path.name} "
                    f"(bins={bin_count}, precision={precision})"
                )
            else:
                self.logger.debug(
                    f"Opened ColumnVault for histograms: {db_path.name} "
                    f"(bins={bin_count}, precision={precision})"
                )

            self.vaults[key] = cv

        return self.vaults[key]

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
        num_bins: Number of bins to use when computing from raw values (defaults to self.num_bins)
        precision: "exact" (int32) or "compact" (uint8)
            bins: Precomputed bin edges (optional)
            counts: Precomputed bin counts (optional)
        """
        # Determine effective number of bins and encode counts
        precision_tag = "u8" if precision == "compact" else "i32"

        if bins is not None and counts is not None:
            bins_array = np.array(bins, dtype=np.float32)
            counts_array = np.array(counts, dtype=np.int32)

            # Use first and last bin edges as min/max
            p1 = float(bins_array[0])
            p99 = float(bins_array[-1])

            if precision_tag == "u8":
                max_count = counts_array.max()
                final_counts = (
                    (counts_array / max_count * 255).astype(np.uint8)
                    if max_count > 0
                    else counts_array.astype(np.uint8)
                )
            else:
                final_counts = counts_array.astype(np.int32)

            bin_count = len(final_counts)
        else:
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
            effective_bins = num_bins or self.num_bins
            counts_array, _ = np.histogram(
                values_array, bins=effective_bins, range=(p1, p99)
            )

            if precision_tag == "u8":
                max_count = counts_array.max()
                final_counts = (
                    (counts_array / max_count * 255).astype(np.uint8)
                    if max_count > 0
                    else counts_array.astype(np.uint8)
                )
            else:
                final_counts = counts_array.astype(np.int32)

            bin_count = len(final_counts)

        # Serialize counts to fixed-size bytes
        counts_bytes = final_counts.tobytes()

        # Extract namespace
        namespace = name.split("/")[0] if "/" in name else name.replace("/", "__")
        key = (namespace, bin_count, precision_tag)

        # Initialize buffer
        if key not in self.buffers:
            self.buffers[key] = []

        # Add to buffer
        self.buffers[key].append(
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

        for key, buffer in list(self.buffers.items()):
            if not buffer:
                continue

            try:
                namespace, bin_count, precision_tag = key
                cv = self._get_or_create_vault(namespace, bin_count, precision_tag)

                steps = [row["step"] for row in buffer]
                global_steps = [row["global_step"] for row in buffer]
                names = [row["name"] for row in buffer]
                counts_list = [row["counts"] for row in buffer]
                mins = [row["min"] for row in buffer]
                maxs = [row["max"] for row in buffer]

                cv["step"].extend(steps)
                cv["global_step"].extend(global_steps)
                cv["name"].extend(names)
                cv["counts"].extend(counts_list)
                cv["min"].extend(mins)
                cv["max"].extend(maxs)

                self.logger.debug(
                    f"Flushed {len(buffer)} histograms to "
                    f"{namespace}_{bin_count}_{precision_tag}.db"
                )
                buffer.clear()

            except Exception as e:
                self.logger.error(
                    f"Failed to flush {namespace}_{bin_count}_{precision_tag}: {e}"
                )

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

    def collect_histograms_range(
        self, start_step: int, end_step: int
    ) -> list[dict[str, Any]]:
        """Collect histogram entries for the given step range."""
        results: list[dict[str, Any]] = []

        if not self.histograms_dir.exists():
            return results

        for db_file in self.histograms_dir.glob("*.db"):
            parts = db_file.stem.split("_")
            if len(parts) >= 3 and parts[-2].isdigit():
                precision = parts[-1]
                bin_count = int(parts[-2])
                namespace = "_".join(parts[:-2])
            else:
                precision = parts[-1]
                bin_count = self.num_bins
                namespace = "_".join(parts[:-1])

            counts_size = bin_count if precision == "u8" else bin_count * 4
            dtype = np.uint8 if precision == "u8" else np.dtype("<i4")

            try:
                cv = ColumnVault(str(db_file))
                steps = list(cv["step"])
                global_steps = list(cv["global_step"])
                names = list(cv["name"])
                counts_list = list(cv["counts"])
                mins = list(cv["min"])
                maxs = list(cv["max"])
            except Exception as exc:
                self.logger.error(f"Failed to read histogram DB {db_file}: {exc}")
                continue
            finally:
                try:
                    cv.close()
                except Exception:
                    pass

            iterator = zip(
                steps,
                global_steps,
                names,
                counts_list,
                mins,
                maxs,
            )

            for step, global_step, raw_name, counts_bytes, min_val, max_val in iterator:
                if not (start_step <= step <= end_step):
                    continue

                if len(counts_bytes) != counts_size:
                    self.logger.warning(
                        f"Counts size mismatch for {db_file.name}: "
                        f"expected {counts_size}, got {len(counts_bytes)}"
                    )

                counts_array = np.frombuffer(counts_bytes, dtype=dtype)
                counts = [int(x) for x in counts_array.tolist()]

                if not counts:
                    continue

                bins = np.linspace(min_val, max_val, len(counts) + 1).tolist()
                hist_name = (
                    raw_name.decode("utf-8")
                    if isinstance(raw_name, (bytes, bytearray))
                    else str(raw_name)
                )

                results.append(
                    {
                        "step": int(step),
                        "global_step": (
                            int(global_step) if global_step is not None else None
                        ),
                        "name": hist_name,
                        "bins": bins,
                        "counts": counts,
                        "precision": precision,
                    }
                )

        return results
