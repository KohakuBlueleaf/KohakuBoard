"""In-memory storage backend used for environments without writable disk.

Provides drop-in replacements for the HybridStorage components so the writer
process and sync worker can operate entirely from RAM (e.g. Google Colab).
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from kohakuboard.logger import get_logger


@dataclass
class _MediaRecord:
    id: int
    media_hash: str
    format: str
    step: int
    global_step: Optional[int]
    name: str
    caption: Optional[str]
    type: str
    width: Optional[int]
    height: Optional[int]
    size_bytes: Optional[int]
    timestamp: Optional[int]


class MemoryMetadataStorage:
    """In-memory metadata storage mirroring SQLiteMetadataStorage."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("MEMORY_STORAGE_META", drop=True)
        self.steps: Dict[int, dict[str, Any]] = {}
        self.media_records: Dict[int, _MediaRecord] = {}
        self.media_index: Dict[tuple[str, str], int] = {}
        self.tables: List[dict[str, Any]] = []
        self._next_media_id = 1

    def append_step_info(
        self, step: int, global_step: Optional[int], timestamp: Optional[int]
    ) -> None:
        self.steps[step] = {
            "step": step,
            "global_step": global_step,
            "timestamp": timestamp,
        }

    def _flush_steps(self) -> None:
        """Compatibility no-op."""

    def append_media(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        media_list: List[dict[str, Any]],
        caption: Optional[str] = None,
    ) -> List[int]:
        ids: List[int] = []

        for meta in media_list:
            key = (meta["media_hash"], meta["format"])
            existing_id = self.media_index.get(key)
            if existing_id is not None:
                ids.append(existing_id)
                continue

            media_id = self._next_media_id
            self._next_media_id += 1

            record = _MediaRecord(
                id=media_id,
                media_hash=meta["media_hash"],
                format=meta["format"],
                step=step,
                global_step=global_step,
                name=name,
                caption=caption,
                type=meta["type"],
                width=meta.get("width"),
                height=meta.get("height"),
                size_bytes=meta.get("size_bytes"),
                timestamp=self.steps.get(step, {}).get("timestamp"),
            )
            self.media_records[media_id] = record
            self.media_index[key] = media_id
            ids.append(media_id)

        return ids

    def _flush_media(self) -> None:
        """Compatibility no-op."""

    def append_table(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        table_data: dict[str, Any],
    ) -> None:
        self.tables.append(
            {
                "step": step,
                "global_step": global_step,
                "name": name,
                "columns": deepcopy(table_data.get("columns", [])),
                "column_types": deepcopy(table_data.get("column_types", [])),
                "rows": deepcopy(table_data.get("rows", [])),
            }
        )

    def _flush_tables(self) -> None:
        """Compatibility no-op."""

    def flush_all(self) -> None:
        """Compatibility no-op."""

    # ------------------------------------------------------------------ Reads
    def fetch_steps_range(self, start_step: int, end_step: int) -> list[dict[str, Any]]:
        return [
            info
            for step, info in sorted(self.steps.items())
            if start_step <= step <= end_step
        ]

    def fetch_media_range(self, start_step: int, end_step: int) -> list[dict[str, Any]]:
        records = []
        for record in self.media_records.values():
            if start_step <= record.step <= end_step:
                records.append(
                    {
                        "id": record.id,
                        "media_hash": record.media_hash,
                        "format": record.format,
                        "step": record.step,
                        "global_step": record.global_step,
                        "name": record.name,
                        "caption": record.caption,
                        "type": record.type,
                        "width": record.width,
                        "height": record.height,
                        "size_bytes": record.size_bytes,
                    }
                )
        records.sort(key=lambda r: r["step"])
        return records

    def fetch_tables_range(
        self, start_step: int, end_step: int
    ) -> list[dict[str, Any]]:
        return [
            deepcopy(table)
            for table in self.tables
            if start_step <= table["step"] <= end_step
        ]

    def fetch_media_info_by_hashes(
        self, hashes: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for record in self.media_records.values():
            if record.media_hash in hashes:
                lookup[record.media_hash] = {
                    "format": record.format,
                    "size_bytes": record.size_bytes,
                }
        return lookup

    def get_latest_step(self) -> Optional[dict[str, Any]]:
        if not self.steps:
            return None
        step = max(self.steps.keys())
        return self.steps[step]

    def get_latest_step_value(self) -> Optional[int]:
        if not self.steps:
            return None
        return max(self.steps.keys())

    def prune_up_to(self, max_step: int) -> None:
        """Drop metadata entries whose step <= max_step."""
        self.steps = {
            step: info for step, info in self.steps.items() if step > max_step
        }

        to_remove = [
            media_id
            for media_id, record in self.media_records.items()
            if record.step <= max_step
        ]
        for media_id in to_remove:
            record = self.media_records.pop(media_id)
            self.media_index.pop((record.media_hash, record.format), None)

        self.tables = [tbl for tbl in self.tables if tbl["step"] > max_step]

    def close(self) -> None:
        """Compatibility no-op."""


class MemoryMetricsStorage:
    """In-memory metrics storage."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("MEMORY_STORAGE_METRICS", drop=True)
        self.metrics: Dict[str, List[dict[str, Any]]] = defaultdict(list)

    def append_metrics(
        self,
        step: int,
        global_step: Optional[int],
        metrics: dict[str, Any],
        timestamp: Any,
    ) -> None:
        if hasattr(timestamp, "timestamp"):
            timestamp_ms = int(timestamp.timestamp() * 1000)
        else:
            timestamp_ms = int(timestamp * 1000) if timestamp is not None else None

        for metric_name, value in metrics.items():
            metric_list = self.metrics[metric_name]

            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    value = 0.0

            metric_list.append(
                {
                    "step": step,
                    "global_step": global_step,
                    "timestamp": timestamp_ms,
                    "value": float(value),
                }
            )

    def flush(self) -> None:
        """Compatibility no-op."""

    def collect_metrics_range(
        self, start_step: int, end_step: int
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for name, values in self.metrics.items():
            filtered = [
                {"step": entry["step"], "value": entry["value"]}
                for entry in values
                if start_step <= entry["step"] <= end_step
            ]
            if filtered:
                result[name] = filtered
        return result

    def close(self) -> None:
        """Compatibility no-op."""

    def prune_up_to(self, max_step: int) -> None:
        """Drop metric rows whose step <= max_step."""
        for metric_name in list(self.metrics.keys()):
            rows = self.metrics[metric_name]
            filtered = [row for row in rows if row["step"] > max_step]
            if filtered:
                self.metrics[metric_name] = filtered
            else:
                self.metrics.pop(metric_name, None)


class MemoryHistogramStorage:
    """In-memory histogram storage with same contract as ColumnVaultHistogramStorage."""

    def __init__(self, num_bins: int = 64, logger=None):
        self.logger = logger or get_logger("MEMORY_STORAGE_HIST", drop=True)
        self.num_bins = num_bins
        self.histograms: Dict[str, List[dict[str, Any]]] = defaultdict(list)

    def append_histogram(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        values: Optional[list[float]] = None,
        num_bins: int = 64,
        precision: str = "compact",
        bins: Optional[list[float]] = None,
        counts: Optional[list[int]] = None,
    ) -> None:
        if bins is not None and counts is not None:
            final_bins = list(bins)
            final_counts = [int(x) for x in counts]
            if precision == "compact":
                final_precision = "u8"
            elif precision == "exact":
                final_precision = "i32"
            else:
                final_precision = precision
        elif values is not None:
            arr = np.array(values, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return

            if precision == "compact":
                actual_bins = min(num_bins, arr.size)
            else:
                actual_bins = num_bins

            p1 = float(np.percentile(arr, 1))
            p99 = float(np.percentile(arr, 99))

            if p99 - p1 < 1e-6:
                p1 = float(arr.min())
                p99 = float(arr.max())
                if p99 - p1 < 1e-6:
                    p1 -= 0.5
                    p99 += 0.5

            counts_array, _ = np.histogram(arr, bins=actual_bins, range=(p1, p99))

            if precision == "compact":
                max_count = counts_array.max()
                if max_count > 0:
                    final_counts = (
                        (counts_array / max_count * 255).astype(np.uint8).tolist()
                    )
                else:
                    final_counts = counts_array.astype(np.uint8).tolist()
                final_precision = "u8"
            else:
                final_counts = counts_array.astype(np.int32).tolist()
                final_precision = "i32"

            final_bins = np.linspace(p1, p99, len(final_counts) + 1).tolist()
        else:
            return

        self.histograms[name].append(
            {
                "step": step,
                "global_step": global_step,
                "name": name,
                "bins": final_bins,
                "counts": final_counts,
                "precision": final_precision,
            }
        )

    def flush(self) -> None:
        """Compatibility no-op."""

    def collect_histograms_range(
        self, start_step: int, end_step: int
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for name, entries in self.histograms.items():
            for entry in entries:
                if start_step <= entry["step"] <= end_step:
                    results.append(deepcopy(entry))
        return results

    def close(self) -> None:
        """Compatibility no-op."""

    def prune_up_to(self, max_step: int) -> None:
        """Drop histogram rows whose step <= max_step."""
        for name in list(self.histograms.keys()):
            entries = self.histograms[name]
            filtered = [entry for entry in entries if entry["step"] > max_step]
            if filtered:
                self.histograms[name] = filtered
            else:
                self.histograms.pop(name, None)


class MemoryTensorStorage:
    """In-memory tensor and KDE storage."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("MEMORY_STORAGE_TENSOR", drop=True)
        self.tensors: Dict[str, List[dict[str, Any]]] = defaultdict(list)
        self.kde_logs: Dict[str, List[dict[str, Any]]] = defaultdict(list)

    def append_tensor(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        payload: bytes,
        tensor_meta: dict[str, Any],
    ) -> None:
        self.tensors[name].append(
            {
                "step": step,
                "global_step": global_step,
                "payload": payload,
                "tensor_meta": tensor_meta,
            }
        )

    def append_kernel_density(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        payload: bytes,
        kde_meta: dict[str, Any],
    ) -> None:
        self.kde_logs[name].append(
            {
                "step": step,
                "global_step": global_step,
                "payload": payload,
                "kde_meta": kde_meta,
            }
        )

    def close(self) -> None:
        self.tensors.clear()
        self.kde_logs.clear()

    def prune_up_to(self, max_step: int) -> None:
        for name in list(self.tensors.keys()):
            entries = [e for e in self.tensors[name] if e["step"] > max_step]
            if entries:
                self.tensors[name] = entries
            else:
                self.tensors.pop(name, None)

        for name in list(self.kde_logs.keys()):
            entries = [e for e in self.kde_logs[name] if e["step"] > max_step]
            if entries:
                self.kde_logs[name] = entries
            else:
                self.kde_logs.pop(name, None)


class MemoryHybridStorage:
    """Hybrid storage variant backed entirely by in-memory structures."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("MEMORY_STORAGE", drop=True)
        self.metadata_storage = MemoryMetadataStorage(logger=self.logger)
        self.metrics_storage = MemoryMetricsStorage(logger=self.logger)
        self.histogram_storage = MemoryHistogramStorage(logger=self.logger)
        self.tensor_storage = MemoryTensorStorage(logger=self.logger)

    def append_metrics(
        self,
        step: int,
        global_step: Optional[int],
        metrics: dict[str, Any],
        timestamp: Any,
    ) -> None:
        if hasattr(timestamp, "timestamp"):
            timestamp_ms = int(timestamp.timestamp() * 1000)
        else:
            timestamp_ms = int(timestamp * 1000) if timestamp is not None else None

        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)
        self.metrics_storage.append_metrics(step, global_step, metrics, timestamp)

    def append_media(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        media_list: list[dict[str, Any]],
        caption: Optional[str] = None,
    ) -> list[int]:
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)
        return self.metadata_storage.append_media(
            step, global_step, name, media_list, caption
        )

    def append_table(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        table_data: dict[str, Any],
    ) -> None:
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)
        self.metadata_storage.append_table(step, global_step, name, table_data)

    def append_histogram(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        values: Optional[list[float]] = None,
        num_bins: int = 64,
        precision: str = "compact",
        bins: Optional[list[float]] = None,
        counts: Optional[list[int]] = None,
    ) -> None:
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)
        self.histogram_storage.append_histogram(
            step,
            global_step,
            name,
            values,
            num_bins,
            precision,
            bins=bins,
            counts=counts,
        )

    def append_tensor(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        payload: bytes,
        tensor_meta: dict[str, Any],
    ) -> None:
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)
        self.tensor_storage.append_tensor(step, global_step, name, payload, tensor_meta)

    def append_kernel_density(
        self,
        step: int,
        global_step: Optional[int],
        name: str,
        payload: bytes,
        kde_meta: dict[str, Any],
    ) -> None:
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.metadata_storage.append_step_info(step, global_step, timestamp_ms)
        self.tensor_storage.append_kernel_density(
            step, global_step, name, payload, kde_meta
        )

    def flush_metrics(self) -> None:
        self.metrics_storage.flush()

    def flush_media(self) -> None:
        self.metadata_storage._flush_media()

    def flush_tables(self) -> None:
        self.metadata_storage._flush_tables()

    def flush_histograms(self) -> None:
        self.histogram_storage.flush()

    def flush_all(self) -> None:
        """No-op flush since everything lives in memory."""
        self.logger.debug("MemoryHybridStorage flush_all called (no-op)")

    def get_latest_step(self) -> Optional[dict[str, Any]]:
        return self.metadata_storage.get_latest_step()

    def close(self) -> None:
        self.metrics_storage.close()
        self.metadata_storage.close()
        self.histogram_storage.close()
        self.tensor_storage.close()

    def prune_up_to(self, max_step: int) -> None:
        """Remove all entries whose step <= max_step."""
        self.metadata_storage.prune_up_to(max_step)
        self.metrics_storage.prune_up_to(max_step)
        self.histogram_storage.prune_up_to(max_step)
        self.tensor_storage.prune_up_to(max_step)
