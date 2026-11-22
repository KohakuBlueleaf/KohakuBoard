"""Offline and live SWMR read-only `RunReader` backed by `kohakuboard._rs._RunReader`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from kohakuboard._rs import _RunReader, decompress_blob
from kohakuboard.errors import RunNotFoundError, translate_rs_error


@dataclass(slots=True)
class ScalarSlice:
    """Zero-copy-style NumPy slice of one scalar metric series."""

    total_rows: int
    start: int
    end: int
    steps: np.ndarray
    global_steps: np.ndarray
    ts_ms: np.ndarray
    values: np.ndarray


class RunReader:
    """Read-only handle to a KohakuBoard v2 run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        try:
            self._rs = _RunReader(str(self._run_dir))
        except Exception as exc:  # noqa: BLE001
            raise translate_rs_error(exc) from exc
        self._closed = False

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def info(self) -> dict[str, Any]:
        return json.loads(self._rs.info_json())

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self._rs.config_json())

    @property
    def env(self) -> dict[str, Any]:
        return json.loads(self._rs.env_json())

    @property
    def summary(self) -> dict[str, dict[str, Any]]:
        return json.loads(self._rs.summary_json())

    @property
    def catalog(self) -> dict[str, Any]:
        return json.loads(self._rs.catalog_json())

    @property
    def metric_definitions(self) -> list[dict[str, Any]]:
        return json.loads(self._rs.metric_definitions_json())

    @property
    def scalar_names(self) -> list[str]:
        return sorted(self.catalog.get("scalars", {}).keys())

    @property
    def histogram_names(self) -> list[str]:
        return sorted(self.catalog.get("histograms", {}).keys())

    @property
    def kde_names(self) -> list[str]:
        return sorted(self.catalog.get("kdes", {}).keys())

    @property
    def media_names(self) -> list[str]:
        return self._rs.media_names()

    @property
    def table_names(self) -> list[str]:
        return self._rs.table_names()

    @property
    def tensor_names(self) -> list[str]:
        return self._rs.tensor_names()

    def scalar_len(self, name: str) -> int:
        return self._rs.scalar_len(name)

    def scalar_slice(
        self,
        name: str,
        *,
        start: int = 0,
        end: int | None = None,
        count: int | None = None,
    ) -> ScalarSlice:
        """Read `[start .. end)` (or `count` rows) of scalar metric `name` as NumPy arrays."""
        if end is not None and count is not None:
            raise ValueError("Specify at most one of `end` or `count`")
        want = count if end is None else max(0, end - start)
        total, sb, gb, tb, vb = self._rs.scalar_slice_bytes(name, start, want)
        steps = np.frombuffer(sb, dtype="<i8")
        gsteps = np.frombuffer(gb, dtype="<i8")
        ts = np.frombuffer(tb, dtype="<i8")
        vals = np.frombuffer(vb, dtype="<f8")
        return ScalarSlice(
            total_rows=total,
            start=start,
            end=start + len(vals),
            steps=steps,
            global_steps=gsteps,
            ts_ms=ts,
            values=vals,
        )

    def scalar(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read the full scalar series `(steps, global_steps, ts_ms, values)` as NumPy arrays."""
        sl = self.scalar_slice(name, start=0, count=None)
        return sl.steps, sl.global_steps, sl.ts_ms, sl.values

    def scalar_newest_first_plan(
        self,
        name: str,
        *,
        first_chunk: int = 8192,
        max_chunk: int = 65536,
    ) -> list[tuple[int, int]]:
        """Return adaptive newest-first `(start, end)` ranges covering `[0 .. total_rows)`."""
        return self._rs.scalar_newest_first_plan(name, first_chunk, max_chunk)

    def scalar_range_frame(
        self,
        run_id: str,
        metric: str,
        start: int,
        end: int,
        *,
        x_dtype: Literal["f64", "i64"] = "f64",
    ) -> bytes:
        """Return an 8-byte-aligned binary chunk frame for `[start .. end)`."""
        return self._rs.scalar_range_frame(run_id, metric, start, end, x_dtype)

    def scalar_page_frame(
        self,
        run_id: str,
        metric: str,
        *,
        before: int | None = None,
        after: int | None = None,
        limit: int = 8192,
        x_dtype: Literal["f64", "i64"] = "f64",
    ) -> bytes:
        """Return a newest-first backward (`before`) or forward live-tail (`after`) binary frame."""
        return self._rs.scalar_page_frame(run_id, metric, before, after, limit, x_dtype)

    def histograms(
        self,
        name: str,
        *,
        start: int = 0,
        count: int | None = None,
    ) -> list[dict[str, Any]]:
        _, raw = self._rs.histogram_slice_json(name, start, count)
        return json.loads(raw)

    def kdes(
        self,
        name: str,
        *,
        start: int = 0,
        count: int | None = None,
    ) -> list[dict[str, Any]]:
        _, raw = self._rs.kde_slice_json(name, start, count)
        return json.loads(raw)

    def media(
        self,
        name: str,
        *,
        before_id: int | None = None,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        items = json.loads(self._rs.media_events_json(name, before_id, limit))
        for item in items:
            item["meta"] = json.loads(item.pop("meta_json", "{}"))
        return items

    def blob(self, key: str) -> bytes | None:
        return self._rs.blob(key)

    def tables(
        self,
        name: str,
        *,
        before_id: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        items = json.loads(self._rs.tables_json(name, before_id, limit))
        for item in items:
            item["columns"] = json.loads(item.pop("columns_json", "[]"))
            item["rows"] = json.loads(item.pop("rows_json", "[]"))
        return items

    def tensors(
        self,
        name: str,
        *,
        before_id: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        items = json.loads(self._rs.tensors_json(name, before_id, limit))
        for item in items:
            item["shape"] = json.loads(item.pop("shape_json", "[]"))
            item["meta"] = json.loads(item.pop("meta_json", "{}"))
        return items

    def read_tensor(self, record: dict[str, Any]) -> np.ndarray:
        """Load and reconstruct a logged NumPy tensor from a record returned by `tensors()`."""
        key = record["blob_key"]
        raw = self.blob(key)
        if raw is None:
            raise RunNotFoundError(f"Tensor blob '{key}' not found in blobs.db")
        if record.get("compressed"):
            raw = decompress_blob(raw)
        arr = np.frombuffer(raw, dtype=np.dtype(record["dtype"]))
        return arr.reshape(tuple(record["shape"])).copy()

    def artifacts(
        self,
        *,
        before_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        items = json.loads(self._rs.artifacts_json(before_id, limit))
        for item in items:
            item["meta"] = json.loads(item.pop("meta_json", "{}"))
        return items

    def alerts(
        self,
        *,
        before_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return json.loads(self._rs.alerts_json(before_id, limit))

    def logs(
        self,
        *,
        before_id: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return json.loads(self._rs.logs_tail_json(before_id, limit))

    def search_logs(self, query: str, *, limit: int = 50) -> list[dict[str, Any]]:
        pairs = json.loads(self._rs.logs_search_json(query, limit))
        return [{**entry, "score": score} for score, entry in pairs]

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._rs.close()

    def __enter__(self) -> RunReader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
