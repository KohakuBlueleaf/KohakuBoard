"""Lightweight rich payload dataclasses logged via `Board.log(...)`."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from kohakuboard._rs import blake3_hex
from kohakuboard.errors import ConfigError

_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
_VIDEO_EXTS = {"mp4", "webm", "mov", "mkv"}
_AUDIO_EXTS = {"wav", "mp3", "ogg", "flac"}


def _to_numpy(obj: Any) -> np.ndarray:
    """Convert numpy arrays, duck-typed torch tensors, or sequences to an ndarray."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return np.asarray(obj.detach().cpu().numpy())
    if hasattr(obj, "convert") and hasattr(obj, "size") and hasattr(obj, "mode"):
        img = obj if obj.mode in ("L", "RGB", "RGBA") else obj.convert("RGB")
        return np.asarray(img)
    return np.asarray(obj)


def _normalize_image_array(arr: np.ndarray) -> tuple[bytes, int, int, int]:
    """Normalize HW / HWC / CHW uint8 or float image arrays into contiguous HWC u8 bytes."""
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        # Convert CHW -> HWC when axis 0 is a standard channel count and axis 2 is not.
        if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ConfigError(f"Image array must be 2D or 3D, got shape {arr.shape}")

    height, width, channels = arr.shape
    if channels not in (1, 3, 4):
        raise ConfigError(f"Image must have 1, 3, or 4 channels, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr_f = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        lo = float(arr_f.min()) if arr_f.size else 0.0
        hi = float(arr_f.max()) if arr_f.size else 1.0
        if lo < 0.0 and hi <= 1.01 and lo >= -1.01:
            arr_f = (arr_f + 1.0) * 0.5
        if hi <= 1.01:
            arr_f = arr_f * 255.0
        u8 = np.clip(np.rint(arr_f), 0, 255).astype(np.uint8)
    elif arr.dtype == np.uint8:
        u8 = arr
    else:
        u8 = np.clip(arr, 0, 255).astype(np.uint8)

    u8 = np.ascontiguousarray(u8)
    return u8.tobytes(), int(width), int(height), int(channels)


def _detect_media_kind(ext: str) -> str:
    clean = ext.lstrip(".").lower()
    if clean in _IMAGE_EXTS:
        return "image"
    if clean in _VIDEO_EXTS:
        return "video"
    if clean in _AUDIO_EXTS:
        return "audio"
    return "binary"


@dataclass(slots=True)
class Media:
    """Image, video, or audio media item encoded/hashed off-GIL in Rust."""

    data: Any
    kind: Literal["auto", "image", "video", "audio"] = "auto"
    caption: str = ""
    format: str = "auto"
    quality: int = 90
    fps: float | None = None
    sample_rate: int | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def prepare(self) -> tuple[str, Any]:
        """Return either `('pixels', ...)` or `('encoded', ...)` ready for `_LoggerEngine`."""
        meta = dict(self.metadata)
        if self.fps is not None:
            meta.setdefault("fps", self.fps)
        if self.sample_rate is not None:
            meta.setdefault("sample_rate", self.sample_rate)
        meta_json = json.dumps(meta, default=str)

        if isinstance(self.data, (str, Path)):
            path = Path(self.data)
            raw = path.read_bytes()
            ext = path.suffix.lstrip(".").lower() or (
                "png" if self.format == "auto" else self.format.lower()
            )
            media_type = _detect_media_kind(ext) if self.kind == "auto" else self.kind
            return ("encoded", (media_type, raw, ext, self.duration_ms, self.caption, meta_json))

        if isinstance(self.data, (bytes, bytearray, memoryview)):
            raw = bytes(self.data)
            ext = "png" if self.format == "auto" else self.format.lstrip(".").lower()
            media_type = _detect_media_kind(ext) if self.kind == "auto" else self.kind
            return ("encoded", (media_type, raw, ext, self.duration_ms, self.caption, meta_json))

        arr = _to_numpy(self.data)
        pixels, width, height, channels = _normalize_image_array(arr)
        fmt = "png" if self.format == "auto" else self.format.lower()
        return (
            "pixels",
            (pixels, width, height, channels, fmt, int(self.quality), self.caption, meta_json),
        )


@dataclass(slots=True)
class Table:
    """Tabular snapshot supporting dicts, `(columns, rows)`, DataFrames, and `Media` cells."""

    columns: list[str]
    rows: list[list[Any]]

    def __init__(
        self,
        data: Any = None,
        *,
        columns: Sequence[str] | None = None,
        rows: Sequence[Sequence[Any]] | None = None,
    ) -> None:
        if data is not None and (columns is not None or rows is not None):
            raise ConfigError("Pass either `data` or `columns`/`rows` to Table, not both")

        if data is not None:
            if hasattr(data, "to_dicts") and hasattr(data, "columns"):
                cols = [str(c) for c in data.columns]
                recs = data.to_dicts()
                self.columns = cols
                self.rows = [[r.get(c) for c in cols] for r in recs]
                return
            if hasattr(data, "to_dict") and hasattr(data, "columns"):
                cols = [str(c) for c in data.columns]
                recs = data.to_dict(orient="records")
                self.columns = cols
                self.rows = [[r.get(c) for c in cols] for r in recs]
                return
            if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
                if len(data) == 0:
                    self.columns = []
                    self.rows = []
                    return
                if isinstance(data[0], Mapping):
                    cols: list[str] = []
                    seen: set[str] = set()
                    for item in data:
                        for k in item:
                            ks = str(k)
                            if ks not in seen:
                                seen.add(ks)
                                cols.append(ks)
                    self.columns = cols
                    self.rows = [[item.get(c) for c in cols] for item in data]
                    return
            raise ConfigError("Unsupported `data` input for Table")

        self.columns = [str(c) for c in (columns or [])]
        self.rows = [list(r) for r in (rows or [])]

    def serialize(
        self,
        stage_media: Callable[[Media], dict[str, Any]],
    ) -> tuple[str, int, str]:
        """Serialize table schema and rows to JSON, staging any nested `Media` cells."""
        out_rows: list[list[Any]] = []
        for row in self.rows:
            conv_row: list[Any] = []
            for cell in row:
                if isinstance(cell, Media):
                    conv_row.append(stage_media(cell))
                elif isinstance(cell, (float, np.floating)):
                    fval = float(cell)
                    conv_row.append(fval if math.isfinite(fval) else None)
                elif isinstance(cell, np.integer):
                    conv_row.append(cell.item())
                elif isinstance(cell, Path):
                    conv_row.append(str(cell))
                else:
                    conv_row.append(cell)
            out_rows.append(conv_row)
        return (
            json.dumps(self.columns, allow_nan=False),
            len(out_rows),
            json.dumps(out_rows, default=str, allow_nan=False),
        )


@dataclass(slots=True)
class Histogram:
    """Distribution histogram computed off-GIL in Rust or passed precomputed."""

    values: Any = None
    bins: int = 64
    counts: Sequence[int] | None = None
    edges: Sequence[float] | None = None
    precision: Literal["exact", "compact"] = "exact"
    range: tuple[float, float] | None = None
    percentiles: tuple[float, float] = (1.0, 99.0)

    def __post_init__(self) -> None:
        has_raw = self.values is not None
        has_pre = self.counts is not None or self.edges is not None
        if has_raw and has_pre:
            raise ConfigError("Histogram accepts either `values` or `(edges, counts)`, not both")
        if not has_raw and not has_pre:
            raise ConfigError("Histogram requires either `values` or `(edges, counts)`")
        if has_pre and (self.counts is None or self.edges is None):
            raise ConfigError("Precomputed Histogram requires both `edges` and `counts`")
        if self.precision not in ("exact", "compact"):
            raise ConfigError(
                f"Histogram precision must be 'exact' or 'compact', got {self.precision!r}"
            )

    def to_raw_f32_bytes(self) -> bytes:
        arr = np.ascontiguousarray(_to_numpy(self.values).ravel(), dtype="<f4")
        return arr.tobytes()


@dataclass(slots=True)
class KernelDensity:
    """1D Gaussian KDE evaluated off-GIL in Rust (exact or FFT-binned) or passed precomputed."""

    values: Any = None
    grid: Sequence[float] | None = None
    density: Sequence[float] | None = None
    num_points: int = 256
    bandwidth: float | Literal["auto"] = "auto"
    approximate: bool = True
    range: tuple[float, float] | None = None
    percentiles: tuple[float, float] = (1.0, 99.0)
    sample_count: int | None = None

    def __post_init__(self) -> None:
        has_raw = self.values is not None
        has_pre = self.grid is not None or self.density is not None
        if has_raw and has_pre:
            raise ConfigError(
                "KernelDensity accepts either `values` or `(grid, density)`, not both"
            )
        if not has_raw and not has_pre:
            raise ConfigError("KernelDensity requires either `values` or `(grid, density)`")
        if has_pre and (self.grid is None or self.density is None):
            raise ConfigError("Precomputed KernelDensity requires both `grid` and `density`")

    def to_raw_f32_bytes(self) -> bytes:
        arr = np.ascontiguousarray(_to_numpy(self.values).ravel(), dtype="<f4")
        return arr.tobytes()


@dataclass(slots=True)
class TensorLog:
    """Arbitrary N-D tensor payload stored in `blobs.db` with optional gzip compression."""

    tensor: Any
    dtype: str | None = None
    compress: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def prepare(self) -> tuple[bytes, str, str, bool, str]:
        arr = _to_numpy(self.tensor)
        if self.dtype is not None:
            arr = arr.astype(np.dtype(self.dtype))
        arr = np.ascontiguousarray(arr)
        return (
            arr.tobytes(),
            str(arr.dtype),
            json.dumps(list(arr.shape)),
            bool(self.compress),
            json.dumps(self.metadata, default=str),
        )


@dataclass(slots=True)
class Artifact:
    """Versioned checkpoint, dataset, or model artifact reference."""

    name: str
    kind: str = "checkpoint"
    path: str | Path | None = None
    uri: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def prepare(self) -> tuple[str, str, str, str | None, int | None, str]:
        uri = self.uri
        digest = self.digest
        size_bytes = self.size_bytes
        if self.path is not None:
            p = Path(self.path)
            if uri is None:
                uri = str(p.resolve()) if p.exists() else str(p)
            if p.is_file():
                data = p.read_bytes()
                if size_bytes is None:
                    size_bytes = len(data)
                if digest is None:
                    digest = blake3_hex(data)
        if uri is None:
            raise ConfigError("Artifact requires either `path` or `uri`")
        return (
            self.name,
            self.kind,
            uri,
            digest,
            size_bytes,
            json.dumps(self.metadata, default=str),
        )
