"""Kernel density logging helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _flatten_array(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Convert supported tensor inputs to a contiguous 1D numpy array."""
    if isinstance(values, np.ndarray):
        arr = values
    elif hasattr(values, "detach"):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


@dataclass(slots=True)
class KernelDensity:
    """Compact representation of a 1D kernel density estimate.

    Supports both raw-value logging (auto-computed KDE) and precomputed grid/density
    pairs. When raw values are provided, the KDE is computed lazily when arrays are
    requested or metadata is summarised.

    Args:
        raw_values: Optional raw samples to build the KDE from.
        grid: Optional precomputed grid positions (ignored when raw_values provided).
        density: Optional precomputed density values (ignored when raw_values provided).
        kernel: Kernel name (currently only 'gaussian' is supported).
        bandwidth: Bandwidth setting. Float for explicit value or "auto"/None to use
            Scott's rule-of-thumb.
        num_points: Number of evaluation points when computing from raw samples.
        percentile_min: Lower percentile used to determine display range when
            range_min is not provided.
        percentile_max: Upper percentile used to determine display range when
            range_max is not provided.
        sample_count: Optional explicit sample count (auto-filled when raw_values provided).
        range_min: Optional explicit display range minimum (otherwise derived from data).
        range_max: Optional explicit display range maximum (otherwise derived from data).
        metadata: Optional dictionary stored alongside KDE metadata.
    """

    raw_values: Any | None = None
    grid: Any | None = None
    density: Any | None = None
    kernel: str = "gaussian"
    bandwidth: float | str | None = None
    num_points: int = 256
    percentile_min: float = 1.0
    percentile_max: float = 99.0
    sample_count: int | None = None
    range_min: float | None = None
    range_max: float | None = None
    metadata: dict[str, Any] | None = None
    _is_computed: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.raw_values is None and (self.grid is None or self.density is None):
            raise ValueError(
                "KernelDensity requires either raw_values or both grid and density."
            )

        if self.kernel != "gaussian":
            raise ValueError("Currently only the 'gaussian' kernel is supported.")

        if self.raw_values is None:
            # Precomputed path – mark as computed after validation
            grid_arr = _flatten_array(self.grid, dtype=np.float32)
            density_arr = _flatten_array(self.density, dtype=np.float32)
            if grid_arr.shape != density_arr.shape:
                raise ValueError(
                    f"KDE grid/density shape mismatch: {grid_arr.shape} vs {density_arr.shape}"
                )
            self.grid = grid_arr
            self.density = density_arr
            self._is_computed = True
            if self.sample_count is None:
                self.sample_count = grid_arr.size
        else:
            if self.grid is not None or self.density is not None:
                raise ValueError(
                    "Provide either raw_values or (grid, density), but not both."
                )
            if self.percentile_min >= self.percentile_max:
                raise ValueError(
                    "percentile_min must be strictly less than percentile_max."
                )

    # ------------------------------------------------------------------ Computation helpers
    def ensure_computed(self) -> None:
        """Compute KDE from raw samples if needed."""
        if self._is_computed:
            return

        values = _flatten_array(self.raw_values, dtype=np.float32)
        if values.size == 0:
            raise ValueError(
                "KernelDensity raw_values must contain at least one sample."
            )

        self.sample_count = int(values.size)

        if self.bandwidth is None or isinstance(self.bandwidth, str):
            # Scott's rule-of-thumb
            std = float(np.std(values))
            bandwidth = 1.06 * std * (values.size ** (-1.0 / 5.0))
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                bandwidth = max(std, 1e-3) or 1e-3
        else:
            bandwidth = float(self.bandwidth)

        self.bandwidth = bandwidth

        # Determine display range
        if self.range_min is None or self.range_max is None:
            p_low = np.percentile(values, self.percentile_min)
            p_high = np.percentile(values, self.percentile_max)
            p_low = float(p_low)
            p_high = float(p_high)
            if p_low == p_high:
                p_low -= 0.5
                p_high += 0.5
            self.range_min = p_low
            self.range_max = p_high

        if self.range_min == self.range_max:
            self.range_min -= 0.5
            self.range_max += 0.5

        grid = np.linspace(
            float(self.range_min),
            float(self.range_max),
            int(self.num_points),
            dtype=np.float32,
        )

        # Gaussian KDE evaluation
        diff = (grid[:, None] - values[None, :]) / bandwidth
        density = (
            np.exp(-0.5 * diff**2).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))
        ).astype(np.float32)

        self.grid = grid
        self.density = density
        self._is_computed = True

        # Clear raw values to free memory (optional)
        self.raw_values = None

    # ------------------------------------------------------------------ Public helpers
    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return contiguous float32 arrays for grid and density."""
        self.ensure_computed()
        return self.grid, self.density

    def summary(self) -> dict[str, Any]:
        """Return metadata summary for transport."""
        grid_array, density_array = self.to_arrays()
        range_min = (
            float(self.range_min)
            if self.range_min is not None
            else float(grid_array.min())
        )
        range_max = (
            float(self.range_max)
            if self.range_max is not None
            else float(grid_array.max())
        )
        bandwidth = self.bandwidth
        if isinstance(bandwidth, (float, int, np.floating)):
            bandwidth_value = float(bandwidth)
        else:
            bandwidth_value = bandwidth

        return {
            "kernel": self.kernel,
            "bandwidth": bandwidth_value,
            "sample_count": (
                int(self.sample_count) if self.sample_count is not None else None
            ),
            "range_min": range_min,
            "range_max": range_max,
            "metadata": self.metadata or {},
            "num_points": int(grid_array.size),
        }
