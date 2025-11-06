"""Tensor logging helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def _to_numpy(array_like: Any) -> np.ndarray:
    """Convert supported tensor inputs to contiguous numpy array."""
    if isinstance(array_like, np.ndarray):
        result = array_like
    elif hasattr(array_like, "detach"):
        result = array_like.detach().cpu().numpy()  # PyTorch tensor
    else:
        result = np.asarray(array_like)

    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)
    return result


@dataclass(slots=True)
class TensorLog:
    """Lightweight wrapper for logging dense tensors.

    Args:
        values: Array-like object representing the tensor payload.
        precision: Optional dtype override used when serialising (defaults to input dtype).
        metadata: Optional mapping of additional metadata to persist alongside the tensor.
    """

    values: Any
    precision: Any | None = None
    metadata: dict[str, Any] | None = None

    def to_numpy(self) -> np.ndarray:
        """Return values as contiguous numpy array, applying precision override."""
        array = _to_numpy(self.values)
        if self.precision is not None:
            array = array.astype(self.precision, copy=False)
        return array

    def summary(self) -> dict[str, Any]:
        """Return shape/dtype summary for queue transport."""
        array = _to_numpy(self.values)
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "metadata": self.metadata or {},
        }
