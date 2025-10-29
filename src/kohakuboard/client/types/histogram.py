"""Histogram type for KohakuBoard logging."""

from typing import Any, Dict, List, Optional, Union
import numpy as np


class Histogram:
    """
    Histogram data type for logging distributions.

    Can store either raw values or precomputed bins/counts.
    Users can choose to precompute bins before logging to reduce overhead.

    Args:
        values: Raw values (numpy array, torch tensor, or list) or None if precomputed
        num_bins: Number of bins for histogram (default: 64)
        precision: 'exact' or 'compact' precision mode
        bins: Precomputed bin edges (optional, if already computed)
        counts: Precomputed bin counts (optional, if already computed)

    Examples:
        # Log raw values (computed in writer process)
        hist = Histogram(gradients)
        board.log(grad_hist=hist)

        # Precompute for efficiency (computed immediately)
        hist = Histogram(gradients)
        hist.compute_bins()
        board.log(grad_hist=hist)

        # Already computed bins
        hist = Histogram(None, bins=bin_edges, counts=bin_counts)
        board.log(grad_hist=hist)
    """

    def __init__(
        self,
        values: Optional[Union[np.ndarray, List, Any]] = None,
        num_bins: int = 64,
        precision: str = "exact",
        bins: Optional[np.ndarray] = None,
        counts: Optional[np.ndarray] = None,
    ):
        """Initialize Histogram with either raw values or precomputed bins."""
        self.values = values
        self.num_bins = num_bins
        self.precision = precision
        self.bins = bins
        self.counts = counts
        self._is_computed = bins is not None and counts is not None

        # Validate inputs
        if values is None and not self._is_computed:
            raise ValueError("Must provide either values or both bins and counts")

        if self._is_computed and (bins is None or counts is None):
            raise ValueError("Both bins and counts must be provided if precomputed")

    def compute_bins(self) -> "Histogram":
        """
        Compute histogram bins and counts from raw values.

        This allows users to precompute histograms on the client side
        to reduce queue message size and writer process overhead.

        Returns:
            self (for method chaining)
        """
        if self._is_computed:
            return self  # Already computed

        if self.values is None:
            raise ValueError("Cannot compute bins: no raw values provided")

        # Convert to numpy array if needed
        values_array = self._to_numpy(self.values)

        if values_array.size == 0:
            # Empty array - create empty histogram
            self.bins = np.array([0.0, 1.0])
            self.counts = np.array([0])
            self._is_computed = True
            return self

        # Flatten array
        values_flat = values_array.flatten()

        # Remove NaN and inf values
        values_clean = values_flat[np.isfinite(values_flat)]

        if values_clean.size == 0:
            # All values were NaN/inf
            self.bins = np.array([0.0, 1.0])
            self.counts = np.array([0])
            self._is_computed = True
            return self

        # Compute histogram
        if self.precision == "compact":
            # Use fewer bins for compact mode
            actual_bins = min(self.num_bins, values_clean.size)
        else:
            actual_bins = self.num_bins

        counts, bin_edges = np.histogram(values_clean, bins=actual_bins)

        self.bins = bin_edges
        self.counts = counts
        self._is_computed = True

        # Can drop raw values after computing to save memory
        # (keeping them allows recomputing with different parameters if needed)

        return self

    def is_computed(self) -> bool:
        """Check if histogram bins/counts have been computed."""
        return self._is_computed

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize histogram for queue/storage.

        Returns dictionary with either:
        - Raw values (if not computed): {"values": [...], "num_bins": 64, "precision": "exact"}
        - Computed bins/counts: {"bins": [...], "counts": [...], "precision": "exact"}
        """
        if self._is_computed:
            return {
                "bins": (
                    self.bins.tolist()
                    if isinstance(self.bins, np.ndarray)
                    else self.bins
                ),
                "counts": (
                    self.counts.tolist()
                    if isinstance(self.counts, np.ndarray)
                    else self.counts
                ),
                "precision": self.precision,
                "computed": True,
            }
        else:
            # Convert values to list for serialization
            values_list = self._to_list(self.values)
            return {
                "values": values_list,
                "num_bins": self.num_bins,
                "precision": self.precision,
                "computed": False,
            }

    @staticmethod
    def _to_numpy(values: Any) -> np.ndarray:
        """Convert various value types to numpy array."""
        if isinstance(values, np.ndarray):
            return values

        # Check for PyTorch tensor
        if hasattr(values, "detach"):
            # PyTorch tensor
            return values.detach().cpu().numpy()

        # Try to convert to numpy
        try:
            return np.array(values)
        except Exception as e:
            raise TypeError(f"Cannot convert values to numpy array: {e}")

    @staticmethod
    def _to_list(values: Any) -> List:
        """Convert various value types to list for serialization."""
        if isinstance(values, list):
            return values

        if isinstance(values, np.ndarray):
            return values.flatten().tolist()

        # Check for PyTorch tensor
        if hasattr(values, "detach"):
            return values.detach().cpu().numpy().flatten().tolist()

        # Try to convert
        try:
            return np.array(values).flatten().tolist()
        except Exception as e:
            raise TypeError(f"Cannot convert values to list: {e}")

    def __repr__(self) -> str:
        """String representation."""
        if self._is_computed:
            return f"Histogram(bins={len(self.bins)-1}, computed=True, precision='{self.precision}')"
        else:
            size = len(self._to_list(self.values)) if self.values is not None else 0
            return f"Histogram(values={size}, num_bins={self.num_bins}, computed=False, precision='{self.precision}')"
