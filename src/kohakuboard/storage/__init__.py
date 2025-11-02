"""Storage backends for KohakuBoard

v0.2.0+: Only SQLite and Hybrid backends are supported.

Available backends:
- HybridStorage: ColumnVault (metrics) + SQLite (metadata) + Histograms (recommended)
- SQLiteMetadataStorage: Pure SQLite storage (simple, reliable)
"""

from kohakuboard.storage.columnar import ColumnVaultMetricsStorage
from kohakuboard.storage.columnar_histogram import ColumnVaultHistogramStorage
from kohakuboard.storage.hybrid import HybridStorage
from kohakuboard.storage.sqlite import SQLiteMetadataStorage

__all__ = [
    "HybridStorage",
    "SQLiteMetadataStorage",
    "ColumnVaultMetricsStorage",
    "ColumnVaultHistogramStorage",
]
