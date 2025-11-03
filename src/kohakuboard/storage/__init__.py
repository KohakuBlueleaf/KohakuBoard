"""Storage backends for KohakuBoard

v0.2.0+: Only SQLite and Hybrid backends are supported.

Available backends:
- HybridStorage: Three-tier SQLite architecture (recommended)
  - KohakuVault KVault: K-V table with B+Tree index (media blobs)
  - KohakuVault ColumnVault: Blob-based columnar (metrics/histograms)
  - Standard SQLite: Traditional tables (metadata)
- SQLiteMetadataStorage: Pure standard SQLite storage (simple, reliable)

All three tiers use SQLite but with specialized implementations optimized
for their specific use cases.
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
