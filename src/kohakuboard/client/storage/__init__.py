"""Storage backends for KohakuBoard

v0.2.0+: Only SQLite and Hybrid backends are supported.

Available backends:
- HybridStorage: Lance (metrics) + SQLite (metadata) + Histograms (recommended)
- SQLiteMetadataStorage: Pure SQLite storage (simple, reliable)
"""

from kohakuboard.client.storage.histogram import HistogramStorage
from kohakuboard.client.storage.hybrid import HybridStorage
from kohakuboard.client.storage.lance import LanceMetricsStorage
from kohakuboard.client.storage.sqlite import SQLiteMetadataStorage

__all__ = [
    "HybridStorage",
    "SQLiteMetadataStorage",
    "HistogramStorage",
    "LanceMetricsStorage",
]
