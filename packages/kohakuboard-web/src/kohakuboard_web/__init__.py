"""KohakuBoard Web — unified FastAPI server and bundled SPA."""

from __future__ import annotations

from kohakuboard_web.app import create_app
from kohakuboard_web.service import BoardService

__version__ = "2.0.0"

__all__ = ["BoardService", "__version__", "create_app"]
