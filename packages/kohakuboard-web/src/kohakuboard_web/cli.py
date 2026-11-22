"""CLI delegates invoked by `kobo open` and `kobo serve`."""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from kohakuboard_web.app import create_app


def _default_dir(arg: str | None) -> Path:
    if arg is not None:
        return Path(arg)
    return Path(os.environ.get("KOBO_DIR", "./kohakuboard"))


def run_open(argv: Sequence[str]) -> int:
    """Start local KohakuBoard web UI + API server on loopback."""
    parser = argparse.ArgumentParser(prog="kobo open", description="Open local KohakuBoard Web UI")
    parser.add_argument("dir", nargs="?", default=None, help="Base KohakuBoard directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=48889)
    parser.add_argument("--browser", action="store_true", help="Open browser automatically")
    parser.add_argument("--project", default=None, help="Initial project to open")
    args = parser.parse_args(argv)

    data_dir = _default_dir(args.dir)
    app = create_app(data_dir, enforce_loopback=args.host in ("127.0.0.1", "localhost", "::1"))

    url = f"http://{args.host}:{args.port}"
    if args.project:
        url = f"{url}/#/projects/{args.project}"
    if args.browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def run_serve(argv: Sequence[str]) -> int:
    """Start KohakuBoard HTTP server (multi-user remote auth arrives later)."""
    parser = argparse.ArgumentParser(prog="kobo serve", description="Serve KohakuBoard Web API")
    parser.add_argument("--data-dir", default=None, help="Base KohakuBoard directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=48889)
    args = parser.parse_args(argv)

    data_dir = _default_dir(args.data_dir)
    enforce = args.host in ("127.0.0.1", "localhost", "::1")
    app = create_app(data_dir, enforce_loopback=enforce)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
