"""CLI entry points for `kobo tui` and `kobo watch`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from kohakuboard.tui.viewer import BoardTuiViewer


def run_tui(argv: list[str] | None = None) -> int:
    """Launch the standalone multi-run KohakuTUI viewer over a local or SSH data directory."""
    parser = argparse.ArgumentParser(
        prog="kobo tui",
        description="Interactive terminal viewer for KohakuBoard projects & runs.",
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default=os.environ.get("KOBO_DIR", "./kohakuboard"),
        help="KohakuBoard data directory (default: $KOBO_DIR or ./kohakuboard)",
    )
    parser.add_argument("-p", "--project", default=None, help="Initial project to focus")
    parser.add_argument("-r", "--run", default=None, help="Initial run_id to inspect")
    parser.add_argument("--theme", default="kohaku-dark", help="KohakuTUI theme name")
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Auto-refresh interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "tty", "plain"),
        default="auto",
        help="Terminal backend (default: auto)",
    )
    args = parser.parse_args(argv)
    target = Path(args.dir)
    if not target.exists():
        print(f"kobo tui: directory not found: {target}", file=sys.stderr)
        return 1
    if args.backend == "auto" and not getattr(sys.stdout, "isatty", lambda: False)():
        print(
            "kobo tui: interactive terminal (TTY) required; pass --backend plain for non-TTY.",
            file=sys.stderr,
        )
        return 2
    viewer = BoardTuiViewer(
        target,
        project=args.project,
        run_id=args.run,
        theme=args.theme,
        refresh_sec=args.refresh,
        backend=args.backend,
    )
    viewer.run()
    return 0


def run_watch(argv: list[str] | None = None) -> int:
    """Watch a specific run directory (or project/run_id) live in the terminal."""
    parser = argparse.ArgumentParser(
        prog="kobo watch",
        description="Live-tail a KohakuBoard run in the terminal (SSH-friendly).",
    )
    parser.add_argument(
        "target",
        help="Run directory path (containing run.db) or data directory",
    )
    parser.add_argument(
        "-p", "--project", default=None, help="Project name (if target is data_dir)"
    )
    parser.add_argument("-r", "--run", default=None, help="Run ID (if target is data_dir)")
    parser.add_argument("--theme", default="kohaku-dark", help="KohakuTUI theme name")
    parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="Live refresh interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "tty", "plain"),
        default="auto",
        help="Terminal backend (default: auto)",
    )
    args = parser.parse_args(argv)
    path = Path(args.target).resolve()
    if (path / "run.db").exists():
        # Target is directly a run directory: <data_dir>/<project>/<run_id[_annotation]>
        run_dir_name = path.name
        run_id = run_dir_name.split("_", 1)[0]
        project = path.parent.name
        data_dir = path.parent.parent
    else:
        data_dir = path
        project = args.project
        run_id = args.run
    if not data_dir.exists():
        print(f"kobo watch: path not found: {path}", file=sys.stderr)
        return 1
    if args.backend == "auto" and not getattr(sys.stdout, "isatty", lambda: False)():
        print(
            "kobo watch: interactive terminal (TTY) required; pass --backend plain for non-TTY.",
            file=sys.stderr,
        )
        return 2
    viewer = BoardTuiViewer(
        data_dir,
        project=project,
        run_id=run_id,
        theme=args.theme,
        refresh_sec=args.refresh,
        backend=args.backend,
    )
    viewer.run()
    return 0
