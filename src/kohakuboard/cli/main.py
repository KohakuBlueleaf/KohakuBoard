"""`kobo` CLI entry point (stdlib argparse, zero extra runtime dependencies)."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kohakuboard._rs import SCHEMA_VERSION
from kohakuboard._rs import version as rs_version
from kohakuboard.errors import KohakuBoardError
from kohakuboard.reader import ProjectReader, RunReader


def _default_dir(path_arg: str | None) -> Path:
    if path_arg is not None:
        return Path(path_arg)
    return Path(os.environ.get("KOBO_DIR", "./kohakuboard"))


def _format_bytes(size: int) -> str:
    val = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if val < 1024.0 or unit == "GiB":
            return f"{val:.1f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= 1024.0
    return f"{size} B"


def _cmd_ls(args: argparse.Namespace) -> int:
    data_dir = _default_dir(args.dir)
    if not data_dir.exists():
        if args.json:
            print("[]")
        else:
            print(f"Directory does not exist: {data_dir}")
        return 0

    with ProjectReader(data_dir, project=args.project) as reader:
        runs = reader.list_runs(
            project=args.project,
            state=args.state,
            group=args.group,
            search=args.search,
            sort=args.sort,
            limit=args.limit,
        )

    if args.json:
        payload = [
            {
                "project": r.project,
                "run_id": r.run_id,
                "dir_name": r.dir_name,
                "name": r.name,
                "state": r.state,
                "group": r.group,
                "tags": r.tags,
                "last_step": r.last_step,
                "last_global_step": r.last_global_step,
                "path": str(r.path),
                "summary": r.summary,
            }
            for r in runs
        ]
        print(json.dumps(payload, indent=2))
        return 0

    if not runs:
        print(f"No runs found in {data_dir}")
        return 0

    header = f"{'PROJECT':<16} {'RUN_ID':<12} {'STATE':<12} {'STEP':>8}  {'NAME'}"
    print(header)
    print("-" * len(header))
    for r in runs:
        step_str = str(r.last_step) if r.last_step >= 0 else "-"
        print(f"{r.project:<16} {r.run_id:<12} {r.state:<12} {step_str:>8}  {r.name}")
    return 0


def _fmt_num(val: Any) -> str:
    if isinstance(val, (int, float)):
        return f"{val:.6g}"
    return "NaN/Inf"


def _cmd_info(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    with RunReader(run_dir) as reader:
        info = reader.info
        config = reader.config
        summary = reader.summary
        catalog = reader.catalog
        media_names = reader.media_names
        table_names = reader.table_names
        tensor_names = reader.tensor_names

    run_db = run_dir / "run.db"
    blobs_db = run_dir / "blobs.db"
    run_db_size = run_db.stat().st_size if run_db.exists() else 0
    blobs_db_size = blobs_db.stat().st_size if blobs_db.exists() else 0

    if args.json:
        print(
            json.dumps(
                {
                    "info": info,
                    "config": config,
                    "summary": summary,
                    "catalog": catalog,
                    "media_names": media_names,
                    "table_names": table_names,
                    "tensor_names": tensor_names,
                    "storage": {
                        "run_db_bytes": run_db_size,
                        "blobs_db_bytes": blobs_db_size,
                    },
                },
                indent=2,
            )
        )
        return 0

    print(f"Run:      {info['name']} ({info['run_id']})")
    print(f"Project:  {info['project']}")
    print(f"State:    {info['state']} (exit_code={info.get('exit_code')})")
    print(f"Steps:    step={info['last_step']}, global_step={info['last_global_step']}")
    if info.get("group"):
        print(f"Group:    {info['group']}")
    if info.get("tags"):
        print(f"Tags:     {', '.join(info['tags'])}")
    print(f"Storage:  run.db={_format_bytes(run_db_size)}, blobs.db={_format_bytes(blobs_db_size)}")
    print(
        f"Data:     {len(catalog.get('scalars', {}))} scalars, "
        f"{len(catalog.get('histograms', {}))} histograms, "
        f"{len(catalog.get('kdes', {}))} kdes, "
        f"{len(media_names)} media, {len(table_names)} tables, {len(tensor_names)} tensors"
    )
    if config:
        print(f"Config:   {json.dumps(config, sort_keys=True)}")
    if summary:
        print("Summary:")
        for key in sorted(summary):
            s = summary[key]
            print(
                f"  {key:<28} last={_fmt_num(s.get('last'))}  "
                f"min={_fmt_num(s.get('min'))}  max={_fmt_num(s.get('max'))}  n={s['count']}"
            )
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    with RunReader(run_dir) as reader:
        if args.search:
            entries = reader.search_logs(args.search, limit=args.tail)
        else:
            entries = list(reversed(reader.logs(limit=args.tail)))

    if args.json:
        print(json.dumps(entries, indent=2))
        return 0

    for entry in entries:
        prefix = f"[{entry['stream']}|step={entry['step']}]"
        if "score" in entry:
            prefix = f"[score={entry['score']:.2f}|{entry['stream']}|step={entry['step']}]"
        print(f"{prefix} {entry['text']}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with RunReader(run_dir) as reader:
        info = reader.info
        config = reader.config
        metrics = args.metrics if args.metrics else reader.scalar_names
        series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
            m: reader.scalar(m) for m in metrics
        }

    if args.format == "csv":
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "step", "global_step", "ts_ms", "value"])
            for metric, (steps, gsteps, ts, vals) in series.items():
                for s, g, t, v in zip(steps, gsteps, ts, vals):
                    writer.writerow([metric, int(s), int(g), int(t), float(v)])
    elif args.format == "json":
        payload: dict[str, Any] = {
            "info": info,
            "config": config,
            "scalars": {
                m: {
                    "step": s.tolist(),
                    "global_step": g.tolist(),
                    "ts_ms": t.tolist(),
                    "value": [None if np.isnan(x) else float(x) for x in v],
                }
                for m, (s, g, t, v) in series.items()
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif args.format == "npz":
        arrays: dict[str, np.ndarray] = {}
        for metric, (s, g, t, v) in series.items():
            safe = metric.replace("/", "__")
            arrays[f"{safe}__step"] = s
            arrays[f"{safe}__global_step"] = g
            arrays[f"{safe}__ts_ms"] = t
            arrays[f"{safe}__value"] = v
        np.savez_compressed(out_path, **arrays)
    else:
        print(f"Unsupported export format: {args.format}", file=sys.stderr)
        return 2

    print(f"Exported {len(series)} scalar series to {out_path}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    data_dir = _default_dir(args.dir)
    print(f"kohakuboard._rs: v{rs_version()} (schema v{SCHEMA_VERSION})")
    print(f"data_dir:        {data_dir.resolve()}")
    if not data_dir.exists():
        print("status:          data directory does not exist yet (OK)")
        return 0

    with ProjectReader(data_dir) as reader:
        updated = reader.refresh(force_rebuild=args.reindex)
        projects = reader.list_projects(refresh=False)
        runs = reader.list_runs(refresh=False)

    print(
        f"catalog:         indexed {len(runs)} runs across {len(projects)} projects "
        f"(updated={updated})"
    )
    corrupt = 0
    for r in runs:
        try:
            with RunReader(r.path) as rr:
                _ = rr.info
        except KohakuBoardError as exc:
            corrupt += 1
            print(f"ERROR {r.path}: {exc}", file=sys.stderr)

    if corrupt:
        print(f"status:          {corrupt} run(s) failed health check", file=sys.stderr)
        return 1
    print("status:          all checks passed")
    return 0


def _delegate_optional(module_name: str, func_name: str, extra: str, argv: list[str]) -> int:
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(
            f"This command requires optional package '{module_name}'. "
            f'Install with: pip install "kohakuboard[{extra}]"',
            file=sys.stderr,
        )
        return 2
    entry = getattr(mod, func_name, None)
    if entry is None:
        print(f"Optional module '{module_name}' does not expose '{func_name}'", file=sys.stderr)
        return 2
    return int(entry(argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kobo",
        description="KohakuBoard v2 CLI — local-first ML experiment tracking",
    )
    parser.add_argument("--version", action="version", version=f"kohakuboard {rs_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ls = sub.add_parser("ls", help="List projects and runs")
    p_ls.add_argument("dir", nargs="?", default=None, help="Base KohakuBoard directory")
    p_ls.add_argument("--project", "-p", default=None)
    p_ls.add_argument("--state", default=None)
    p_ls.add_argument("--group", default=None)
    p_ls.add_argument("--search", "-q", default=None)
    p_ls.add_argument("--sort", default="-updated_at_ms")
    p_ls.add_argument("--limit", type=int, default=None)
    p_ls.add_argument("--json", action="store_true")
    p_ls.set_defaults(func=_cmd_ls)

    p_info = sub.add_parser("info", help="Show run metadata, config, summary, and storage size")
    p_info.add_argument("run_dir", help="Path to a run directory containing run.db")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=_cmd_info)

    p_logs = sub.add_parser("logs", help="View or FTS5-search captured console logs")
    p_logs.add_argument("run_dir", help="Path to a run directory containing run.db")
    p_logs.add_argument("--search", "-q", default=None, help="BM25 full-text search query")
    p_logs.add_argument("--tail", "-n", type=int, default=100)
    p_logs.add_argument("--json", action="store_true")
    p_logs.set_defaults(func=_cmd_logs)

    p_exp = sub.add_parser("export", help="Export run scalar series to CSV, JSON, or NPZ")
    p_exp.add_argument("run_dir", help="Path to a run directory containing run.db")
    p_exp.add_argument("--format", "-f", choices=["csv", "json", "npz"], required=True)
    p_exp.add_argument("--out", "-o", required=True, help="Output file path")
    p_exp.add_argument("--metric", dest="metrics", action="append", default=None)
    p_exp.set_defaults(func=_cmd_export)

    p_doc = sub.add_parser("doctor", help="Check native extension, storage health, and catalog")
    p_doc.add_argument("dir", nargs="?", default=None, help="Base KohakuBoard directory")
    p_doc.add_argument("--reindex", action="store_true", help="Rebuild .catalog.db from scratch")
    p_doc.set_defaults(func=_cmd_doctor)

    for web_cmd in ("open", "serve"):
        p_web = sub.add_parser(web_cmd, help="Web server command (requires kohakuboard[web])")
        p_web.add_argument("rest", nargs=argparse.REMAINDER)
        p_web.set_defaults(
            func=lambda a, c=web_cmd: _delegate_optional(
                "kohakuboard_web.cli", f"run_{c}", "web", a.rest
            )
        )

    for tui_cmd in ("tui", "watch"):
        p_tui = sub.add_parser(tui_cmd, help="Terminal UI command (requires kohakuboard[tui])")
        p_tui.add_argument("rest", nargs=argparse.REMAINDER)
        p_tui.set_defaults(
            func=lambda a, c=tui_cmd: _delegate_optional(
                "kohakuboard.tui.cli", f"run_{c}", "tui", a.rest
            )
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KohakuBoardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
