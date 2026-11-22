"""Reproducible KohakuBoard v2 performance benchmark suite.

Measures:
  1. Scalar hot-path latency (mean, p50, p95, p99) & end-to-end flush throughput
  2. High-metric-count scaling (1,000 metrics inside a single `run.db`)
  3. Newest-first reverse-paged binary chunk reads on a 500,000-row scalar series
  4. Off-GIL rich compute enqueue latency vs total flush time (Histogram, FFT-KDE, PNG/WebP)
  5. `.catalog.db` refresh & `TextVault` FTS5 BM25 search latency
"""

from __future__ import annotations

import argparse
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from kohakuboard import (
    Board,
    Histogram,
    KernelDensity,
    Media,
    ProjectReader,
    RunReader,
    __version__,
)


def bench_scalar_hot_path(root: Path, n_calls: int) -> dict[str, float]:
    """Measure per-call `board.log()` latency and flush throughput (4 scalars/call)."""
    latencies_ns = np.empty(n_calls, dtype=np.int64)
    with Board(
        run_id="sc01",
        project="bench",
        dir=root,
        system_metrics=False,
        capture_output=False,
        flush_interval_ms=250,
    ) as board:
        # Warm up column registration.
        for i in range(200):
            board.log(
                {"train/loss": float(i), "train/lr": 3e-4, "train/gnorm": 1.0, "train/acc": 0.5}
            )

        t_start = time.perf_counter()
        for i in range(n_calls):
            t0 = time.perf_counter_ns()
            board.log(
                {
                    "train/loss": 1.0 / (i + 1.0),
                    "train/lr": 3e-4,
                    "train/gnorm": 0.95,
                    "train/acc": 0.85,
                }
            )
            latencies_ns[i] = time.perf_counter_ns() - t0
        t_log = time.perf_counter() - t_start

        t_f0 = time.perf_counter()
        board.flush()
        t_flush = time.perf_counter() - t_f0
        run_dir = board.run_dir

    assert run_dir is not None
    db_bytes = (run_dir / "run.db").stat().st_size
    total_pts = (n_calls + 200) * 4
    lat_us = latencies_ns / 1_000.0
    return {
        "n_calls": float(n_calls),
        "total_points": float(n_calls * 4),
        "mean_us": (t_log / n_calls) * 1e6,
        "p50_us": float(np.percentile(lat_us, 50)),
        "p95_us": float(np.percentile(lat_us, 95)),
        "p99_us": float(np.percentile(lat_us, 99)),
        "calls_per_sec": n_calls / t_log,
        "scalars_per_sec": (n_calls * 4) / t_log,
        "flush_ms": t_flush * 1e3,
        "total_ms": (t_log + t_flush) * 1e3,
        "bytes_per_point": db_bytes / total_pts,
        "db_mib": db_bytes / (1024 * 1024),
    }


def bench_thousand_metrics(root: Path, n_metrics: int, n_steps: int) -> dict[str, float]:
    """Measure scaling with 1,000 distinct scalar metrics in one `run.db`."""
    keys = [f"layer_{i // 20:02d}/metric_{i:04d}" for i in range(n_metrics)]
    payload = {k: float(i) * 0.01 for i, k in enumerate(keys)}

    t0 = time.perf_counter()
    with Board(
        run_id="m1k1",
        project="bench",
        dir=root,
        system_metrics=False,
        capture_output=False,
        flush_interval_ms=250,
    ) as board:
        for _ in range(n_steps):
            board.log(payload)
        t_enq = time.perf_counter() - t0
        board.flush()
        t_total = time.perf_counter() - t0
        run_dir = board.run_dir

    assert run_dir is not None
    db_files = [p.name for p in run_dir.iterdir() if p.suffix == ".db"]
    db_mib = (run_dir / "run.db").stat().st_size / (1024 * 1024)
    return {
        "n_metrics": float(n_metrics),
        "n_steps": float(n_steps),
        "total_points": float(n_metrics * n_steps),
        "enqueue_ms": t_enq * 1e3,
        "total_ms": t_total * 1e3,
        "db_file_count": float(len(db_files)),
        "db_mib": db_mib,
    }


def bench_newest_first_chunks(root: Path, n_rows: int) -> dict[str, float]:
    """Measure newest-first 8,192-row chunk read, full adaptive scan, and live tail read."""
    with Board(
        run_id="chk1",
        project="bench",
        dir=root,
        system_metrics=False,
        capture_output=False,
        flush_interval_ms=250,
    ) as board:
        for i in range(n_rows):
            board.log({"loss": float(i) * 1e-4})
        run_dir = board.run_dir

    assert run_dir is not None
    with RunReader(run_dir) as reader:
        # 1. First newest-first 8,192-row frame (what paints the initial chart in the browser).
        t0 = time.perf_counter()
        first_frame = reader.scalar_page_frame("chk1", "loss", limit=8192, x_dtype="f64")
        t_first_ms = (time.perf_counter() - t0) * 1e3

        # 2. Complete newest-first adaptive sweep down to row 0 (lossless full data).
        t1 = time.perf_counter()
        plan = reader.scalar_newest_first_plan("loss", first_chunk=8192, max_chunk=65536)
        total_bytes = 0
        for start, end in plan:
            total_bytes += len(reader.scalar_range_frame("chk1", "loss", start, end, x_dtype="f64"))
        t_all_ms = (time.perf_counter() - t1) * 1e3

        # 3. Incremental live-tail poll (`?after=n_rows - 128`).
        t2 = time.perf_counter()
        tail_frame = reader.scalar_page_frame(
            "chk1", "loss", after=n_rows - 128, limit=8192, x_dtype="f64"
        )
        t_tail_ms = (time.perf_counter() - t2) * 1e3

    return {
        "n_rows": float(n_rows),
        "first_chunk_ms": t_first_ms,
        "first_chunk_kib": len(first_frame) / 1024.0,
        "chunk_count": float(len(plan)),
        "full_sweep_ms": t_all_ms,
        "full_sweep_mib": total_bytes / (1024 * 1024),
        "live_tail_ms": t_tail_ms,
        "live_tail_kib": len(tail_frame) / 1024.0,
    }


def bench_off_gil_rich_compute(root: Path) -> dict[str, float]:
    """Compare main-thread enqueue time vs background Rust compute + write time."""
    rng = np.random.default_rng(7)
    samples_100k = rng.standard_normal(100_000, dtype=np.float32)
    img_512 = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)

    with Board(
        run_id="rich",
        project="bench",
        dir=root,
        system_metrics=False,
        capture_output=False,
    ) as board:
        t0 = time.perf_counter()
        board.log({"h": Histogram(samples_100k, bins=64, precision="exact")})
        hist_enq_ms = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        board.flush()
        hist_flush_ms = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        board.log({"k": KernelDensity(samples_100k, num_points=256, approximate=True)})
        kde_enq_ms = (time.perf_counter() - t2) * 1e3
        t3 = time.perf_counter()
        board.flush()
        kde_flush_ms = (time.perf_counter() - t3) * 1e3

        t4 = time.perf_counter()
        board.log({"img_png": Media(img_512, format="png")})
        png_enq_ms = (time.perf_counter() - t4) * 1e3
        t5 = time.perf_counter()
        board.flush()
        png_flush_ms = (time.perf_counter() - t5) * 1e3

        t6 = time.perf_counter()
        board.log({"img_webp": Media(img_512, format="webp")})
        webp_enq_ms = (time.perf_counter() - t6) * 1e3
        t7 = time.perf_counter()
        board.flush()
        webp_flush_ms = (time.perf_counter() - t7) * 1e3

    return {
        "hist_enq_ms": hist_enq_ms,
        "hist_flush_ms": hist_flush_ms,
        "kde_enq_ms": kde_enq_ms,
        "kde_flush_ms": kde_flush_ms,
        "png_enq_ms": png_enq_ms,
        "png_flush_ms": png_flush_ms,
        "webp_enq_ms": webp_enq_ms,
        "webp_flush_ms": webp_flush_ms,
    }


def bench_catalog_and_fts(root: Path, n_log_lines: int) -> dict[str, float]:
    """Measure `.catalog.db` refresh and FTS5 BM25 search across captured logs."""
    with Board(
        run_id="log1",
        project="bench",
        dir=root,
        system_metrics=False,
        capture_output=False,
    ) as board:
        lines = [
            (
                10_000 + i,
                i,
                "stdout" if i % 100 else "stderr",
                (
                    f"step {i}: CRITICAL NaN gradient spike in attn_proj"
                    if i % 500 == 0
                    else f"step {i}: loss={1.0 / (i + 1):.5f} tok/s=48210"
                ),
            )
            for i in range(n_log_lines)
        ]
        assert board._engine is not None
        board._engine.log_lines(lines)
        board.flush()
        run_dir = board.run_dir

    t0 = time.perf_counter()
    with ProjectReader(root) as proj:
        proj.refresh(force_rebuild=True)
        t_cold_ms = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        proj.refresh(force_rebuild=False)
        t_warm_ms = (time.perf_counter() - t1) * 1e3

    assert run_dir is not None
    with RunReader(run_dir) as reader:
        t2 = time.perf_counter()
        hits = reader.search_logs("CRITICAL NaN gradient", limit=20)
        t_fts_ms = (time.perf_counter() - t2) * 1e3

    return {
        "n_log_lines": float(n_log_lines),
        "catalog_cold_ms": t_cold_ms,
        "catalog_warm_ms": t_warm_ms,
        "fts_ms": t_fts_ms,
        "fts_hits": float(len(hits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KohakuBoard v2 performance benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Run smaller problem sizes")
    args = parser.parse_args()

    n_scalar_calls = 20_000 if args.quick else 100_000
    n_chunk_rows = 100_000 if args.quick else 500_000
    n_log_lines = 2_000 if args.quick else 10_000

    print(
        f"KohakuBoard v{__version__} Benchmark — "
        f"Python {platform.python_version()} on {platform.platform()}"
    )
    print("=" * 78)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        s = bench_scalar_hot_path(root, n_scalar_calls)
        print(
            f"[1] Scalar hot-path "
            f"({int(s['n_calls']):,} calls x 4 = {int(s['total_points']):,} pts):\n"
            f"    mean={s['mean_us']:.2f} us/call | p50={s['p50_us']:.2f} us | "
            f"p95={s['p95_us']:.2f} us | p99={s['p99_us']:.2f} us\n"
            f"    throughput={s['calls_per_sec']:,.0f} calls/s "
            f"({s['scalars_per_sec']:,.0f} scalars/s)\n"
            f"    flush={s['flush_ms']:.1f} ms | total={s['total_ms']:.1f} ms | "
            f"size={s['db_mib']:.2f} MiB ({s['bytes_per_point']:.1f} B/pt)"
        )

        m = bench_thousand_metrics(root, n_metrics=1_000, n_steps=200)
        print(
            f"[2] 1,000 metrics x 200 steps ({int(m['total_points']):,} pts):\n"
            f"    enqueue={m['enqueue_ms']:.1f} ms | total+flush={m['total_ms']:.1f} ms | "
            f"db_files={int(m['db_file_count'])} (run.db={m['db_mib']:.2f} MiB)"
        )

        c = bench_newest_first_chunks(root, n_chunk_rows)
        print(
            f"[3] Newest-first binary chunks ({int(c['n_rows']):,} rows):\n"
            f"    first 8,192-row frame ({c['first_chunk_kib']:.1f} KiB)"
            f"={c['first_chunk_ms']:.2f} ms\n"
            f"    full lossless sweep ({int(c['chunk_count'])} frames, "
            f"{c['full_sweep_mib']:.2f} MiB)={c['full_sweep_ms']:.1f} ms\n"
            f"    live-tail 128-row poll ({c['live_tail_kib']:.1f} KiB)={c['live_tail_ms']:.2f} ms"
        )

        r = bench_off_gil_rich_compute(root)
        print(
            "[4] Off-GIL rich compute (caller enqueue vs background compute+flush):\n"
            f"    Histogram(100k f32, 64 bins): enqueue={r['hist_enq_ms']:.2f} ms | "
            f"flush={r['hist_flush_ms']:.2f} ms\n"
            f"    FFT-KDE(100k f32, 256 pts):   enqueue={r['kde_enq_ms']:.2f} ms | "
            f"flush={r['kde_flush_ms']:.2f} ms\n"
            f"    Media(512x512 RGB, PNG):      enqueue={r['png_enq_ms']:.2f} ms | "
            f"flush={r['png_flush_ms']:.2f} ms\n"
            f"    Media(512x512 RGB, WebP):     enqueue={r['webp_enq_ms']:.2f} ms | "
            f"flush={r['webp_flush_ms']:.2f} ms"
        )

        f = bench_catalog_and_fts(root, n_log_lines)
        print(
            f"[5] Catalog & TextVault FTS5 ({int(f['n_log_lines']):,} log lines):\n"
            f"    catalog cold rebuild={f['catalog_cold_ms']:.2f} ms | "
            f"warm refresh={f['catalog_warm_ms']:.2f} ms\n"
            f"    FTS5 BM25 search={f['fts_ms']:.2f} ms ({int(f['fts_hits'])} hits)"
        )


if __name__ == "__main__":
    main()
