# KohakuBoard

> **V2 WIP:** KohakuBoard v2 is currently under active development. The Rust core, Python logging SDK, offline SWMR readers, and base `kobo` CLI are implemented; web UI, terminal UI, and remote sync are still in progress.

**Local-first, zero-overhead ML experiment tracking backed by Rust and [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault).**

KohakuBoard v2 is a ground-up rewrite designed to feel as ergonomic as Weights & Biases while keeping your training loop completely unblocked and your experiment data stored locally in portable, inspectable SQLite files.

## Key Capabilities

- **~2.5 μs scalar hot path:** `board.log(...)` pushes onto a bounded Rust `crossbeam-channel` (`block` or `drop_oldest`); all SQLite/KohakuVault I/O, histogram binning, `rustfft` Gaussian KDE, PNG/WebP/JPEG encoding, `blake3` hashing, and gzip tensor compression run off-GIL on a dedicated Rust OS thread.
- **Single-file `run.db` + lazy `blobs.db` (powered by [KohakuVault 1.2](https://github.com/KohakuBlueleaf/KohakuVault)):**
  - Every scalar metric is stored as one atomic 32-byte fixed-width `ScalarRow` (`step`, `global_step`, `ts_ms`, `value`) column in bounded `ColumnVault` BLOB blocks — preserving IEEE-754 `NaN`/`±Inf` verbatim, eliminating cross-column SWMR tearing, and supporting 1,000+ metrics in **one file** (zero FD explosion).
  - Media and tensors are deduplicated by `blake3` hash inside a lazy `blobs.db` (`KVault`) that is only created when binary payloads are logged.
  - Captured `stdout`/`stderr` streams are indexed into `TextVault` with SQLite FTS5 BM25 full-text search.
  - Rebuildable `{data_dir}/.catalog.db` indexes projects/runs for sub-millisecond listing and live-PID crash detection.
- **Newest-first reverse-paged binary chunk framing:** Ready for zero-copy JS `Float64Array` / NumPy `np.frombuffer` progressive rendering — first 8,192 newest points arrive in milliseconds while full lossless history loads backward to step 0.
- **DDP & lifecycle safe:** `rank_zero_only=True` automatically turns non-zero `RANK`/`LOCAL_RANK` workers into zero-overhead no-ops; `resume="allow"|"must"` continues interrupted runs seamlessly; no `with` block required — `atexit` + a non-destructive chained `sys.excepthook` + Rust `Drop` flush everything and record exact `finished` / `interrupted` / `crashed` state automatically.
- **Minimal runtime footprint:** Base `kohakuboard` depends only on `numpy>=1.24` (and ships a single `abi3-py310` native extension `kohakuboard._rs`).

---

## Installation

From source (requires Rust toolchain + Python 3.10+):

```bash
pip install maturin numpy
maturin develop --release
```

With development/test tools:

```bash
uv pip install -e ".[dev]"
```

---

## Quick Start

### 1. Drop-in logging (no `with` block required)

Use either the W&B-style module API (`kobo.init` / `kobo.log`) or a plain `Board(...)` handle (`with Board(...)` is also supported for scoped multi-run loops):

```python
import kohakuboard as kobo
from kohakuboard import Artifact, Histogram, KernelDensity, Media, Table, TensorLog

kobo.init(
    name="dit-xl-stage1",
    project="vision",
    group="lr-3e-4",
    tags=["bf16", "baseline"],
    config={"lr": 3e-4, "batch_size": 128, "layers": 24},
)
kobo.define_metric("eval/*", step_metric="global_step", summary="min")

for step in range(1_000):
    kobo.log({"train/loss": 1.0 / (step + 1), "train/lr": 3e-4})

    if step % 200 == 0:
        kobo.log(
            {
                "samples": Media(rgb_array, caption=f"step {step}", format="webp"),
                "grad/norm": Histogram(grad_array, bins=64, precision="exact"),
                "grad/kde": KernelDensity(grad_array, num_points=256),
                "eval/table": Table(
                    columns=["prompt", "score", "preview"],
                    rows=[["amber terrarium", 0.94, Media(rgb_array)]],
                ),
                "debug/attn": TensorLog(attn_tensor, compress=True),
                "ckpt": Artifact(f"step-{step}", kind="checkpoint", path="model.safetensors"),
            },
            commit=False,
        )

kobo.alert("Finished", "Stage 1 converged cleanly", level="info")
# kobo.finish() is optional at process exit — atexit + Rust Drop flushes automatically.
```

Prefer an explicit handle (or multiple concurrent runs in one process)?

```python
from kohakuboard import Board

board = Board(name="ablation-b", project="vision", config={"lr": 1e-4})
board.log({"train/loss": 0.42})
```

### 2. Read runs offline (or live via SWMR) in Python

```python
from kohakuboard import ProjectReader, RunReader

with ProjectReader("./kohakuboard", project="vision") as proj:
    for run in proj.list_runs(group="lr-3e-4", sort="summary.train/loss"):
        print(run.run_id, run.name, run.state, run.summary.get("train/loss"))

with RunReader("./kohakuboard/vision/<run_dir>") as run:
    steps, gsteps, ts_ms, values = run.scalar("train/loss")  # zero-copy-style NumPy views
    hits = run.search_logs("NaN", limit=20)                  # TextVault FTS5 BM25 search
    hists = run.histograms("grad/norm")
    tensor = run.read_tensor(run.tensors("debug/attn")[0])
```

### 3. Inspect from the CLI (`kobo`)

```bash
kobo ls ./kohakuboard --project vision
kobo info ./kohakuboard/vision/<run_dir>
kobo logs ./kohakuboard/vision/<run_dir> --search "NaN" --tail 50
kobo export ./kohakuboard/vision/<run_dir> --format csv --out metrics.csv
kobo doctor ./kohakuboard --reindex
```

---

## Measured Performance

Reproducible via [`python examples/benchmark.py`](examples/benchmark.py) (`kohakuboard 2.0.0`, release build):

| Benchmark | Workload | Linux (WSL2, Py 3.12) | Windows 11 (Py 3.13) |
| --- | --- | --- | --- |
| **Scalar hot path** | 100k `board.log()` × 4 scalars (400k pts) | **1.66 μs/call** (`p50=0.83 μs`, **2.41M scalars/s**) | **2.45 μs/call** (`p50=2.20 μs`, **1.63M scalars/s**) |
| **Scalar durability** | Flush all 400k scalars to `run.db` (14.94 MiB, 39.1 B/pt) | **24.0 ms** flush (**190.1 ms** total) | **45.4 ms** flush (**290.8 ms** total) |
| **Wide-metric scaling** | 1,000 metrics × 200 steps (200k pts, **1 file** `36.91 MiB`) | **90.6 ms** enq / **559.2 ms** total | **70.0 ms** enq / **500.3 ms** total |
| **Newest-first first chunk** | Latest 8,192 rows of 500k series (256.2 KiB aligned frame) | **3.77 ms** | **6.77 ms** |
| **Full lossless sweep** | All 500k rows newest-first (10 frames, 15.26 MiB) | **8.1 ms** | **21.0 ms** |
| **Live-tail poll** | Incremental 128-row `?after=N` frame (4.2 KiB) | **0.07 ms** | **0.10 ms** |
| **Off-GIL Histogram** | `Histogram(100k f32, 64 bins, exact)` | **0.18 ms** enq (`3.80 ms` bg) | **0.35 ms** enq (`20.11 ms` bg) |
| **Off-GIL FFT-KDE** | `KernelDensity(100k f32, 256 pts, rustfft)` | **0.15 ms** enq (`32.58 ms` bg) | **0.14 ms** enq (`3.95 ms` bg) |
| **Off-GIL Media (PNG / WebP)** | `Media(512×512 RGB)` + `blake3` dedup | **0.19 / 0.17 ms** enq (`23.4 / 3.6 ms` bg) | **0.56 / 0.46 ms** enq (`25.1 / 4.5 ms` bg) |
| **Catalog & FTS5 search** | Warm `.catalog.db` refresh / 10k-line BM25 query | **0.04 ms** / **3.50 ms** | **0.59 ms** / **1.36 ms** |

---

## Examples

Runnable scripts live in [`examples/`](examples/README.md):

- [`examples/quick_start.py`](examples/quick_start.py) — minimal `Board` logging + offline `ProjectReader` / `RunReader` workflow.
- [`examples/all_features_demo.py`](examples/all_features_demo.py) — end-to-end multimodal tour (scalars + `NaN`/`Inf`, deduplicated `Media`, `Table` with embedded `Media`, `Histogram`, FFT `KernelDensity`, `TensorLog`, `Artifact`, alerts, FTS5 log search, live SWMR read, and `resume="must"`).
- [`examples/benchmark.py`](examples/benchmark.py) — reproduces the full benchmark table above (`--quick` for a fast run).

---

## Storage Layout

```text
{data_dir}/
├── .catalog.db                  # Rebuildable SQLite project/run index
└── {project}/
    └── {run_id}[_{annotation}]/
        ├── run.json             # Atomic JSON manifest for fast scanning
        ├── run.db               # KohakuVault 1.2: kb_meta + ColumnVault + TextVault + event tables
        └── blobs.db             # Lazy blake3-deduplicated KVault (created only when blobs are logged)
```

---

## Licensing

KohakuBoard v2 uses a 2-way license split at the package boundary:

- **`kohakuboard`** (Rust core, Python client, offline readers, TUI integration, base CLI): [Apache License 2.0](LICENSE)
- **`kohakuboard-web`** (`packages/kohakuboard-web/`, `apps/web/`): [Kohaku Software License 1.0](packages/kohakuboard-web/LICENSE)
