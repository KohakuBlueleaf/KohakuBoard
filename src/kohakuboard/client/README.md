# KohakuBoard Client Library

High-performance, non-blocking logging for ML experiments. This document focuses on the **training-side** implementation that lives in `src/kohakuboard/client/`.

---

## Architecture Overview

```
┌────────────────────────────────────┐      ┌───────────────────────────────┐
│          Training Process          │      │        Writer Process         │
│                                    │      │                               │
│  board.log(...)   ───────────────┐ │      │  Queue.get() (drain burst)    │
│    ↳ classify payloads           │ │      │    ↳ scalars → ColumnVault     │
│    ↳ queue.put(message) ---------┼─┼────► │    ↳ histograms → ColumnVault  │
│  board.step() (global_step)      │ │      │    ↳ tables/media → SQLite     │
│  stdout/stderr capture hooks     │ │      │    ↳ tensors/media blobs → KV  │
└────────────────────────────────────┘      │  Flush every batch             │
                                            │  Optional sync worker          │
                                            └───────────────────────────────┘
```

- **Non-blocking logging**: `board.log()` returns immediately after pushing a message to a `multiprocessing.Queue` (50k capacity on Windows, unbounded on POSIX).
- **Dedicated writer**: `LogWriter` batches messages, writes to disk, and optionally streams to a sync worker.
- **Three-tier storage** (all SQLite under the hood, powered by [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault)):
  1. `data/metrics/*.db` – ColumnVault databases (one per metric/histogram).
  2. `data/metadata.db` – SQLite tables for steps, tables, tensors, media metadata.
  3. `media/blobs.db` + files – KVault for deduplicated blobs + actual PNG/MP4/etc.
- **Manual sync**: Because local and remote deployments read the same folders, copying `{project}/{run_id}` between machines is the supported sharing strategy until the refreshed HTTP sync lands.

---

## Directory Layout

```
{base_dir}/
└── {project}/
    └── {run_id}[+annotation]/
        ├── metadata.json
        ├── data/
        │   ├── metrics/<metric>.db        # ColumnVault
        │   ├── metadata.db                # SQLite tables
        │   └── tensors/<name>.db          # Optional tensor vaults
        ├── media/
        │   ├── blobs.db                   # KVault KV store
        │   └── <name>_<idx>_<step>.png    # Content-addressed files
        └── logs/
            ├── output.log                 # Captured stdout/stderr
            └── writer.log                 # Writer diagnostics
```

Each component is SWMR-safe thanks to SQLite WAL mode, so readers can open files while the writer is active.

---

## Key Classes

| Module | Responsibility |
|--------|----------------|
| `client/board.py` | Public API (`Board`, `board.log`, `board.step`, etc.) |
| `client/writer.py` | Background writer process + optional sync worker |
| `client/types/` | Data-type helpers (`Media`, `Table`, `Histogram`, `TensorLog`, `KernelDensity`) |
| `client/storage/` | Hybrid storage layer (ColumnVault metrics + SQLite metadata + KVault) |
| `client/capture.py` | stdout/stderr tee used for `capture_output=True` |

---

## Logging API Cheatsheet

```python
from kohakuboard.client import Board, Histogram, Media, Table, TensorLog, KernelDensity

board = Board(
    name="resnet50",
    project="vision",
    base_dir="/mnt/kobo",
    config={"lr": 1e-3, "batch_size": 128},
)

for batch_idx, batch in enumerate(train_loader):
    loss = train_step(batch)
    optimizer.step()

    board.step()  # once per optimizer update
    board.log(
        **{
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "attention/tensor": TensorLog(attn_tensor, metadata={"layer": 3}),
            "kde/logits": KernelDensity(values=logits.cpu().numpy()),
        }
    )

    if batch_idx % 200 == 0:
        board.log(
            samples=Media(sample_images, caption="Predictions"),
            grads=Histogram(model.layer.weight.grad).compute_bins(),
        )

board.flush()   # optional (atexit already handles this)
board.finish()  # optional (also handled automatically)
```

**Best practices**

- **`board.step()`** represents optimizer steps. Never tie it to epochs.
- **Batch heavy payloads** in a single `board.log()` call to avoid step inflation.
- **Use `Histogram(...).compute_bins()`** and `TensorLog` to shrink queue traffic.
- **Log media sparingly** (e.g., once per epoch) to prevent queue saturation.
- **Check `writer.log`** if something fails—errors are routed there.

---

## Writer Behavior

- Drains up to 10k messages per burst and flushes immediately.
- Adaptive sleep (0.5s → 5s) when the queue is empty to keep CPU usage low.
- Warns if queue size exceeds 40k entries (Windows only, since `qsize()` is supported there).
- Starts an optional sync worker when `Board(..., sync_enabled=True, remote_url=...)` is configured. **Note:** the current sync worker still targets the legacy DuckDB uploader, so prefer manual copy until the HTTP sync is refreshed.

---

## stdout/stderr Capture

- Enabled by default (`capture_output=True`).
- Uses `MemoryOutputCapture` to tee both streams and push chunks through the queue so the writer can store them in `logs/output.log`.
- Disable capture if your process already manages logging (e.g., distributed training frameworks).

---

## Troubleshooting

| Symptom | Likely Cause | Mitigation |
|---------|--------------|-----------|
| Queue warning: `size is 40000/50000` | Logging too many large payloads per step | Reduce frequency, precompute histograms, or turn off verbose media logging. |
| Missing run in UI | `metadata.json` not written (crash before init) | Ensure `Board` was constructed and `metadata.json` exists; otherwise delete folder. |
| Writer crash | Exception in `writer.log` | Fix root cause, runs are durable because ColumnVault/SQLite flush on every batch. |
| `kobo sync` fails | Command still expects DuckDB export | Copy the `{project}/{run}` folder manually or wait for the new sync API. |

---

## Sharing Runs

```
# Copy run to another machine (manual sync)
rsync -a /mnt/kobo/vision/20250201_120301_xyz \\
      user@server:/var/kohakuboard/vision/

# Restart or refresh kobo-serve / kobo open on the server
```

Because storage is entirely SQLite-based, copying folders between machines is safe and keeps everything (metrics, metadata, media, tensors) intact.

---

## Additional Resources

- [Usage Manual](../../../docs/kohakuboard/usage-manual.md) – Workflow-oriented checklist.
- [API Reference](../../../docs/kohakuboard/api.md) – Detailed method docs.
- [Architecture Guide](../../../docs/kohakuboard/architecture.md) – Deep dive into the storage stack.

Happy logging!
