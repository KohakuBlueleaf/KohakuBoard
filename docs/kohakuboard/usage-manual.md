---
title: KohakuBoard Usage Manual
description: Day-to-day workflow for logging, inspecting, and sharing boards
icon: i-carbon-notebook-reference
---

# KohakuBoard Usage Manual

Practical checklist for running KohakuBoard in production or research labs. This guide complements the README by focusing on **repeatable workflows** instead of feature marketing.

---

## 1. Prepare the Workspace

1. **Choose a base directory** – default is `./kohakuboard`. Override per run with `Board(..., base_dir="/mnt/experiments")` or globally with `KOHAKU_BOARD_DATA_DIR`.
2. **Organize by project** – pass `project="vision"` (or let it default to `default`). A run lives at `{base_dir}/{project}/{run_id}`.
3. **Version configs** – include your hyperparameters in `config={...}` so they are written to `metadata.json`.
4. **Capture output** – leave `capture_output=True` (default) to mirror stdout/stderr into `logs/output.log` for later debugging.

```python
from kohakuboard.client import Board

board = Board(
    name="cifar10-resnet18",
    project="vision",
    base_dir="/mnt/kohakuboard",
    config={"lr": 1e-3, "batch_size": 128, "optimizer": "AdamW"},
)
```

---

## 2. Logging During Training

| Action | When to call | Notes |
|--------|--------------|-------|
| `board.step()` | **Once per optimizer update** | Sets `global_step`. Never tie it to epochs. |
| `board.log(...)` | Whenever you have new measurements | Scalars, `Media`, `Table`, `Histogram`, `TensorLog`, and `KernelDensity` can be mixed in one call so they share the same step. |
| `board.flush()` | Before long pauses or just prior to process exit | Blocks until the writer drains the queue. Usually unnecessary because `finish()` is called automatically. |

Example loop:

```python
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    loss = model(data, target)
    loss.backward()
    optimizer.step()

    board.step()  # 1 call per optimizer step
    board.log(
        **{
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
        }
    )

    if batch_idx % 200 == 0:
        board.log(
            grad_hist=Histogram(model.layer.weight.grad).compute_bins(),
            samples=Table(prediction_rows),
        )
```

**Best practices**

- Batch heavy payloads (histograms, tensors, media) so they travel through the queue once per step.
- Log media/tables sparingly (e.g., once per epoch) to keep queue utilization low (`50_000` message buffer).
- When `memory_mode=True`, configure `sync_enabled` with a real endpoint or expect data loss after exit.

---

## 3. Inspecting Boards Locally

```bash
# Watch an entire directory of runs
kobo open /mnt/kohakuboard --browser

# Serve a single project
KOHAKU_BOARD_DATA_DIR=/mnt/kohakuboard/vision kobo open . --port 5175
```

- `kobo open` always operates in **local mode** (no auth, no DB). It reads the same files that the writer produced.
- Metadata discovery relies on `{run}/metadata.json`. If a run does not appear, verify that file exists and contains valid JSON.
- Logs live under `{run}/logs/`; inspect `writer.log` if something fails to load.

---

## 4. Sharing Runs (Manual Sync)

Local and remote deployments share identical storage (KohakuVault column stores + SQLite). To move runs between machines:

1. Copy the folder:  
   ```bash
   rsync -a /mnt/kohakuboard/vision/20250201_120301_abcd remote:/var/kohakuboard/vision/
   ```
2. Ensure permissions allow the viewer/server to read the files.
3. Restart or reload `kobo open` / `kobo-serve`. The run appears immediately because the files are final.

> ℹ️ `kobo sync` is still wired to the legacy DuckDB exporter. Until the new API lands, manual copy/rsync is the supported way to sync boards.

---

## 5. Running the Authenticated Server

Use the separate `kohakuboard_server` package when you need multi-user access:

```bash
kobo-serve \
  --data-dir /var/kohakuboard \
  --db sqlite:///kohakuboard.db \
  --port 48889 \
  --workers 4 \
  --session-secret "$(openssl rand -hex 32)"
```

- The server reads exactly the same folders produced by the client. Drop new runs into `/var/kohakuboard/{project}` to share them.
- `--no-auth` is available for development only. Production deployments should configure a proper secret and TLS.
- PostgreSQL is available via `--db-backend postgres`, but the board storage itself remains pure SQLite/KohakuVault files.

---

## 6. Maintenance & Troubleshooting

- **Queue warnings** – The client logs a warning when the queue exceeds 80% capacity. Reduce logging frequency or precompute histograms (`Histogram(...).compute_bins()`).
- **Disk usage** – Media is content-addressed (`media/blobs.db` + files). Deleting a run folder removes both metrics and associated blobs.
- **Corruption recovery** – Because logs are flushed continuously, a crash typically leaves a consistent SQLite/WAL state. Re-open the viewer; if a board is missing, check `writer.log` for the last successful flush.
- **Renaming runs** – Use annotations: `Board(..., annotation="ft-bert")` appends `+ft-bert` to the folder name without touching the stable `run_id`.
- **Exporting** – Use the inspector tooling (`scripts/`, `kohakuboard.inspector`) to export metrics to Parquet/CSV if you need offline analysis.

---

## 7. Quick Reference

- `Board.step()` ⇒ optimizer steps, not epochs.
- `Board.log()` ⇒ mix scalars + structured types in one call to avoid step inflation.
- `kobo open DIR` ⇒ local, no auth.
- `kobo-serve` ⇒ authenticated server (still stabilizing).
- Manual sync ⇒ copy `{base_dir}/{project}/{run}` between machines.
- Logs & metadata ⇒ plain text/SQLite files for easy inspection/backups.

Happy logging!
