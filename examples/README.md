# KohakuBoard Examples

This directory contains example scripts demonstrating KohakuBoard features and best practices.

## Quick Start

**For beginners:** Start with `quick_start.py`

```bash
# Run with defaults (syncs to local server at localhost:48889)
python examples/quick_start.py

# Disable remote sync (local only)
python examples/quick_start.py --remote-url ""

# Sync to production server
python examples/quick_start.py \
    --remote-url https://board.example.com \
    --remote-token YOUR_AUTH_TOKEN \
    --remote-project my_project
```

A minimal example showing the most common use cases:
- Creating a board
- Logging scalars (metrics)
- Logging images
- Logging histograms
- Logging tables
- Logging tensor snapshots (stored in `data/tensors/*.db`)
- Logging kernel density coefficients for smooth histogram flows

## Comprehensive Examples

### 1. `all_features_demo.py` - Complete Feature Tour

Demonstrates ALL KohakuBoard capabilities:

```bash
python examples/all_features_demo.py
```

**What it covers:**
- ✅ Scalars: Basic metrics logging
- ✅ Namespace organization: `train/`, `val/` tabs
- ✅ Media: Images (SQLite KV storage)
- ✅ Media: Multiple images (galleries)
- ✅ Tables: Basic data tables
- ✅ Tables: With embedded media
- ✅ Histograms: Raw values (SharedMemory)
- ✅ Histograms: Precomputed bins/counts
- ✅ Histograms: Batch logging (multiple at once)
- ✅ Tensors: Full payload snapshots via KohakuVault
- ✅ Kernel density: Precomputed KDE coefficients with recommended ranges
- ✅ Unified API: Mix all data types in one call
- ✅ Performance: Large histograms (1M values)

**Best for:** Understanding all available features

---

### 2. `kohakuboard_cifar_training.py` - Real Training Example

A complete CIFAR-10 training script with ConvNeXt model:

```bash
python examples/kohakuboard_cifar_training.py
```

**What it covers:**
- Real neural network training loop
- Gradient and parameter histograms
- Validation metrics and per-class accuracy
- Sample prediction tables with images
- Namespace organization (`train/`, `val/`, `gradients/`, `params/`)
- Best practices for avoiding step inflation

**Best for:** Real-world ML training scenarios

---

### 3. `performance_showcase.py` - Performance Benchmarks

Tests and demonstrates v0.3.0 performance improvements:

```bash
python examples/performance_showcase.py
```

**What it benchmarks:**
- **SQLite KV media storage**: 100+ images
- **SharedMemory histograms**: 10K, 100K, 1M values
- **Tensor logging**: KohakuVault throughput with 100+ payloads
- **Kernel density logging**: KDE coefficient generation/resampling
- **Mixed workload**: Scalars + media + histograms
- **Queue performance**: 1000+ messages/sec

**Best for:** Understanding performance characteristics

---

### 4. `non_blocking_benchmark.py` - Queue/Writer Stress Test

Runs seven phases to profile the non-blocking client:
1. Scalar-only logging
2. Histogram-only (1e4 values per hist)
3. Media-only (random RGB images)
4. Tensor-only (large float32 blobs)
5. KDE-only (raw samples automatically converted)
6. Mixed scalar + tensor workload
7. Mixed scalar + tensor + KDE workload

Each phase reports throughput in **values/sec**.

```bash
# Reuse the same board directory; wipe once at the start if desired
python examples/non_blocking_benchmark.py --clean-start \
    --scalar-steps 4000 \
    --tensor-size 131072
```

Key flags:
- `--base-dir`, `--board-id` for deterministic storage paths
- `--clean-start` to delete any previous run before the suite begins
- `--purge-after-phase` to delete the board directory after each scenario (default: keep data to avoid Windows file locks)
- `--tensor-size`, `--hist-values`, `--media-width`/`--media-height`, `--kde-samples` to scale load
- `--mix-tensor-interval`, `--mix2-tensor-interval`, `--mix2-kde-interval` to control the mixed phases
- `--skip-*` flags (e.g., `--skip-media`, `--skip-kde`) to disable individual tests

**Best for:** Understanding how each data type stresses the queue and verifying the writer keeps up.

---

## New in v0.3.0

All examples showcase the latest features:

### 🚀 SQLite KV Media Storage
- **Before:** Thousands of individual files on filesystem
- **Now:** Single SQLite database file with optimized KV table
- **Benefits:** No filesystem overhead, dynamic size, WAL mode for concurrent reads

```python
# Media automatically stored in SQLite KV
board.log(image=Media(img_array, media_type="image"))
# Stored in: {board_dir}/media/blobs.db (not individual files!)
```

### 🚀 SharedMemory Histogram Transfer
- **Before:** Serialized arrays through queue
- **Now:** Zero-copy transfer via SharedMemory
- **Benefits:** 10-100x faster for large arrays, no memory overhead

```python
# Large histogram (1M values) transferred via SharedMemory
values = np.random.randn(1_000_000)
board.log(**{"weights/dist": Histogram(values)})
# Uses SharedMemory internally - transparent to user!
```

### 🚀 mp.Queue Performance
- **Before:** manager.Queue with proxy overhead
- **Now:** Direct mp.Queue
- **Benefits:** Better performance, proper cleanup

```python
# Automatically uses mp.Queue (no code changes needed)
board = Board(name="my_experiment")
# Queue cleanup happens automatically in board.finish()
```

### 🚀 Tensor Logging via KohakuVault
- **Before:** No first-class tensor storage
- **Now:** Namespace-scoped KVault files in `data/tensors/`
- **Benefits:** Efficient binary storage with SQLite metadata for discovery

```python
weights = np.random.randn(128, 128).astype(np.float32)
board.log_tensor(
    "tensors/block0",
    weights,
    metadata={"layer": "block0", "shape": list(weights.shape)},
)
# Data stored in: {board_dir}/data/tensors/tensors.db (namespaced)
```

### 🚀 Kernel Density Logging
- **Before:** Histograms with varying ranges across steps
- **Now:** Store KDE coefficients + recommended ranges, resample on demand
- **Benefits:** Smooth distribution flows with consistent binning in the UI

```python
activations = np.random.randn(4096).astype(np.float32)
board.log_kernel_density(
    "activations/kde_flow",
    values=activations,
    num_points=256,
    percentile_clip=(1.0, 99.0),
)
```

---

## Viewing Results

After running any example:

```bash
# Start the web UI
python -m kohakuboard.main

# Open browser
http://localhost:48889
```

Navigate to your board and explore:
- **Scalars**: Line charts with namespace tabs
- **Media**: Image galleries
- **Tables**: Sortable, filterable tables
- **Histograms**: Interactive distribution visualizations

---

## Remote Sync (Local + Cloud)

All examples **sync by default** to test the full workflow!

### Default Behavior (Development Mode)

By default, all examples are configured for **local development/testing**:
- ✅ **Remote sync enabled** by default
- 🏠 **Syncs to localhost:48889** (your local KohakuBoard server)
- 🔓 **No authentication** (`no-auth-mode`)
- ⚡ **Fast sync interval** (1 second for demos)

```bash
# Just run - syncs to local server automatically
python examples/quick_start.py

# Disable sync if you only want local logging
python examples/quick_start.py --remote-url ""
```

### Production Mode

For production servers with authentication:

```bash
python examples/quick_start.py \
    --remote-url https://board.example.com \
    --remote-token YOUR_AUTH_TOKEN \
    --remote-project my_project \
    --sync-interval 30
```

### How It Works

1. **Logs locally** - Full logging to local storage (KohakuVault ColumnVault + SQLite metadata + KV)
2. **Syncs to remote** - Background worker syncs incremental changes every N seconds
3. **View anywhere** - Access logs locally OR remotely

### Command-Line Arguments

All examples support these flags:

| Argument | Description | Default |
|----------|-------------|---------|
| `--remote-url` | Remote server URL | `http://127.0.0.1:48889` |
| `--remote-token` | Authentication token | `no-auth-mode` |
| `--remote-project` | Project name on remote server | Script-specific |
| `--sync-interval` | Sync interval in seconds | `1` (demos) / `30` (training) |

**Note:** To disable sync entirely, use `--remote-url ""`

### Examples

**Run with defaults (local server sync):**
```bash
# Just run - syncs to localhost:48889 automatically
python examples/quick_start.py
python examples/all_features_demo.py
python examples/performance_showcase.py
python examples/kohakuboard_cifar_training.py
```

**Local logging only (no sync):**
```bash
python examples/quick_start.py --remote-url ""
```

**Production server with authentication:**
```bash
python examples/kohakuboard_cifar_training.py \
    --epochs 25 \
    --batch-size 256 \
    --remote-url https://board.example.com \
    --remote-token YOUR_AUTH_TOKEN \
    --remote-project cifar_experiments
```

### What Gets Synced

✅ **Synced to remote:**
- Scalars (metrics)
- Media metadata + files (via SQLite KV)
- Tables
- Histograms (precomputed bins/counts)
- Board configuration

📍 **Always local (unless you copy the folder):**
- SQLite KV database (`media/blobs.db`)
- ColumnVault metric files (`data/metrics/*.db`)
- SQLite metadata (`data/metadata.db`)

**Result:** You get the best of both worlds - fast local access + cloud backup/sharing!

---

## Tips and Best Practices

### 1. Avoid Step Inflation

**❌ Bad - Each log increments step:**
```python
for i in range(100):
    board.log(loss=loss)              # Step 0
    board.log(accuracy=acc)           # Step 1 (different step!)
    board.log_histogram("grads", g)   # Step 2 (different step!)
```

**✅ Good - All data at same step:**
```python
for i in range(100):
    board.step()  # Explicit step increment
    board.log(
        loss=loss,
        accuracy=acc,
        **{"grads/layer1": Histogram(g)}
    )  # All at same step!
```

### 2. Use Namespaces for Organization

```python
# Creates separate tabs in UI
board.log(**{
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
})
# UI shows: train/ tab and val/ tab
```

### 3. Precompute Large Histograms

```python
# For very large arrays, precompute to save worker CPU
hist = Histogram(large_array, num_bins=128)
hist.compute_bins()  # Compute now
board.log(**{"weights/dist": hist})
```

### 4. Use Unified API

```python
# Log everything together - same step, one queue message
board.log(
    loss=loss,
    accuracy=accuracy,
    sample_image=Media(img),
    **{"gradients/all": Histogram(grads)},
    metrics_table=Table(table_data),
)
```

---

## Common Issues

### Queue Size Warning

```
Queue size is 40000/50000. Consider reducing logging frequency.
```

**Solution:** Log less frequently or use batch logging
```python
# Log every 10 steps instead of every step
if board._global_step % 10 == 0:
    board.log(loss=loss)
```

### Memory Usage

For very large histograms (10M+ values), consider:
1. Using `precision="compact"` (uint8 instead of int32)
2. Reducing `num_bins`
3. Sampling values before logging

```python
# Sample 100K values from 10M array
sampled = values[::100]  # Every 100th value
board.log(**{"weights/dist": Histogram(sampled)})
```

---

## Questions?

- **Documentation:** Check the main README.md
- **Issues:** https://github.com/KohakuBlueleaf/KohakuBoard/issues
- **Examples:** This directory!

Happy logging! 🎉
