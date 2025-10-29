# KohakuBoard Client Library

**Production-ready, non-blocking logging library for ML experiments**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Process                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Your Training Code                                    │ │
│  │  ↓                                                     │ │
│  │  board.log(loss=0.5)        ← Non-blocking (instant)  │ │
│  │  board.log_images(imgs)     ← Non-blocking (instant)  │ │
│  │  board.log_table(data)      ← Non-blocking (instant)  │ │
│  │     ↓                                                  │ │
│  │  Queue (maxsize=10,000)                               │ │
│  └─────────────────┬───────────────────────────────────────┘ │
│                    │ Messages                                │
└────────────────────┼─────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                   Writer Process                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  LogWriter.run()                                       │ │
│  │  ├─ Process messages from queue                        │ │
│  │  ├─ Convert images (PIL, numpy, torch → PNG)         │ │
│  │  ├─ Buffer logs (100 rows or 30 seconds)             │ │
│  │  └─ Flush to Parquet files                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                     ↓                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Storage                                               │ │
│  │  ├─ metrics.parquet  (scalars)                        │ │
│  │  ├─ media.parquet    (image metadata)                 │ │
│  │  ├─ tables.parquet   (table data)                     │ │
│  │  └─ media/*.png      (actual images)                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Non-Blocking Logging**
- All `log*()` calls return instantly
- Background process handles all I/O
- Large queue (10,000 messages) prevents blocking
- Training code runs at full speed

### 2. **Dual-Step Tracking**
Two step columns for maximum flexibility:

| Column | Type | Description | Use Case |
|--------|------|-------------|----------|
| `step` | Auto-increment | Increments on every log call | Timeline, sequential ordering |
| `global_step` | Explicit | Controlled via `board.step()` | Epochs, phases, checkpoints |

**Example:**
```python
for epoch in range(10):
    board.step()  # Set global_step = 0, 1, 2, ...
    for batch in train_loader:
        board.log(loss=...)  # auto_step increments, global_step stays same
```

**Result in Parquet:**
```
step | global_step | batch_loss
-----|-------------|------------
0    | 0           | 1.234       ← Epoch 0, Batch 0
1    | 0           | 1.156       ← Epoch 0, Batch 1
2    | 0           | 1.078       ← Epoch 0, Batch 2
3    | 1           | 0.876       ← Epoch 1, Batch 0
4    | 1           | 0.798       ← Epoch 1, Batch 1
```

### 3. **Efficient Storage (Parquet)**
- **Columnar format:** Fast queries on specific metrics
- **Compression:** Snappy compression saves disk space
- **Schema evolution:** Add new metrics dynamically (NULL for missing values)
- **Type safety:** Proper numeric types (not strings)

**Schema Example:**
```python
# metrics.parquet schema (auto-expands)
{
    "step": int64,
    "global_step": int64 (nullable),
    "loss": float64 (nullable),
    "accuracy": float64 (nullable),
    "learning_rate": float64 (nullable),
    # ... new metrics added as columns automatically
}
```

### 4. **Media Handling**
- Supports: PIL Images, numpy arrays, torch Tensors, file paths
- Automatic conversion to PNG
- Content-addressed storage (deduplication via SHA256)
- Metadata in Parquet, files in `media/` directory

### 5. **Table Logging**
- Custom `Table` class or list of dicts
- Automatic type inference (text, number, boolean)
- JSON serialization in Parquet
- Supports images in table cells

### 6. **stdout/stderr Capture**
- Tee output to both terminal and log file
- Automatic capture on board creation
- Prefixed stderr: `[STDERR] error message`
- Useful for debugging and audit trails

### 7. **Robust Cleanup**
- Automatic cleanup via `atexit` hooks
- Context manager support (`with Board() as board`)
- Graceful shutdown (30s timeout → terminate)
- Final queue drain before exit

---

## File Structure

```
src/kohakuboard/client/
├── __init__.py         # Public API exports
├── board.py            # Main Board class
├── writer.py           # Background LogWriter process
├── storage.py          # Parquet storage backend
├── media.py            # Image conversion & storage
├── table.py            # Table data structure
└── capture.py          # stdout/stderr capture

kohakuboard/            # Data directory (created on first run)
└── <board_id>/
    ├── metadata.json   # Board info, config, timestamps
    ├── data/
    │   ├── metrics.parquet
    │   ├── media.parquet
    │   └── tables.parquet
    ├── media/          # PNG images
    │   └── <name>_<step>_<hash>.png
    └── logs/
        ├── output.log  # Captured stdout/stderr
        └── writer.log  # Writer process internal logs
```

---

## API Reference

### Board

**Constructor:**
```python
Board(
    name: str,                      # Human-readable name
    board_id: Optional[str] = None, # Auto-generated if not provided
    config: Optional[Dict] = None,  # Run configuration
    base_dir: Optional[Path] = None,# Base directory (default: ./kohakuboard)
    capture_output: bool = True,    # Capture stdout/stderr
)
```

**Methods:**
```python
# Scalar logging
board.log(**metrics: Union[int, float])

# Image logging
board.log_images(
    name: str,
    images: Union[Any, List[Any]],  # PIL, numpy, torch, path
    caption: Optional[str] = None,
)

# Table logging
board.log_table(
    name: str,
    table: Union[Table, List[Dict]],
)

# Explicit step control
board.step(increment: int = 1)

# Manual flush (usually not needed)
board.flush()

# Clean shutdown
board.finish()
```

**Properties:**
```python
board.name           # Board name
board.board_id       # Unique ID
board.board_dir      # Path to board directory
board._step          # Current auto-increment step (read-only)
board._global_step   # Current global step (read-only)
```

---

## Examples

### Quick Start
```python
from kohakuboard.client import Board

def main():
    # Create board - auto-finishes on program exit (atexit)
    board = Board(name="my_experiment", config={"lr": 0.001})

    for i in range(1000):
        loss = train_step()
        board.log(loss=loss, step_num=i)

    # board.finish() called automatically on exit

if __name__ == "__main__":
    main()
```

**⚠️ Windows Users:** The `if __name__ == "__main__":` guard is **required** on Windows due to multiprocessing using `spawn` instead of `fork`.

### Training with Explicit Steps
```python
from kohakuboard.client import Board

# Just create the board at the start - no 'with' needed!
board = Board(name="resnet_training", config={"lr": 0.001})

for epoch in range(100):
    board.step()  # global_step = epoch

    for batch in train_loader:
        loss = train_step(batch)
        board.log(batch_loss=loss.item())

    # Epoch metrics
    val_loss = validate()
    board.log(epoch_val_loss=val_loss)

    # Log images
    if epoch % 10 == 0:
        board.log_images("predictions", pred_images[:8])

# Auto-finishes via atexit (or call board.finish() explicitly)
```

### Using Tables
```python
from kohakuboard.client import Table

# From list of dicts
table = Table([
    {"class": "cat", "precision": 0.85, "recall": 0.80},
    {"class": "dog", "precision": 0.88, "recall": 0.85},
])
board.log_table("class_metrics", table)

# With explicit columns
table = Table(
    columns=["Sample", "Precision", "Recall"],
    column_types=["text", "number", "number"],
    rows=[
        ["Cat", 0.85, 0.80],
        ["Dog", 0.88, 0.85],
    ],
)
board.log_table("results", table)
```

---

## Implementation Details

### Non-Blocking Design

**Challenge:** Disk I/O is slow and blocks training code

**Solution:** Multiprocessing

1. Main process: Puts messages in Queue, returns immediately
2. Writer process: Reads from Queue, handles all I/O
3. Large queue (10,000) buffers bursts of logging

**Message Flow:**
```
Training Code → Queue.put() → [instant return] ← Non-blocking!
                   ↓
             Writer Process
                   ↓
          Batch + Flush to Disk
```

### Best-Effort Flushing Strategy

**Challenge:** Need data available for online sync ASAP, but writing to Parquet on every log is expensive

**Solution:** Aggressive immediate flushing (best-effort)

- **Immediate flush after every write** - Data available within milliseconds
- **Background auto-flush every 5 seconds** - Catches any missed flushes
- Trades some performance for data availability
- Perfect for online syncing - data is always fresh

**Flush triggers:**
1. **After every log/media/table append** (immediate, best-effort)
2. **Every 5 seconds** (auto-flush timer)
3. **Explicit `board.flush()` call** (user-requested)
4. **Board shutdown** (`board.finish()`)

This aggressive flushing ensures:
- ✅ Data available for online sync within milliseconds
- ✅ Minimal data loss on crashes
- ✅ Real-time monitoring possible
- ⚠️ Slightly higher I/O overhead (still non-blocking for training code!)

### Schema Evolution

**Challenge:** Metrics logged change over time (e.g., add new metric at epoch 50)

**Solution:** Dynamic schema in Parquet

```python
# Epoch 0-49: Only log loss
board.log(loss=0.5)
# Parquet: [step, global_step, loss]

# Epoch 50+: Add accuracy
board.log(loss=0.3, accuracy=0.95)
# Parquet: [step, global_step, loss, accuracy]
#          Previous rows have NULL for accuracy
```

Pandas/Parquet handles this automatically via `pd.concat(..., sort=False)`.

### Image Deduplication

Images are content-addressed:
```
filename = f"{name}_{step:08d}_{sha256[:8]}.png"
```

- Same image logged multiple times → same hash → deduplicated
- Saves disk space for repeated logging (e.g., validation set)

### Process Cleanup

**Challenge:** Ensure data is flushed even on crashes, Ctrl+C, or exceptions

**Solution:** Multiple safety mechanisms

1. **Signal handlers:** SIGINT (Ctrl+C) and SIGTERM (kill)
2. **Exception hook:** Catches uncaught exceptions via `sys.excepthook`
3. **atexit hooks:** Normal program exit
4. **Context manager:** `with Board() as board` (optional)
5. **Timeout:** Force terminate after 30s if stuck

**Shutdown sequence:**
1. Stop capturing stdout/stderr
2. Signal writer process to stop
3. Wait up to 30 seconds
4. Drain remaining queue messages
5. Flush all buffers
6. Terminate writer process

---

## Performance Characteristics

| Operation | Latency | Throughput |
|-----------|---------|------------|
| `board.log()` | < 1 µs | 100K+ logs/sec |
| `board.log_images()` | < 1 µs | Limited by queue size |
| Queue full (blocking) | Until space | Rare with 10K queue |
| Parquet write | ~10-50 ms | Batched (100 rows) |
| Final flush | ~100-500 ms | On shutdown only |

**Memory usage:**
- Queue: ~80 MB (10K messages × 8 KB average)
- Writer buffers: ~1-10 MB (depends on data)
- Image processing: ~50-200 MB (PIL overhead)

**Disk usage:**
- Parquet: Highly compressed (10-100x vs CSV)
- Images: PNG with optimization (~50-200 KB each)
- Logs: Rotated at 10 MB

---

## Future Enhancements

1. **Online Syncing** (already designed for):
   - Add sync process that reads Parquet and uploads
   - Use file watcher or periodic polling
   - No changes to Board API needed

2. **Compression Tuning**:
   - Switch to `zstd` compression (better ratio)
   - Configurable flush thresholds

3. **Distributed Training Support**:
   - Shard Board by rank
   - Merge on completion

4. **Video Logging**:
   - Same pattern as images
   - FFmpeg for encoding

5. **Offline Mode**:
   - Detect network failures
   - Queue sync operations
   - Retry with exponential backoff

---

## Dependencies

**Required:**
- `pandas` - DataFrame operations
- `pyarrow` - Parquet I/O
- `loguru` - Structured logging

**Optional:**
- `pillow` - Image conversion (for `log_images`)
- `numpy` - Array support
- `torch` - Tensor support

**Install:**
```bash
pip install pandas pyarrow pillow loguru

# Or with all optional
pip install pandas pyarrow pillow loguru numpy torch
```

---

## License

Same as KohakuBoard (parent project)

---

## Contributing

See `CONTRIBUTING.md` in the root repository.
