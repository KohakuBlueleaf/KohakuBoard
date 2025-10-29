# KohakuBoard - Lightweight ML Experiment Tracking

**Non-blocking, file-based experiment tracking with DuckDB backend**

**Status:** ✅ Production Ready
**License:** Kohaku Software License 1.0 (Non-Commercial with Trial)

---

## Overview

KohakuBoard is a lightweight experiment tracking system designed for ML/AI training workflows. Unlike WandB or MLflow, it uses a **file-based approach** with DuckDB for storage, making it perfect for:

- **Local development** - No server required during training
- **Offline training** - Works without internet
- **High throughput** - Non-blocking logging (< 1µs per log call)
- **Simple deployment** - Just point backend at a directory
- **Self-hosted** - Your data stays on your infrastructure

---

## Quick Start

### Installation

```bash
pip install kohakuhub[kohakuboard]
# Or from source
pip install -e ".[kohakuboard]"
```

### Basic Usage

```python
from kohakuboard.client import Board

# Create board (auto-finishes on exit)
board = Board(name="my_training", config={"lr": 0.001, "batch_size": 32})

# Training loop
for batch_idx, (data, target) in enumerate(dataloader):
    # Step-wise logging (per optimizer step)
    board.step()  # global_step tracks optimizer steps

    loss = train_step(data, target)

    # Log metrics (non-blocking, <1µs)
    board.log(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

# Board auto-saves on exit
```

### View Results

```bash
# Start backend
python -m kohakuboard.main

# Open browser
# http://localhost:48889
```

---

## Core Concepts

### Two Types of Steps

KohakuBoard tracks TWO types of steps:

| Step Type | What It Tracks | When It Increments | Use Case |
|-----------|----------------|---------------------|----------|
| **`_step`** (auto) | Timeline of log events | Every `board.log()` call | Debugging, finding when things were logged |
| **`global_step`** (explicit) | Training progress | Every `board.step()` call | **Optimizer steps / batch updates** |

### ⚠️ IMPORTANT: `global_step` is for OPTIMIZER STEPS, not epochs!

```python
# ✅ CORRECT: global_step = optimizer steps
for batch in dataloader:
    board.step()  # Increment ONCE per batch/optimizer step
    loss = train_step(batch)
    board.log(loss=loss, epoch=current_epoch)  # Epoch is just a metric!

# ❌ WRONG: Don't use global_step for epochs
for epoch in range(10):
    board.step()  # NO! This makes global_step = epoch number
    for batch in dataloader:
        # ...
```

### Why This Design?

1. **`global_step` = optimizer steps** is the **standard in ML**:
   - TensorBoard uses `step` for optimizer steps
   - WandB uses `step` for training steps
   - PyTorch Lightning uses `global_step` for optimizer steps
   - Hugging Face Transformers logs per optimizer step

2. **Epochs are just metadata**:
   - Different runs may have different epoch lengths
   - What matters is **how many gradient updates** the model has seen
   - Epoch numbers should be logged as a regular metric

3. **`_step` exists for event ordering**:
   - Sometimes you log multiple things at the same `global_step`
   - `_step` gives you the exact order of log calls
   - Useful for debugging and timeline visualization

---

## Architecture

### Client (Training Script)

```
Training Script
    ↓ board.log(loss=0.5) [<1µs, non-blocking]
Multiprocessing Queue (10K capacity)
    ↓ Background process
LogWriter Process
    ↓ Batch writes every 10 items
DuckDB File (board.duckdb)
    ↓ Columnar storage, true incremental append
File System (./kohakuboard/{board_id}/)
```

**Key Features:**
- **Non-blocking**: `log()` returns in < 1µs
- **Queue**: 10,000 message buffer
- **Batch writes**: Writes every 10 metrics (configurable)
- **True incremental**: DuckDB's `connection.append()` - no read overhead
- **Automatic flush**: Every 5 seconds + on shutdown

### Backend (Visualization Server)

```
FastAPI Backend (Port 48889)
    ↓ Read-only DuckDB connections
Board Files (./kohakuboard/)
    ├── {board_id}/
    │   ├── metadata.json
    │   ├── data/board.duckdb  ← SQL queries here
    │   └── media/*.png
    └── ...
        ↓ REST API
Vue 3 Frontend (WebGL Charts)
```

**Key Features:**
- **Zero-copy serving**: Reads DuckDB files directly
- **Concurrent reads**: Multiple connections supported
- **SQL queries**: Efficient columnar storage
- **Static file serving**: Media files served directly

---

## Data Model

### DuckDB Schema

```sql
-- Metrics table (dynamic columns added automatically)
CREATE TABLE metrics (
    step BIGINT NOT NULL,           -- Auto-increment (_step)
    global_step BIGINT,             -- Explicit step (optimizer steps)
    -- Dynamic columns added as you log:
    loss DOUBLE,
    accuracy DOUBLE,
    learning_rate DOUBLE,
    epoch INTEGER,
    -- ... any metric you log
);

-- Media table (images, videos, audio)
CREATE TABLE media (
    step BIGINT NOT NULL,
    global_step BIGINT,
    name VARCHAR NOT NULL,
    caption VARCHAR,
    media_id VARCHAR,
    type VARCHAR,
    filename VARCHAR,
    path VARCHAR,
    size_bytes BIGINT,
    format VARCHAR,
    width INTEGER,
    height INTEGER
);

-- Tables table (structured data)
CREATE TABLE tables (
    step BIGINT NOT NULL,
    global_step BIGINT,
    name VARCHAR NOT NULL,
    columns VARCHAR,          -- JSON array
    column_types VARCHAR,     -- JSON array
    rows VARCHAR             -- JSON array of arrays
);
```

### Directory Structure

```
./kohakuboard/
└── {board_id_timestamp}/
    ├── metadata.json           # Board info, config, timestamps
    ├── data/
    │   └── board.duckdb        # Single DuckDB file
    ├── media/
    │   ├── {name}_{idx}_{step}_{hash}.png
    │   ├── {name}_{idx}_{step}_{hash}.mp4
    │   └── {name}_{idx}_{step}_{hash}.wav
    └── logs/
        ├── output.log          # Captured stdout/stderr
        └── writer.log          # Writer process logs
```

---

## API Reference

### Board

```python
Board(
    name: str,                    # Human-readable name
    board_id: Optional[str] = None,  # Auto-generated if not provided
    config: Optional[Dict] = None,   # Hyperparameters, etc.
    base_dir: Optional[Path] = None,  # Default: ./kohakuboard
    capture_output: bool = True,     # Capture stdout/stderr
    backend: str = "duckdb"          # Storage backend
)
```

#### Methods

**`board.step()`** - Increment global_step (call once per optimizer step)

```python
for batch in dataloader:
    board.step()  # global_step += 1
    # ... train and log
```

**`board.log(**metrics)`** - Log scalar metrics (non-blocking)

```python
board.log(
    loss=0.5,
    accuracy=0.95,
    learning_rate=0.001,
    epoch=5,  # Epochs are just metrics!
    batch_idx=100
)
```

**`board.log_images(name, images, caption=None)`** - Log images

```python
# List of numpy arrays (H, W, C) or PIL Images
board.log_images(
    "predictions",
    [img1, img2, img3],
    caption="Model predictions at step 1000"
)
```

**`board.log_table(name, table)`** - Log structured data

```python
from kohakuboard.client import Table

table = Table([
    {"class": "cat", "precision": 0.85, "recall": 0.80},
    {"class": "dog", "precision": 0.88, "recall": 0.85},
])
board.log_table("class_metrics", table)
```

**`board.finish()`** - Manual cleanup (auto-called on exit)

```python
board.finish()  # Flush buffers, close connections
```

---

## Usage Patterns

### Pattern 1: Standard Training Loop

```python
from kohakuboard.client import Board

board = Board(name="resnet_training", config={"lr": 0.001, "batch_size": 32})

for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Step-wise logging
        board.step()  # global_step = total optimizer steps

        loss = train_step(data, target)

        # Log per-batch metrics
        board.log(
            loss=loss.item(),
            lr=optimizer.param_groups[0]['lr'],
            epoch=epoch  # Epoch is a metric!
        )

    # Log per-epoch metrics (same global_step as last batch)
    val_loss, val_acc = validate()
    board.log(
        val_loss=val_loss,
        val_accuracy=val_acc
    )

# Auto-finishes on exit
```

### Pattern 2: Multiple Logs Per Step

Sometimes you want to log multiple things at the same optimizer step:

```python
for batch in dataloader:
    board.step()  # global_step += 1

    # Forward pass
    loss = model(batch)

    # Log loss immediately
    board.log(loss=loss.item())

    # Backward pass
    loss.backward()
    grad_norm = compute_grad_norm()

    # Log gradient norm (same global_step, different _step)
    board.log(grad_norm=grad_norm)

    # Optimizer step
    optimizer.step()
```

Both logs share the same `global_step`, but have different `_step` values (auto-incremented). This preserves the order of events.

### Pattern 3: Epoch Boundaries

```python
for epoch in range(10):
    for batch_idx, batch in enumerate(train_loader):
        board.step()  # global_step tracks batches, not epochs!

        loss = train_step(batch)
        board.log(
            loss=loss.item(),
            epoch=epoch,  # Current epoch (just a metric)
            batch_in_epoch=batch_idx  # Optional: track position in epoch
        )

    # Epoch summary (same global_step as last batch)
    board.log(
        epoch_train_loss=epoch_avg_loss,
        epoch_val_loss=val_loss,
        epoch_complete=True  # Marker for epoch boundary
    )
```

### Pattern 4: Variable Batch Sizes

With gradient accumulation or variable batch sizes:

```python
accumulation_steps = 4
optimizer_step_count = 0

for batch_idx, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    # Log per micro-batch
    board.log(
        micro_batch_loss=loss.item(),
        micro_batch_idx=batch_idx,
        optimizer_step=optimizer_step_count
    )

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

        # Increment global_step AFTER optimizer step
        board.step()
        optimizer_step_count += 1
```

---

## Visualization

### Frontend Features

- **WebGL Charts**: Handle 100K+ data points smoothly
- **Smoothing**: EMA, Gaussian, SMA algorithms
- **X-Axis Selection**: Choose `step`, `global_step`, or any metric
- **Multiple Metrics**: Overlay multiple metrics on one chart
- **Media Viewer**: Step-synchronized image/video viewing
- **Table Viewer**: View logged tables with image columns
- **Dark Mode**: Built-in dark mode support
- **Responsive**: Mobile-friendly design

### Default Behavior

- If `global_step` has non-zero values → Charts default to `global_step` x-axis
- If all `global_step` values are 0 → Charts default to `_step` x-axis
- User can manually override via chart settings

---

## Performance

### Client Performance

| Metric | Value |
|--------|-------|
| `board.log()` latency | < 1 µs |
| Throughput | 100K+ logs/sec |
| Queue capacity | 10K messages (80 MB) |
| Memory overhead | ~50 MB |
| DuckDB append (1K rows) | ~10 ms |
| Final flush | ~100-500 ms |

### Backend Performance

| Operation | Performance |
|-----------|-------------|
| List boards | ~10ms (file system scan) |
| Load summary | ~50ms (DuckDB query) |
| Load 10K metric points | ~100ms |
| Load 100K metric points | ~500ms |
| Serve media file | Instant (static file) |

---

## Backend API

### Experiments API (Compatibility)

```bash
GET  /api/experiments                           # List all boards
GET  /api/experiments/{id}                      # Get board details
GET  /api/experiments/{id}/summary              # Summary with available data
GET  /api/experiments/{id}/scalars/{metric}     # Get scalar time series
GET  /api/experiments/{id}/media/{name}         # Get media log entries
GET  /api/experiments/{id}/tables/{name}        # Get table data
```

### Boards API (Direct Access)

```bash
GET  /api/boards                                # List all boards
GET  /api/boards/{id}/metadata                  # Get metadata.json
GET  /api/boards/{id}/summary                   # Full summary
GET  /api/boards/{id}/scalars                   # List available metrics
GET  /api/boards/{id}/scalars/{metric}          # Get metric data
GET  /api/boards/{id}/media                     # List media logs
GET  /api/boards/{id}/media/{name}              # Get media entries
GET  /api/boards/{id}/tables                    # List table logs
GET  /api/boards/{id}/tables/{name}             # Get table data
GET  /api/boards/{id}/media/files/{filename}    # Serve media file
```

---

## Configuration

### Environment Variables

```bash
# Backend configuration
export KOHAKU_BOARD_HOST="0.0.0.0"
export KOHAKU_BOARD_PORT="48889"
export KOHAKU_BOARD_DATA_DIR="./kohakuboard"

# Client configuration (in code)
board = Board(
    name="my_training",
    base_dir="./my_boards",  # Override default
    backend="duckdb",         # or "parquet" (legacy)
    capture_output=True       # Capture stdout/stderr
)
```

---

## FAQ

### Q: When should I call `board.step()`?

**A: Once per optimizer step (batch update).** If you're using gradient accumulation, call it after the actual optimizer step, not per micro-batch.

### Q: What if I never call `board.step()`?

**A:** All `global_step` values will be 0. Charts will default to using `_step` for x-axis, which is fine for simple cases.

### Q: Can I log epoch numbers?

**A:** Yes! Log them as a regular metric: `board.log(epoch=5)`. Don't use `global_step` for epochs.

### Q: What's the difference between `step` and `global_step` in the frontend?

**A:**
- `step` = `_step` in backend = timeline of log events
- `global_step` = optimizer steps (what you increment with `board.step()`)

### Q: How do I compare runs?

**A:** Multiple boards are automatically listed in the frontend. Use the experiments list page to view all boards.

### Q: Can I delete old boards?

**A:** Yes, just delete the directory: `rm -rf kohakuboard/{board_id}`

### Q: How do I deploy the backend?

**A:** See deployment docs. TL;DR: `docker-compose up` or `python -m kohakuboard.main`

---

## Examples

See `examples/` directory:

- `kohakuboard_basic.py` - Simple scalar logging
- `kohakuboard_advanced.py` - Images and tables
- `kohakuboard_explicit_steps.py` - Step tracking patterns
- `kohakuboard_media_in_tables.py` - Tables with image columns
- `kohakuboard_all_media_types.py` - Images, videos, audio

---

## Comparison with Alternatives

| Feature | WandB | TensorBoard | MLflow | KohakuBoard |
|---------|-------|-------------|--------|-------------|
| **Latency** | ~10ms | ~1ms | ~5ms | < 1µs |
| **Offline** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **File-based** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **SQL Queries** | ❌ No | ❌ No | ✅ Yes | ✅ Yes (DuckDB) |
| **WebGL Charts** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Non-blocking** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Self-hosted** | Limited | ✅ Yes | ✅ Yes | ✅ Yes |

---

## License

**Kohaku Software License 1.0** (Non-Commercial with Trial)

- Free for non-commercial use
- Commercial trial: 3 months OR $25k revenue/year
- After trial: Commercial license required
- Source code available

---

## Contributing

KohakuBoard is part of the KohakuHub project.

**Questions?** kohaku@kblueleaf.net
**Issues:** https://github.com/KohakuBlueleaf/KohakuHub/issues

---

## Roadmap

**Current (v0.1):**
- ✅ DuckDB backend
- ✅ Non-blocking logging
- ✅ WebGL visualization
- ✅ Media logging
- ✅ Table logging

**Next (v0.2):**
- [ ] Histogram logging
- [ ] Run comparison UI
- [ ] Export to CSV/JSON
- [ ] PyTorch/TensorFlow callbacks

**Future (v1.0):**
- [ ] Remote sync (upload to server)
- [ ] Multi-user support
- [ ] Advanced analysis tools
- [ ] KohakuHub integration
