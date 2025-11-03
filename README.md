# KohakuBoard

High-performance ML experiment tracking with **zero training overhead**.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/KohakuBlueleaf/KohakuBoard)

> **Part of [KohakuHub](https://github.com/KohakuBlueleaf/KohakuHub)** - Self-hosted AI Infrastructure

---

## Quick Start

```bash
pip install -e .
```

```python
from kohakuboard.client import Board

# Create board
board = Board(name="my-experiment", config={"lr": 0.001, "batch_size": 32})

# Training loop
for epoch in range(10):
    for data, target in train_loader:
        loss = train_step(data, target)

        board.step()  # Once per optimizer step
        board.log(loss=loss.item())  # Non-blocking, <0.1ms

# log will be saved under ./kohakuboard with sqlite/lance format
```


---

|![1761752427584](image/README/1761752427584.png)|![1761752450957](image/README/1761752450957.png)|
|-|-|

**Join our community:** https://discord.gg/xWYrkyvJ2s

---

## Why KohakuBoard?

### KohakuBoard's Advantages

- **Zero Training Overhead** - Non-blocking logging returns in <0.1ms
- **Local-First** - No server required during training, view results instantly
- **High Throughput** - 20,000+ metrics/second sustained
- **Rich Data Types** - Scalars, images, videos, tables, histograms
- **WebGL Visualization** - Handle 100K+ datapoints smoothly
- **Self-Hosted** - Your data stays on your infrastructure

---

## Features

### Non-Blocking Architecture

**Background Writer Process** ensures training never waits:

```
Training Script          Background Process
     │                          │
     ├─ board.log(loss=0.5)     │
     │  └─> Queue.put()         │
     │      (<0.1ms return!)    │
     │                          ├─ Queue.get()
     ├─ Continue training...    ├─ Batch write
     │                          └─ Flush to disk
```

**Performance:**
- Log call latency: **<0.1ms**
- Throughput: **20,000+ metrics/sec**
- Queue capacity: **50,000 messages**
- Memory overhead: **~100-200 MB**

### Rich Data Types

**Unified API** for all data types - no step inflation:

```python
board.log(
    loss=0.5,                           # Scalar
    sample_img=Media(image),            # Image
    predictions=Table(results),         # Table
    gradients=Histogram(grads)          # Histogram
)
# All logged at SAME step with 1 queue message!
```

**Supported Types:**
- **Scalars** - Metrics, learning rates, accuracies
- **Media** - Images (PNG/JPG), videos (MP4), audio (WAV)
- **Tables** - Structured data with embedded images
- **Histograms** - Weight/gradient distributions with compression (99.8% size reduction)

### Three-Tier SQLite Storage Architecture

**Powered by [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault)** - A high-performance storage library with dual interfaces over SQLite:

```
Three Specialized SQLite Implementations:

1. KohakuVault KVault        2. KohakuVault ColumnVault     3. Standard SQLite
   (K-V Store)                   (Columnar Storage)             (Relational)
   ├─ Media blobs                ├─ Metrics                     ├─ Media metadata
   ├─ B+Tree index on K          ├─ Histograms                  ├─ Tables
   ├─ Content-addressable        ├─ Blob-based columnar         └─ Step info
   └─ .cache() for bulk ops      └─ Dynamic chunk growth
```

**Why KohakuVault?**
- **Zero dependencies** - Single SQLite file, no external services
- **Simple deployment** - Just .db files, no infrastructure
- **Dual-interface design** - Dict-like for blobs, list-like for sequences
- **High performance** - Native speed with Pythonic API
- **Memory efficient** - Streaming support, dynamic chunk growth
- **True SWMR** - Multiple readers, single writer via SQLite WAL

**Why Three Tiers?**
- **KVault**: Optimized for blob storage with B+Tree index, content-addressable
- **ColumnVault**: Optimized for append-heavy time-series with columnar layout
- **Standard SQLite**: Optimized for structured metadata with ACID guarantees

### Advanced Visualization

**WebGL-Based Charts** powered by Plotly.js:
- Handle **100K+ datapoints** smoothly
- Configurable smoothing (EMA, MA, Gaussian)
- X-axis selection (step, global_step, any metric)
- Multi-metric overlays
- Dark/light mode
- Responsive design

**Rich Viewers:**
- **Histogram Navigator** - Step-by-step distribution exploration
- **Media Viewer** - Image grids, video playback
- **Table Viewer** - Structured data with embedded images
- **Dashboard** - Customizable metric layouts

### Local-First Workflow

```bash
# Train locally
python train.py              # Logs to ./kohakuboard/

# View results (no server required!)
kobo open ./kohakuboard --browser

# Or start server for team sharing
kobo serve --port 48889
```

**No server setup, no configuration, no hassle.**

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from kohakuboard.client import Board

# Create board - automatically saves on program exit
board = Board(name="my-experiment", config={"lr": 0.001, "batch_size": 32})

# Training loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_step(data, target)

        # Increment step once per optimizer step (not per epoch!)
        board.step()

        # Log metrics (non-blocking, returns in <0.1ms)
        board.log(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    # Log validation at end of epoch
    val_loss = validate(model, val_loader)
    board.log(**{"val/loss": val_loss})

# That's it! No .finish() needed - auto-cleanup via atexit
```

### View Results

```bash
# Local viewer (no server)
kobo open ./kohakuboard --browser

# Or start server
kobo serve --port 48889
# Access at http://localhost:48889
```

---

## Complete Example

```python
from kohakuboard.client import Board, Histogram, Table, Media
import torch

# Create board with hyperparameters
board = Board(
    name="cifar10-resnet18",
    config={"lr": 0.001, "batch_size": 128, "epochs": 100, "optimizer": "AdamW"}
)

# Training loop
for epoch in range(100):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Step once per optimizer step
        board.step()

        # Log scalars (non-blocking, <0.1ms)
        board.log(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss, correct, predictions_table = 0, 0, []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

            # Sample predictions for table (first batch only)
            if batch_idx == 0:
                for i in range(min(8, len(data))):
                    predictions_table.append({
                        "image": Media(data[i].cpu().numpy()),
                        "true": class_names[target[i]],
                        "pred": class_names[pred[i]],
                        "correct": "✓" if pred[i] == target[i] else "✗"
                    })

    # Log validation (scalars + table + histograms - all at same step!)
    hist_data = {f"grad/{n}": Histogram(p.grad) for n, p in model.named_parameters() if p.grad is not None}
    board.log(**{
        "val/loss": val_loss / len(val_loader),
        "val/accuracy": correct / len(val_loader.dataset),
        "val/predictions": Table(predictions_table),
        **hist_data
    })

# No .finish() needed - automatic cleanup when script exits
```

---

## Architecture

### Client (Training Script)

```
Main Process (Training)          Background Writer Process
       │                                   │
       ├─ board.log(loss=0.5)              │
       │  └─> Queue.put()                  │
       │      (returns instantly!)         │
       │                                   ├─ Queue.get()
       │                                   ├─ Process batch
       ├─ Continue training...             ├─ Write to storage
       │                                   └─ Flush to disk
```

**Key Features:**
- **Non-blocking**: `log()` returns in <0.1ms
- **Message Queue**: 50,000 message capacity
- **Writer Process**: Background process drains queue
- **Storage Layer**: Three-tier SQLite architecture (KohakuVault KVault + ColumnVault + Standard SQLite)
- **Graceful Shutdown**: atexit hooks + signal handlers ensure no data loss

### Backend (Visualization Server)

```
FastAPI Backend (Port 48889)
    ↓ Read-only connections
Board Files (./kohakuboard/)
    ├── {board_id}/
    │   ├── metadata.json
    │   ├── data/           ← SQL/columnar queries here
    │   │   ├── metrics/    ← KohakuVault DB files
    │   │   └── metadata.db ← SQLite database
    │   └── media/
    │       └── *.png, *.mp4
        ↓ REST API
Vue 3 Frontend (WebGL Charts)
```

**Key Features:**
- **Zero-copy serving**: Reads files directly (no database)
- **Concurrent reads**: Multiple connections supported
- **Fast queries**: Columnar storage for metrics
- **Static serving**: Media files served directly

---

## Data Model

### Directory Structure

```
./kohakuboard/
└── {board_id_timestamp}/
    ├── metadata.json           # Board info, config, timestamps
    ├── data/                   # Storage backend files
    │   ├── metrics/            # (hybrid) KohakuVault columnar files
    │   │   ├── train__loss.db
    │   │   ├── val__accuracy.db
    │   │   └── ...
    │   ├── metadata.db         # (hybrid) SQLite metadata
    │   └── histograms/
    │       ├── gradients_i32.db  # int32 precision
    │       └── params_u8.db      # uint8 precision (compact)
    ├── media/                  # Content-addressed storage
    │   ├── {name}_{idx}_{step}_{sha256}.png
    │   ├── {name}_{idx}_{step}_{sha256}.mp4
    │   └── {name}_{idx}_{step}_{sha256}.wav
    └── logs/
        ├── output.log          # Captured stdout/stderr
        └── writer.log          # Writer process logs
```

### Metadata Schema

```json
{
  "board_id": "20250129_150423_abc123",
  "name": "cifar10-resnet18",
  "config": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 100
  },
  "created_at": "2025-01-29T15:04:23",
  "finished_at": "2025-01-29T18:32:45",
  "status": "finished",
  "backend": "hybrid",
  "version": "0.0.1"
}
```

---

## CLI Tool

```bash
# Open local viewer (no server)
kobo open ./kohakuboard --browser

# Start backend server
kobo serve --port 48889 --host 0.0.0.0

# Sync to remote (WIP)
kobo sync board_id -r https://kohakuboard.example.com
```

---

## Configuration

### Storage Backends

```python
# Hybrid (default, recommended)
board = Board(name="exp", backend="hybrid")
# - Metrics: KohakuVault (fastest)
# - Media/Tables: SQLite (best concurrency)

# DuckDB
board = Board(name="exp", backend="duckdb")
# - NaN/inf preservation
# - SQL query support

# Parquet
board = Board(name="exp", backend="parquet")
# - Maximum compatibility
# - Good for post-processing
```

### Advanced Options

```python
board = Board(
    name="experiment",
    board_id="custom-id",           # Auto-generated if not provided
    config={"lr": 0.001},           # Hyperparameters
    base_dir="./my-boards",         # Custom directory
    backend="hybrid",               # Storage backend
    capture_output=True,            # Capture stdout/stderr
)
```

### Context Manager

```python
with Board(name="experiment") as board:
    board.log(loss=0.5)
    # Automatic flush() and finish() on exit
```

---

## API Reference

### Board

```python
Board(
    name: str,                      # Human-readable name
    board_id: str | None = None,    # Auto-generated if not provided
    config: dict | None = None,     # Hyperparameters, etc.
    base_dir: Path | None = None,   # Default: ./kohakuboard
    backend: str = "hybrid",        # Storage backend
    capture_output: bool = True,    # Capture stdout/stderr
)
```

**Methods:**

**`board.step()`** - Increment global_step

```python
for epoch in range(10):
    board.step()  # global_step += 1
    # ... train and log
```

**`board.log(**metrics)`** - Log data (non-blocking)

```python
board.log(
    loss=0.5,
    accuracy=0.95,
    learning_rate=0.001,
)

# Namespaces (creates tabs in UI)
board.log(**{
    "train/loss": 0.5,
    "val/accuracy": 0.95
})
```

**`board.flush()`** - Force flush (blocks until complete)

```python
board.flush()  # Wait for all pending writes
```

**`board.finish()`** - Manual cleanup (auto-called on exit)

```python
board.finish()  # Flush buffers, close connections
```

### Data Types

#### Media

```python
from kobo.client.types import Media

# Images
board.log(
    sample_img=Media(image_array),  # numpy, PIL, torch tensor
    prediction=Media(pred_img, caption="Predicted: cat")
)

# Video
board.log(
    training_video=Media("output.mp4", media_type="video")
)

# Audio (if supported)
board.log(
    audio_sample=Media("sample.wav", media_type="audio")
)
```

#### Table

```python
from kobo.client.types import Table

# From list of dicts
results = Table([
    {"name": "Alice", "score": 95, "pass": True},
    {"name": "Bob", "score": 87, "pass": True},
])
board.log(student_results=results)

# Tables with embedded images
predictions = Table([
    {"image": Media(img), "label": "cat", "confidence": 0.95},
    {"image": Media(img2), "label": "dog", "confidence": 0.87},
])
board.log(val_predictions=predictions)
```

#### Histogram

```python
from kobo.client.types import Histogram

# Log gradient distributions
board.log(
    gradients=Histogram(param.grad),
    weights=Histogram(param.data)
)

# Precompute for efficiency (optional)
hist = Histogram(gradients).compute_bins()
board.log(grad_distribution=hist)

# Compact precision (75% size reduction, ~1% accuracy loss)
hist = Histogram(gradients, precision="compact")
board.log(grad_distribution=hist)
```

---

## Deployment

### Local Mode (Recommended)

```bash
# Install
pip install -e .

# Train
python train.py

# View results
kobo open ./kohakuboard --browser
```

### Remote Mode (WIP)

```bash
# Docker deployment
docker-compose -f docker-compose.kohakuboard.yml up -d

# Access at http://localhost:28081
```

See [docs/kohakuboard/](./docs/kohakuboard/) for complete deployment guides.

---

## Comparison with Alternatives

| Feature | WandB | TensorBoard | MLflow | KohakuBoard |
|---------|-------|-------------|--------|-------------|
| **Latency** | ~10ms | ~1ms | ~5ms | **<0.1ms** |
| **Throughput** | ~1K/sec | ~10K/sec | ~5K/sec | **20K+/sec** |
| **Offline** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **File-Based** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Non-Blocking** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **SQL Queries** | ❌ No | ❌ No | ✅ Yes | ✅ Yes (DuckDB) |
| **WebGL Charts** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **100K+ Points** | Slow | Slow | Slow | **Fast** |
| **Self-Hosted** | Limited | ✅ Yes | ✅ Yes | ✅ Yes |
| **Setup** | Cloud | Local | Server | **None** |

---

## Documentation

- [Getting Started](./docs/kohakuboard/getting-started.md) - Installation and basic usage
- [API Reference](./docs/kohakuboard/api.md) - Complete API documentation
- [Architecture](./docs/kohakuboard/architecture.md) - System design and internals
- [CLI Guide](./docs/kohakuboard/cli.md) - Command-line tool usage
- [Examples](./examples/) - Real-world usage patterns

---

## Examples

See `examples/` directory:

- `kohakuboard_basic.py` - Simple scalar logging
- `kohakuboard_all_media_types.py` - Images, videos, tables
- `kohakuboard_cifar_training.py` - Complete CIFAR-10 training example
- `kohakuboard_media_in_tables.py` - Tables with embedded images
- `kohakuboard_histogram_logging.py` - Gradient distribution tracking

---

## Roadmap

### ✅ Complete

**Client Library:**
- Non-blocking logging architecture
- Rich data types (scalars, media, tables, histograms)
- Three-tier SQLite architecture (KohakuVault KVault + ColumnVault + Standard SQLite)
- Alternative backends (DuckDB, Parquet)
- Graceful shutdown with queue draining
- Content-addressed media storage

**Backend & UI:**
- FastAPI REST API
- Vue 3 interface with dark/light mode
- WebGL charts (100K+ points)
- Histogram navigator
- Media/table viewers
- CLI tool (`kobo`)

### 🚧 In Progress

- Remote server mode with authentication
- Sync protocol for uploading local boards
- Project management (group related boards)
- Run comparison UI (side-by-side metrics)
- Real-time streaming (live updates while training)

### 📋 Planned

**Client Features:**
- PyTorch Lightning integration
- Keras callback
- Hugging Face Trainer integration
- Custom callback system

**Backend Features:**
- Multi-board comparison API
- Advanced filtering (tags, date range)
- Export to CSV/JSON
- Aggregation queries

**UI Features:**
- Diff viewer (compare runs)
- Scatter plots (metric vs metric)
- Custom dashboards
- Annotations
- Search and filter

**Infrastructure:**
- Docker/Kubernetes deployment
- Cloud storage backends (S3, GCS)
- Multi-user authentication


---

## License

KohakuBoard is a multi-component project with different licenses:

- **Client Library** (`kohakuboard`): [Apache License 2.0](src/kohakuboard/LICENSE)
  - Free for commercial and non-commercial use
  - Permissive license with minimal requirements

- **Web UI** (`kohaku-board-ui`): [AGPL-3.0](src/kohaku-board-ui/LICENSE)
  - Free to use and modify
  - Source code disclosure required for network services

- **Server** (`kohakuboard_server`): [Kohaku Software License 1.0](src/kohakuboard_server/LICENSE)
  - Free for non-commercial use
  - Free for commercial use under revenue/duration limits
  - Commercial licenses available for larger deployments

**Commercial Licensing:** For commercial licenses or exemptions, contact kohaku@kblueleaf.net

See [LICENSE](./LICENSE) for complete details.

---

## Contributing

KohakuBoard is part of the KohakuHub ecosystem. We welcome contributions!

**Before contributing:**
- Read [CONTRIBUTING.md](./CONTRIBUTING.md) for code style and guidelines
- Join our [Discord](https://discord.gg/xWYrkyvJ2s) for discussions
- Check [open issues](https://github.com/KohakuBlueleaf/KohakuHub/issues) tagged with `kohakuboard`

**Areas we need help:**
- 🎨 Frontend (chart improvements, UI/UX)
- 🔧 Backend (storage backends, performance)
- 📊 Client library (framework integrations)
- 📚 Documentation (tutorials, guides)
- 🧪 Testing (unit tests, benchmarks)

---

## Support

- **Discord:** https://discord.gg/xWYrkyvJ2s (Use #kohakuboard channel)
- **Issues:** https://github.com/KohakuBlueleaf/KohakuHub/issues (Tag with `kohakuboard`)
- **Email:** kohaku@kblueleaf.net

---

## Acknowledgments

- [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault) - High-performance storage library with dual SQLite interfaces (KVault for blobs, ColumnVault for sequences)
- [DuckDB](https://duckdb.org/) - Alternative storage backend
- [Plotly.js](https://plotly.com/javascript/) - WebGL charts
- [Vue 3](https://vuejs.org/) - Modern UI framework
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework

---

**Production Ready!** Core features are stable and performant. Use in real training workflows and help us improve.
