---
title: KohakuBoard Documentation
description: High-performance experiment tracking for ML workflows
icon: i-carbon-analytics
---

# KohakuBoard Documentation

High-performance, non-blocking experiment logging library for machine learning.

---

## 🎯 What is KohakuBoard?

KohakuBoard is a **local-first experiment tracking system** designed for ML/AI training workflows. It provides:

- **Non-blocking logging** - Background writer process, zero training overhead
- **Rich data types** - Scalars, images, videos, histograms, tables
- **Unified storage** - KohakuVault ColumnVault + SQLite metadata (identical on train + server)
- **Local-first** - View experiments locally with `kobo open`, no server needed
- **Manual remote sync** - Copy run folders directly into any server's data directory

---

## 📚 Documentation Sections

### [Getting Started](/docs/kohakuboard/getting-started)

Quick start guide, first experiment, core features

**Topics:**
- Installation
- Your first experiment
- Rich data types (scalars, media, tables, histograms)
- No step inflation
- Complete example

**Start here** if you're new to KohakuBoard.

---

### [Python API Reference](/docs/kohakuboard/api)

Complete API documentation for the Python client

**Topics:**
- `Board` class constructor and methods
- Logging methods (`log()`, `log_images()`, `log_histogram()`, etc.)
- Data types (`Media`, `Table`, `Histogram`)
- Step management
- Lifecycle management

**Use this** as a reference while coding.

---

### [CLI Tools](/docs/kohakuboard/cli)

Command-line interface for managing boards

**Topics:**
- `kobo open` - Browse local boards (no auth)
- `kobo-serve` - Authenticated FastAPI server (still stabilizing)
- Manual file copy workflow (recommended sync method today)
- Environment variables
- Systemd service setup

**Use this** to view your experiments locally.

---

### [Configuration](/docs/kohakuboard/configuration)

Storage backends, performance tuning, advanced configuration

**Topics:**
- KohakuVault ColumnVault + SQLite layout
- Performance optimization and queue sizing
- Directory structure
- Memory mode vs. on-disk mode

**Use this** to optimize for your use case.

---

### [Server Setup](/docs/kohakuboard/server) ⚠️ WIP

Remote server deployment (work in progress)

**Topics:**
- Server architecture
- Database setup (PostgreSQL)
- Authentication
- Frontend overview

⚠️ **Note:** Remote mode is still stabilizing. Until the sync APIs are refreshed, move runs by copying the `{project}/{run_id}` folders into the server's `--data-dir`.

---

### [Usage Manual](/docs/kohakuboard/usage-manual)

Hands-on workflow for capturing, inspecting, and sharing boards with your team.

**Topics:**
- Training loop checklist
- Recommended logging cadence
- Manual rsync/copy steps for sharing runs
- Troubleshooting queue pressure, disk usage, and metadata issues

---

## 🚀 Quick Start

### Installation

```bash
cd /path/to/KohakuHub
pip install -e src/kohakuboard/
```

### Log Your First Experiment

```python
from kohakuboard.client import Board

board = Board(name="my-experiment", project="vision")

for batch in train_loader:
    loss = train_step(batch)
    board.step()              # optimizer step
    board.log(loss=loss.item())
```

### View Results

```bash
kobo open ./kohakuboard --browser
```

---

## 🎓 Key Features

### 1. No Step Inflation

**Problem:**
```python
# ❌ BAD: Each histogram call increments step
for name, param in model.named_parameters():
    board.log_histogram(f"grad/{name}", param.grad)
# Result: 50 histograms = 50 different steps!
```

**Solution:**
```python
# ✅ GOOD: All histograms share same step
grad_data = {
    f"grad/{name}": Histogram(param.grad)
    for name, param in model.named_parameters()
}
board.log(**grad_data)
# Result: 50 histograms = 1 step!
```

### 2. Mixed Type Logging

```python
board.log(
    loss=0.5,                      # Scalar
    sample_img=Media(image),       # Image
    results=Table(data),           # Table
    gradients=Histogram(grads)     # Histogram
)
# All at the SAME step!
```

### 3. Non-Blocking Performance

```python
board.log(loss=0.5)  # Returns immediately!
# Background writer handles disk I/O
```

### 4. Local-First Workflow

```bash
# No server needed!
python train.py              # Log experiments
kobo open ./kohakuboard      # View locally
```

---

## 📊 Supported Data Types

| Type | Description | Example |
|------|-------------|---------|
| **Scalars** | Metrics (loss, accuracy, etc.) | `board.log(loss=0.5)` |
| **Media** | Images, videos, audio | `board.log(img=Media(array))` |
| **Tables** | Structured data (can embed Media) | `board.log(results=Table(data))` |
| **Histograms** | Distributions with compression | `board.log(grad=Histogram(values))` |
| **TensorLog** | High-dimensional tensors | `board.log(weights=TensorLog(tensor))` |
| **KernelDensity** | KDE results or raw samples | `board.log(density=KernelDensity(values))` |

---

## 🏗️ Architecture

### Local Mode (Current, Fully Working)

```
Python Script                    Local Viewer
     │                                │
     ├─ Board.log(...)                │
     │  └─> Queue (non-blocking)      │
     │                                │
Writer Process                        │
     ├─ Drain queue                   │
     ├─ Write to KohakuVault column stores + SQLite
     └─ Flush to disk                 │
                                      │
                    kobo open ./kohakuboard
                                      │
                                FastAPI Server
                                      │
                                   Vue UI
```

**Benefits:**
- ✅ No authentication needed
- ✅ Direct file access
- ✅ Works offline
- ✅ Fast and simple

### Remote Mode (WIP, Not Fully Usable)

```
Python Script              Remote Server              Web UI
     │                          │                       │
     ├─ Board.log(...)          │                       │
     │  └─> Local storage       │                       │
     │                          │                       │
rsync / copy board folder  ─────┤                       │
                                │                       │
                          FastAPI + Auth (kobo-serve)   │
                                │                       │
                          PostgreSQL / SQLite metadata  │
                                │                       │
                                └───────> View ─────────┤
```

**Status:** ⚠️ Work in progress
- ✅ Manual sync by copying `{project}/{run}` folders into the server data dir
- ⏳ Authenticated multi-user flows (kobo-serve)
- ⏳ New HTTP sync protocol (will replace legacy DuckDB uploader)
- ⏳ Frontend integration for multi-project dashboards

---

## 🔄 Workflow Comparison

### Local Workflow (Recommended)

```bash
# 1. Install
pip install -e src/kohakuboard/

# 2. Log experiments
python train.py

# 3. View locally
kobo open ./kohakuboard --browser
```

**Pros:**
- ✅ Simple and fast
- ✅ No server setup
- ✅ Works offline
- ✅ Full control

**Cons:**
- ❌ No multi-user collaboration
- ❌ No remote access

### Remote Workflow (Manual Copy Today)

```bash
# 1. Start server (once)
kobo-serve --db postgresql://... --workers 4 --data-dir /var/kohakuboard

# 2. Train locally
python train.py

# 3. Copy run folder into the server's data dir
rsync -a ./kohakuboard/default/20250201_120301_xyz \
      server:/var/kohakuboard/default/

# 4. Refresh the UI
```

**Pros:**
- ✅ Team collaboration
- ✅ Remote access
- ✅ Centralized storage

**Cons:**
- ❌ Requires server setup
- ❌ Authentication config still stabilizing
- ⚠️ Command-line sync (`kobo sync`) is not compatible with current storage yet

---

## 💡 Best Practices

### Logging Frequency

```python
# ✅ DO: Log scalars every batch
for batch in train_loader:
    board.log(loss=loss)

# ✅ DO: Log histograms every N epochs
if epoch % 10 == 0:
    board.log(**histogram_data)

# ❌ DON'T: Log media every batch
for batch in train_loader:
    board.log(img=Media(batch[0]))  # Too frequent!
```

### Namespace Organization

```python
# ✅ DO: Use namespaces for organization
board.log(**{
    "train/loss": 0.5,
    "train/lr": 0.001,
    "val/accuracy": 0.95,
    "val/loss": 0.3
})

# Creates tabs: train/, val/
```

### Histogram Optimization

```python
# ✅ DO: Precompute if CPU available
hist = Histogram(gradients).compute_bins()
board.log(grad=hist)

# ✅ DO: Use compact precision for large datasets
hist = Histogram(values, precision="compact")  # 75% smaller
board.log(weights=hist)
```

---

## 🔧 Storage Architecture

Modern boards always use the hybrid KohakuVault + SQLite layout:

1. **ColumnVault per metric (`data/metrics/*.db`)** – columnar blobs for scalars/histograms with SWMR safety.
2. **SQLite metadata (`data/metadata.db`)** – tables, tensors, steps, namespaces.
3. **KVault media store (`media/blobs.db` + files)** – content-addressed images/video/audio/tensors.

Because every layer is plain SQLite, copying `{project}/{run}` directories between machines is safe and fast. Legacy DuckDB/Parquet backends have been removed from the training client.

---

## 📖 Examples

### CIFAR-10 Training

See [examples/kohakuboard_cifar_training.py](https://github.com/KohakuBlueleaf/KohakuHub/blob/main/examples/kohakuboard_cifar_training.py) for a complete example with:
- Gradient histograms
- Validation tables
- Sample prediction images
- Namespace organization

### Simple Training Loop

```python
from kohakuboard.client import Board, Histogram

board = Board(name="resnet-training", config={"lr": 0.001, "batch_size": 32})

for epoch in range(100):
    board.step()

    # Training
    for batch in train_loader:
        loss = train_step(batch)
        board.log(**{"train/loss": loss, "train/lr": optimizer.lr})

    # Log gradients (every epoch, not every batch!)
    grad_data = {
        f"grad/{name}": Histogram(param.grad).compute_bins()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    board.log(**grad_data)

    # Validation
    val_loss, val_acc = validate()
    board.log(**{"val/loss": val_loss, "val/acc": val_acc})
```

---

## 🐛 Troubleshooting

### Queue Size Warning

```
WARNING: Queue size is 40000 (80% capacity)
```

**Fix:** Reduce logging frequency or precompute histograms

### Step Inflation

**Problem:** Histograms logged at different steps

**Fix:** Use unified `.log()` API with `Histogram` objects

### Slow Performance

**Fix:**
1. Precompute histograms: `.compute_bins()`
2. Use `precision="compact"` for histograms
3. Reduce logging frequency

---

## 🔗 Links

- **GitHub:** [KohakuBlueleaf/KohakuHub](https://github.com/KohakuBlueleaf/KohakuHub)
- **Example:** [CIFAR-10 Training](https://github.com/KohakuBlueleaf/KohakuHub/blob/main/examples/kohakuboard_cifar_training.py)
- **License:** Kohaku Software License 1.0 (Non-Commercial with Trial)

---

## 📋 Roadmap

### ✅ Completed

- [x] Python client library
- [x] Rich data types (scalars, media, tables, histograms)
- [x] Non-blocking async logging
- [x] Hybrid KohakuVault + SQLite storage
- [x] Local viewer (`kobo open`)
- [x] Step management
- [x] Namespace organization

### ⏳ In Progress (WIP)

- [ ] Remote server mode
- [ ] Authentication system
- [ ] Project management
- [ ] New sync protocol + refreshed `kobo sync`
- [ ] Frontend UI improvements
- [ ] Multi-user collaboration

### 🔮 Planned

- [ ] Real-time streaming (SSE)
- [ ] Run comparison UI
- [ ] Hyperparameter search visualization
- [ ] WandB import tool
- [ ] PyTorch Lightning integration
- [ ] Hugging Face Transformers integration

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/KohakuBlueleaf/KohakuHub/issues)
- **Discussions:** [GitHub Discussions](https://github.com/KohakuBlueleaf/KohakuHub/discussions)
- **Email:** kohaku@kblueleaf.net

---

## 🎉 Get Started

Ready to track your experiments?

```bash
# Install
pip install -e src/kohakuboard/

# Log your first experiment
python examples/kohakuboard_cifar_training.py

# View results
kobo open ./kohakuboard --browser
```

**[→ Go to Getting Started Guide](/docs/kohakuboard/getting-started)**
