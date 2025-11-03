---
title: KohakuBoard - Getting Started
description: Quick start guide for experiment tracking with KohakuBoard
icon: i-carbon-analytics
---

# Getting Started with KohakuBoard

High-performance, non-blocking experiment logging for ML training workflows.

---

## 🚀 Quick Start

### Installation

```bash
# From KohakuHub repository
cd /path/to/KohakuHub
pip install -e src/kohakuboard/

# Or install from source
pip install kohakuboard  # (when published)
```

### Your First Experiment

```python
from kohakuboard.client import Board

# Create a board (auto-creates directory structure)
board = Board(
    name="my-first-experiment",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "ResNet-50"
    }
)

# Training loop
for epoch in range(10):
    board.step()  # Mark new epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_step(data, target)

        # Log metrics (non-blocking!)
        board.log(
            loss=loss,
            lr=optimizer.param_groups[0]['lr']
        )

# Auto-saves on exit (via atexit hook)
```

**That's it!** Your metrics are saved to `./kohakuboard/{board_id}/data/`

---

## 📊 Rich Data Types

KohakuBoard supports 4 data types in a unified API:

### 1. Scalars (metrics)

```python
board.log(
    loss=0.5,
    accuracy=0.95,
    learning_rate=0.001
)

# Namespaces (creates tabs in UI)
board.log(**{
    "train/loss": 0.5,
    "val/accuracy": 0.95
})
```

### 2. Media (images, video, audio)

```python
from kohakuboard.client import Media

# Images
board.log(
    sample_image=Media(image_array),  # numpy, PIL, torch tensor
    predictions=Media(pred_image, caption="Predicted: cat")
)

# Video
board.log(
    training_video=Media("output.mp4", media_type="video")
)
```

### 3. Tables (structured data)

```python
from kohakuboard.client import Table

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

### 4. Histograms (distributions)

```python
from kohakuboard.client import Histogram

# Log gradient distributions
board.log(
    gradients=Histogram(param.grad),
    weights=Histogram(param.data)
)

# Precompute for efficiency (optional)
hist = Histogram(gradients).compute_bins()
board.log(grad_distribution=hist)
```

---

## 🎯 Key Features

### No Step Inflation!

**Problem:** Logging multiple histograms causes step inflation
```python
# BAD: Each call increments step separately
for name, param in model.named_parameters():
    board.log_histogram(f"gradients/{name}", param.grad)
# Result: 50 histograms = 50 different steps! ❌
```

**Solution:** Unified `.log()` API
```python
# GOOD: All histograms share same step
histogram_data = {}
for name, param in model.named_parameters():
    histogram_data[f"gradients/{name}"] = Histogram(param.grad)

board.log(**histogram_data)
# Result: 50 histograms = 1 step! ✅
```

### Mixed Type Logging

Log scalars, histograms, tables, and media together:

```python
board.log(
    loss=0.5,                      # Scalar
    sample_img=Media(image),       # Image
    results=Table(data),           # Table
    gradients=Histogram(grads),    # Histogram
)
# All logged at the SAME step with 1 queue message!
```

### Non-Blocking Writes

```python
board.log(loss=0.5)  # Returns immediately!
# Background writer process handles disk I/O
```

**Architecture:**
```
Main Thread              Background Process
     │                           │
     ├─ log(loss=0.5)            │
     │  └─> Queue.put()          │
     │      (non-blocking!)      │
     ├─ Continue training...     ├─ Queue.get()
     │                           ├─ Write to disk
     │                           └─ Flush
```

---

## 🎓 Complete Example

```python
from kohakuboard.client import Board, Histogram, Table, Media
import torch
from torch import nn, optim

# Initialize board
board = Board(
    name="cifar10-resnet18",
    config={
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 100,
        "optimizer": "AdamW"
    }
)

# Training loop
for epoch in range(100):
    board.step()  # Global step for epoch

    # Training phase
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log training metrics
        board.log(**{
            "train/loss": loss.item(),
            "train/lr": optimizer.param_groups[0]['lr']
        })

    # Log gradients every epoch (not every batch!)
    if epoch % 1 == 0:
        grad_data = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data[f"gradients/{name}"] = Histogram(param.grad).compute_bins()
        board.log(**grad_data)

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    predictions_table = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

            # Collect sample predictions (first batch only)
            if batch_idx == 0:
                for i in range(min(8, len(data))):
                    predictions_table.append({
                        "image": Media(data[i].cpu().numpy()),
                        "true": class_names[target[i]],
                        "pred": class_names[pred[i]],
                        "correct": "✓" if pred[i] == target[i] else "✗"
                    })

    # Log validation results (scalars + table together!)
    board.log(**{
        "val/loss": val_loss / len(val_loader),
        "val/accuracy": correct / len(val_loader.dataset),
        "val/predictions": Table(predictions_table)
    })

# Auto-cleanup on exit (atexit hook)
```

---

## 📂 Directory Structure

KohakuBoard creates this structure automatically:

```
kohakuboard/
└── {board_id}/                           # e.g., 20250129_150423_abc123
    ├── metadata.json                     # Board config
    ├── data/                             # Storage backend files
    │   ├── metrics/                      # (hybrid) KohakuVault DB files
    │   │   ├── train__loss.db
    │   │   ├── val__accuracy.db
    │   │   └── ...
    │   ├── metadata.db                   # (hybrid) SQLite metadata
    │   └── histograms/
    │       ├── gradients_i32.db       # int32 precision
    │       └── params_u8.db           # uint8 precision
    ├── media/                            # Images, videos, audio
    │   ├── sample_image_00000000_a1b2c3d4.png
    │   └── ...
    └── logs/
        ├── output.log                    # Captured stdout/stderr
        └── writer.log                    # Background writer logs
```

---

## ⚙️ Configuration

### Storage Backends

KohakuBoard supports 3 storage backends:

```python
# Hybrid (default, recommended)
board = Board(name="exp")
# - Metrics: KohakuVault (fastest)
# - Media/Tables: SQLite (best concurrency)

# DuckDB
board = Board(name="exp")
# - NaN/inf preservation
# - SQL query support

# Parquet
board = Board(name="exp", backend="sqlite")
# - Maximum compatibility
# - Good for post-processing
```

See [Configuration Guide](/docs/kohakuboard/configuration) for details.

### Context Manager

Ensure cleanup with context manager:

```python
with Board(name="experiment") as board:
    board.log(loss=0.5)
    # Automatic flush() and finish() on exit
```

### Manual Control

```python
board = Board(name="experiment")

# Explicit flush (blocks until all writes complete)
board.flush()

# Manual finish (called automatically via atexit)
board.finish()
```

---

## 🔧 Advanced Features

### Step Control

```python
# Auto-increment (default)
board.log(loss=0.5, auto_step=True)

# Manual control
board.log(loss=0.5, auto_step=False)  # No step increment

# Global step for grouping
for epoch in range(100):
    board.step()  # global_step = epoch
    for batch in train_loader:
        board.log(loss=loss)  # All batches share global_step
```

### Histogram Optimization

```python
# Default: compute in writer process
board.log(gradients=Histogram(param.grad))

# Precompute client-side (reduces queue size)
hist = Histogram(param.grad).compute_bins()
board.log(gradients=hist)

# Compact precision (75% size reduction, ~1% accuracy loss)
hist = Histogram(param.grad, precision="compact")
board.log(gradients=hist)
```

### Output Capture

```python
# Enabled by default
board = Board(name="exp", capture_output=True)

# Stdout/stderr saved to logs/output.log
print("Training started...")  # Captured!
```

---

## 📚 Next Steps

- **[API Reference](/docs/kohakuboard/api)** - Complete API documentation
- **[Configuration](/docs/kohakuboard/configuration)** - Storage backends, performance tuning
- **[Examples](/docs/kohakuboard/examples)** - Real-world usage patterns
- **[Best Practices](/docs/kohakuboard/best-practices)** - Performance tips and optimization

---

## 💡 Tips

✅ **DO:**
- Use unified `.log()` for all types
- Log histograms every N epochs (not every batch)
- Precompute histograms if CPU available
- Use namespaces for organization (`train/`, `val/`)

❌ **DON'T:**
- Mix `.log()` and `.log_histogram()` (causes step inflation)
- Log images/videos every batch (queue overload)
- Forget to call `board.step()` at epoch boundaries

---

## 🐛 Troubleshooting

### Queue Size Warning

```
WARNING: Queue size is 40000 (80% capacity)
```

**Fix:** Reduce logging frequency or precompute histograms

### Step Inflation

**Problem:** Multiple histograms logged separately
```python
# BAD
board.log_histogram("grad1", grad1)  # step 100
board.log_histogram("grad2", grad2)  # step 101
```

**Fix:** Use unified API
```python
# GOOD
board.log(grad1=Histogram(grad1), grad2=Histogram(grad2))  # step 100
```

---

## 📖 Further Reading

- [Storage Backend Comparison](/docs/kohakuboard/configuration#storage-backends)
- [Performance Optimization](/docs/kohakuboard/best-practices#performance)
- [CIFAR-10 Training Example](https://github.com/KohakuBlueleaf/KohakuHub/blob/main/examples/kohakuboard_cifar_training.py)
