---
title: KohakuBoard API Reference
description: Complete API documentation for KohakuBoard client
icon: i-carbon-api
---

# KohakuBoard API Reference

Complete API documentation for the KohakuBoard Python client.

---

## Board Class

Main interface for experiment logging.

### Constructor

```python
Board(
    name: str | None = None,
    board_id: str | None = None,
    config: dict[str, Any] | None = None,
    project: str | None = None,
    base_dir: str | Path | None = None,
    capture_output: bool = True,
    remote_url: str | None = None,
    remote_token: str | None = None,
    remote_project: str | None = None,
    sync_enabled: bool = False,
    sync_interval: int = 10,
    memory_mode: bool = False,
    *,
    annotation: str | None = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | Friendly words | Human-readable board name (auto-generated if `None`) |
| `board_id` | `str \| None` | Auto-generated | Stable ID (`YYYYmmdd_HHMMSS_<4-char>`) |
| `config` | `dict \| None` | `{}` | Hyperparameters/configuration dict |
| `project` | `str \| None` | `"default"` | Subdirectory for grouping runs |
| `base_dir` | `str \| Path \| None` | `./kohakuboard` | Root directory containing projects |
| `capture_output` | `bool` | `True` | Mirror stdout/stderr into `logs/output.log` |
| `remote_url` | `str \| None` | - | Remote server base URL (future sync) |
| `remote_token` | `str \| None` | - | Auth token for remote sync |
| `remote_project` | `str \| None` | Project name | Remote project override |
| `sync_enabled` | `bool` | `False` | Start sync worker (requires remote_* args) |
| `sync_interval` | `int` | `10` | Seconds between sync attempts |
| `memory_mode` | `bool` | `False` | Keep storage in RAM only (requires sync to persist) |
| `annotation` | `str \| None` | `None` | Optional suffix appended to folder name |

**Example:**

```python
board = Board(
    name="bert-finetuning",
    config={
        "learning_rate": 2e-5,
        "batch_size": 32,
        "max_epochs": 10,
        "model": "bert-base-uncased"
    },
    
)
```

**Auto-Generated Structure:**

```
{base_dir}/
└── {board_id}/
    ├── metadata.json          # Board configuration
    ├── data/                  # Storage backend files
    ├── media/                 # Media files (images/video/audio)
    └── logs/
        ├── output.log         # Captured stdout/stderr
        └── writer.log         # Writer process logs
```

---

## Logging Methods

### board.log()

**Unified logging method** supporting all data types (recommended).

```python
board.log(
    auto_step: bool = True,
    **metrics: Union[int, float, Media, Table, Histogram]
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_step` | `bool` | `True` | Whether to auto-increment internal step |
| `**metrics` | `Union[int, float, Media, Table, Histogram]` | Required | Name-value pairs to log |

**Supported Value Types:**

1. **Scalars**: `int`, `float`, numpy scalars, single-item tensors
2. **Media**: `Media` objects (images, video, audio)
3. **Tables**: `Table` objects (structured data)
4. **Histograms**: `Histogram` objects (distributions)
5. **TensorLog**: Large tensors stored once, streamed on demand
6. **KernelDensity**: KDE objects (precomputed or raw samples)

**Examples:**

```python
# Scalars only
board.log(loss=0.5, accuracy=0.95, lr=0.001)

# Mixed types (all share same step!)
board.log(
    loss=0.5,
    sample_img=Media(image_array),
    results=Table(data),
    gradients=Histogram(grad_values)
)

# Namespaces (creates tabs in UI)
board.log(**{
    "train/loss": 0.5,
    "train/lr": 0.001,
    "val/accuracy": 0.95
})

# Manual step control
board.log(loss=0.5, auto_step=False)  # No step increment
```

**Key Feature:**

All values logged in a single `log()` call share the same step, avoiding step inflation.

---

### board.log_images()

Log single or multiple images.

```python
board.log_images(
    name: str,
    images: Union[Any, List[Any]],
    caption: Optional[str] = None
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Log name (e.g., `"sample_predictions"`) |
| `images` | `Any \| List[Any]` | Image(s) - PIL Image, numpy array, torch Tensor, file path |
| `caption` | `str \| None` | Optional caption |

**Supported Formats:**

- **PIL Images**: `Image.open(...)` objects
- **NumPy arrays**: `(H, W)`, `(H, W, C)`, `(C, H, W)` - auto-converts
- **PyTorch Tensors**: Auto-converts to numpy
- **File paths**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.tiff`

**Examples:**

```python
# Single image
board.log_images("prediction", pred_image)

# Multiple images
board.log_images("samples", [img1, img2, img3], caption="Generated samples")

# From file (preserves format, e.g., GIF animation)
board.log_images("animation", "output.gif")
```

**Note:** Prefer using `Media` objects with `board.log()` for unified API.

---

### board.log_video()

Log video file.

```python
board.log_video(
    name: str,
    video_path: Union[str, Path],
    caption: Optional[str] = None
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Log name |
| `video_path` | `str \| Path` | Path to video file |
| `caption` | `str \| None` | Optional caption |

**Supported Formats:**

`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`, `.m4v`

**Example:**

```python
board.log_video("training_progress", "training.mp4", caption="Epoch 1-10")
```

---

### board.log_audio()

Log audio file.

```python
board.log_audio(
    name: str,
    audio_path: Union[str, Path],
    caption: Optional[str] = None
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Log name |
| `audio_path` | `str \| Path` | Path to audio file |
| `caption` | `str \| None` | Optional caption |

**Supported Formats:**

`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`

**Example:**

```python
board.log_audio("generated_speech", "output.wav")
```

---

### board.log_table()

Log structured tabular data.

```python
board.log_table(
    name: str,
    table: Union[Table, List[Dict[str, Any]]]
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Log name |
| `table` | `Table \| List[Dict]` | Table object or list of dicts |

**Example:**

```python
# From list of dicts
board.log_table("results", [
    {"name": "Alice", "score": 95},
    {"name": "Bob", "score": 87},
])

# Using Table object (recommended)
from kohakuboard.client import Table
board.log_table("results", Table([
    {"name": "Alice", "score": 95},
]))
```

---

### board.log_histogram()

Log distribution as histogram.

```python
board.log_histogram(
    name: str,
    values: Union[List[float], np.ndarray, torch.Tensor],
    num_bins: int = 64,
    precision: str = "exact"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Log name |
| `values` | `List[float] \| array` | Required | Raw values for histogram |
| `num_bins` | `int` | `64` | Number of histogram bins |
| `precision` | `str` | `"exact"` | `"exact"` (int32) or `"compact"` (uint8, 75% smaller) |

**Example:**

```python
# Log gradients
board.log_histogram("gradients/layer1", param.grad)

# Compact precision (smaller storage)
board.log_histogram("weights", param.data, precision="compact")
```

**Warning:** Each call increments step. Prefer `Histogram` objects with `board.log()` for multiple histograms.

---

### board.log_tensor()

Log high-dimensional tensors once and stream them lazily in the UI.

```python
from kohakuboard.client import TensorLog

board.log_tensor(
    name: str,
    values: Any,
    precision: Any | None = None,
    metadata: dict[str, Any] | None = None,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Log name (namespaces supported) |
| `values` | `Any` | Tensor/array-like payload (PyTorch, NumPy, etc.) |
| `precision` | `Any \| None` | Optional dtype override during serialization |
| `metadata` | `dict \| None` | Additional metadata stored alongside the tensor |

**Usage:**

```python
board.log_tensor(
    "attention/layer3",
    values=attn_tensor,
    metadata={"head": 4, "token": 128},
)
```

---

### board.log_kernel_density()

Log Kernel Density Estimates (KDE) either from raw samples or precomputed arrays.

```python
from kohakuboard.client import KernelDensity

board.log_kernel_density(
    name: str,
    grid: Any | None = None,
    density: Any | None = None,
    *,
    values: Any | None = None,
    num_points: int = 256,
    kernel: str = "gaussian",
    bandwidth: float | str | None = None,
    sample_count: int | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
    metadata: dict[str, Any] | None = None,
    approximate: bool = False,
    approximate_bins: int | None = None,
)
```

- Pass `values=` to have the writer compute the KDE automatically.
- Pass `grid=` + `density=` to reuse existing KDE computations.
- `approximate=True` switches to a histogram+FFT path for very large sample counts.

```python
board.log_kernel_density(
    "val/logits",
    values=logits.cpu().numpy(),
    num_points=512,
    percentile_clip=(0.5, 99.5),
)
```

---

## Step Management

### board.step()

Increment global step counter.

```python
board.step(increment: int = 1)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `increment` | `int` | `1` | Amount to increment |

**Example:**

Call this **once per optimizer step** (not per epoch).

```python
for batch_idx, batch in enumerate(train_loader):
    loss = train_step(batch)
    optimizer.step()

    board.step()  # global_step += 1
    board.log(loss=loss.item())
```

**Two Step Tracking Systems:**

1. **`_step` (auto-increment):**
   - Increments on every `log()` call (unless `auto_step=False`)
   - Internal sequential counter
   - Used for ordering logs

2. **`_global_step` (user-controlled):**
   - Controlled via `board.step()`
   - Used for grouping (e.g., all batches in epoch)
   - Used for charting/aggregation

---

## Lifecycle Management

### board.flush()

Flush all pending logs to disk (blocking).

```python
board.flush()
```

Blocks until background writer completes all pending writes.

**Example:**

```python
board.log(loss=0.5)
board.flush()  # Wait for write to complete
# Now safe to read files
```

---

### board.finish()

Cleanup and shutdown (called automatically via `atexit`).

```python
board.finish()
```

**Behavior:**

1. Stop writer process
2. Drain remaining queue messages
3. Final flush
4. Close storage backends
5. Restore stdout/stderr

**Note:** Usually don't need to call manually - `atexit` hook handles it.

---

### Context Manager

Recommended usage pattern for guaranteed cleanup:

```python
with Board(name="experiment") as board:
    board.log(loss=0.5)
    # Automatic flush() and finish() on exit
```

---

## Data Types

### Media

Wrapper for images, videos, and audio.

```python
Media(
    data: Any,
    caption: str = "",
    media_type: str = "image"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Any` | Required | Image array, file path, or media data |
| `caption` | `str` | `""` | Optional caption |
| `media_type` | `str` | `"image"` | `"image"`, `"video"`, or `"audio"` |

**Example:**

```python
from kohakuboard.client import Media

# Image from numpy array
img = Media(image_array, caption="Sample prediction")

# Image from file
img = Media("sample.png", caption="Input")

# Video
video = Media("training.mp4", media_type="video")

# In table
table = Table([
    {"image": Media(img), "label": "cat", "confidence": 0.95}
])
```

---

### Table

Structured tabular data with optional media embedding.

```python
Table(
    data: Optional[Union[List[Dict], List[List]]] = None,
    columns: Optional[List[str]] = None,
    column_types: Optional[List[str]] = None,
    rows: Optional[List[List[Any]]] = None
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `List[Dict] \| List[List] \| None` | Data as list of dicts or list of lists |
| `columns` | `List[str] \| None` | Column names (auto-inferred if None) |
| `column_types` | `List[str] \| None` | Column types: `text`, `number`, `media`, `boolean` |
| `rows` | `List[List] \| None` | Row data (if not using `data`) |

**Column Types:**

- `text`: String values
- `number`: int/float values
- `media`: Media objects
- `boolean`: True/False values

**Examples:**

```python
from kohakuboard.client import Table, Media

# From list of dicts (simplest)
table = Table([
    {"name": "Alice", "score": 95, "pass": True},
    {"name": "Bob", "score": 87, "pass": True},
])

# With explicit schema
table = Table(
    columns=["Sample", "Precision", "Recall"],
    column_types=["text", "number", "number"],
    rows=[
        ["Cat", 0.85, 0.80],
        ["Dog", 0.88, 0.85],
    ]
)

# With embedded media
table = Table([
    {
        "image": Media(img1),
        "label": "cat",
        "confidence": 0.95
    },
    {
        "image": Media(img2),
        "label": "dog",
        "confidence": 0.87
    }
])
```

**Methods:**

```python
table.to_dict() -> Dict[str, Any]
```

Returns serialized table data with media objects extracted.

---

### Histogram

Distribution tracking with configurable precision.

```python
Histogram(
    values: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    num_bins: int = 64,
    precision: str = "exact",
    bins: Optional[np.ndarray] = None,
    counts: Optional[np.ndarray] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `array \| List \| None` | `None` | Raw values (if not precomputed) |
| `num_bins` | `int` | `64` | Number of histogram bins |
| `precision` | `str` | `"exact"` | `"exact"` (int32) or `"compact"` (uint8) |
| `bins` | `array \| None` | `None` | Precomputed bin edges |
| `counts` | `array \| None` | `None` | Precomputed bin counts |

**Two Usage Modes:**

**1. Raw values (computed in writer process):**

```python
hist = Histogram(gradients)
board.log(grad_hist=hist)
```

**2. Precomputed (reduces queue size):**

```python
hist = Histogram(gradients)
hist.compute_bins()  # Compute immediately
board.log(grad_hist=hist)
```

**Methods:**

```python
# Compute bins/counts from raw values
hist.compute_bins() -> Histogram

# Check if computed
hist.is_computed() -> bool

# Serialize for storage
hist.to_dict() -> Dict[str, Any]
```

**Precision Modes:**

- **`exact` (int32):** Preserves exact bin counts, larger size
- **`compact` (uint8):** Normalized 0-255, ~1% accuracy loss, 75% size reduction

**Example:**

```python
from kohakuboard.client import Histogram

# Basic usage
hist = Histogram(param.grad, num_bins=64, precision="exact")
board.log(gradients=hist)

# Precompute for efficiency
hist = Histogram(param.grad).compute_bins()
board.log(gradients=hist)

# Compact precision (smaller storage)
hist = Histogram(param.grad, precision="compact")
board.log(gradients=hist)

# From precomputed bins/counts
hist = Histogram(bins=bin_edges, counts=bin_counts)
board.log(custom_hist=hist)
```

---

## Properties

### board.name

Board name (human-readable).

```python
board.name -> str
```

---

### board.board_id

Unique board ID (timestamp + UUID).

```python
board.board_id -> str
```

---

### board.board_dir

Path to board directory.

```python
board.board_dir -> Path
```

---

### board._step

Auto-increment step counter (internal).

```python
board._step -> int
```

**Warning:** Private API. May change.

---

### board._global_step

User-controlled global step.

```python
board._global_step -> int
```

**Warning:** Private API. Prefer using `board.step()`.

---

## Utilities

### Scalar Conversion

KohakuBoard automatically converts scalars to Python numbers:

- **NumPy scalars**: `np.float32(0.5)` → `0.5`
- **PyTorch tensors**: `torch.tensor(0.5)` → `0.5`
- **Single-item tensors**: `torch.tensor([0.5])` → `0.5`

---

## Error Handling

### Queue Size Warning

```
WARNING: Queue size is 40000 (80% capacity)
```

Reduce logging frequency or precompute histograms.

---

### Graceful Shutdown

KohakuBoard handles:

- **Ctrl+C (SIGINT)**: Flush and exit gracefully
- **Kill (SIGTERM)**: Flush and exit
- **Double Ctrl+C**: Force exit (within 3 seconds)
- **Uncaught exceptions**: Auto-finish before crash
- **Normal exit**: `atexit` hook calls `finish()`

---

## Performance

### Non-Blocking Architecture

```
Main Thread                Background Process
     │                              │
     ├─ log(loss=0.5)               │
     │  └─> Queue.put()             │
     │      (instant return!)       │
     │                              ├─ Queue.get()
     │                              ├─ Write to disk
     │                              └─ Flush
     │
     ├─ Continue training...
```

### Queue Configuration

- **Capacity**: 50,000 messages
- **Warning threshold**: 40,000 (80%)
- **Batch processing**: Drains all available messages per iteration

### Adaptive Batching

Writer process uses exponential backoff when queue is empty:
- Initial: 10ms sleep
- Maximum: 1s sleep
- Ensures low latency when active, low CPU when idle

---

## See Also

- [Getting Started](/docs/kohakuboard/getting-started) - Quick start guide
- [Configuration](/docs/kohakuboard/configuration) - Storage backends and tuning
- [Examples](/docs/kohakuboard/examples) - Real-world usage patterns
- [Best Practices](/docs/kohakuboard/best-practices) - Performance tips
