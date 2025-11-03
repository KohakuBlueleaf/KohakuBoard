---
title: KohakuBoard Architecture
description: Deep dive into KohakuBoard's logging mechanism and storage architecture
icon: i-carbon-diagram
---

# KohakuBoard Architecture

Deep dive into the non-blocking logging system, storage backends, and data flow.

---

## Overview

KohakuBoard uses a **multi-process architecture** with background workers to achieve zero-overhead logging during model training.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "Main Process (Training Script)"
        A[Training Loop] --> B[board.log]
        B --> C[Queue.put]
        C --> D[Return Immediately]
    end

    subgraph "Background Process (Writer)"
        E[Queue.get] --> F[Message Dispatcher]
        F --> G{Message Type?}
        G -->|scalar| H[Metrics Writer]
        G -->|media| I[Media Handler]
        G -->|table| J[Table Writer]
        G -->|histogram| K[Histogram Writer]
        G -->|batch| L[Batch Handler]
    end

    subgraph "Storage Layer"
        H --> M[KohakuVault Metrics]
        I --> N[SQLite + Media Files]
        J --> N
        K --> O[KohakuVault Histograms]
        L --> P{Split by Type}
        P --> M
        P --> N
        P --> O
    end

    C -.->|Non-blocking| E
```

**Key Points:**
- Main process **never blocks** on disk I/O
- Queue acts as buffer (capacity: 50,000 messages)
- Background writer drains queue in batches
- All storage operations happen in background

---

## Logging Flow (Detailed)

### 1. Client-Side Logging

```mermaid
graph LR
    A[board.log] --> B{Categorize Values}
    B --> C[Scalars]
    B --> D[Media Objects]
    B --> E[Table Objects]
    B --> F[Histogram Objects]

    C --> G{Multiple Types?}
    D --> G
    E --> G
    F --> G

    G -->|Yes| H[Create Batch Message]
    G -->|No| I[Create Individual Messages]

    H --> J[Queue.put]
    I --> J
    J --> K[Return to Caller]

```

**Example - Single Type:**
```python
board.log(loss=0.5)  # Creates scalar message
```

**Example - Multiple Types (Batched):**
```python
board.log(
    loss=0.5,              # Scalar
    img=Media(...),        # Media
    hist=Histogram(...)    # Histogram
)
# Creates SINGLE batch message containing all 3 types!
```

---

### 2. Background Writer Process

```mermaid
graph TB
    A[Writer Process Start] --> B[Drain Queue]
    B --> C{Messages Available?}
    C -->|Yes| D[Process Batch]
    C -->|No| E[Adaptive Sleep]

    D --> F{Message Type?}
    F -->|scalar| G[Append Metrics]
    F -->|media| H[Process & Save Media]
    F -->|table| I[Extract Media + Save Table]
    F -->|histogram| J{Precomputed?}
    F -->|batch| K[Handle All Types]

    J -->|Yes| L[Store Bins/Counts]
    J -->|No| M[Compute Histogram]
    M --> L

    G --> N[Flush to Storage]
    H --> N
    I --> N
    L --> N
    K --> N

    N --> O[Log Stats]
    O --> B

    E --> P[10ms-1s Backoff]
    P --> B

```

**Adaptive Batching:**
- If queue has messages: Process immediately
- If queue empty: Sleep 10ms → 20ms → 50ms → ... → 1s (exponential backoff)
- Minimizes CPU usage while maintaining low latency

---

## Storage Architecture: Hybrid KohakuVault + SQLite

KohakuBoard uses a **hybrid storage system** combining KohakuVault (columnar) and SQLite (row-oriented) for optimal performance. After trying various approaches including DuckDB and Parquet, we found this combination works best.

### Design Philosophy

**Reading Logs = Reading Entire Columns**

When viewing experiments, you typically read **all values** of a metric at once:
```python
# Typical read pattern
loss_values = read_metric("train/loss")
# → [0.5, 0.48, 0.45, 0.42, ...] (entire column)
```

This is a **column-oriented access pattern** → Perfect for KohakuVault!

**Metadata = Row-Oriented Queries**

When accessing media or tables, you query **specific rows**:
```python
# Typical read pattern
media = get_media_at_step(step=100)
# → Single row with filename, dimensions, caption
```

This is a **row-oriented access pattern** → Perfect for SQLite!

```mermaid
graph TB
    A[Storage Decision] --> B{Access Pattern?}

    B -->|Read whole column<br/>at once| C[Use KohakuVault]
    B -->|Random row<br/>access| D[Use SQLite]

    C --> E[Metrics<br/>Histograms]
    D --> F[Media metadata<br/>Tables<br/>Step info]

    E --> G[Non-blocking writes<br/>Fast column scans<br/>Automatic schema]
    D --> H[WAL mode<br/>Concurrent reads<br/>Relational queries]

```

---

### Hybrid Storage Architecture

```mermaid
graph TB
    subgraph "Hybrid Storage (KohakuVault + SQLite)"
        A[Writer Process] --> B{Data Type}

        B -->|Scalars| C[ColumnVaultMetricsStorage]
        B -->|Media Metadata| D[SQLiteMetadataStorage]
        B -->|Tables| D
        B -->|Step Info| D
        B -->|Histograms| E[HistogramStorage]

        C --> F[Per-Metric DB Files]
        F --> F1[train__loss.db]
        F --> F2[val__accuracy.db]

        D --> G[metadata.db<br/>SQLite WAL mode]
        G --> G1[media table]
        G --> G2[tables table]
        G --> G3[step_info table]

        E --> H[Namespace-Grouped DBs]
        H --> H1[gradients_i32.db]
        H --> H2[params_u8.db]
    end

    subgraph "Media Files"
        I[Media Handler] --> J[media/]
        J --> J1[*.png, *.jpg]
        J --> J2[*.mp4, *.wav]
    end

    D -.->|References| I

```

**Directory Structure:**
```
data/
├── metrics/
│   ├── train__loss.db       # Per-metric KohakuVault files (column-oriented)
│   ├── val__accuracy.db
│   └── ...
├── metadata.db                 # SQLite WAL (row-oriented metadata)
│   ├── media table             # Media file references
│   ├── tables table            # Table data
│   └── step_info table         # Step timestamps
└── histograms/
    ├── gradients_i32.db     # Namespace-grouped histograms
    ├── params_u8.db
    └── ...
media/
├── sample_image_00000100_a1b2c3d4.png
├── training_video_00000200_e5f6g7h8.mp4
└── ...
```

---

### Why KohakuVault for Metrics & Histograms?

**Column-Oriented Benefits:**
- ✅ **Non-blocking incremental writes:** No locks, multiple readers possible
- ✅ **Fast column reads:** Read entire metric efficiently (typical use case)
- ✅ **Automatic schema evolution:** New metrics added dynamically
- ✅ **Compression:** Efficient storage for numeric data

**Trade-offs:**
- ⚠️ NaN/Inf converted to None (handled by KohakuVault)
- ⚠️ Not ideal for row-oriented random access

**Why we chose KohakuVault over alternatives:**
- DuckDB: Great for SQL but had locking issues with high-frequency writes (if you want to support multiple reader you need to keep start/close connection which is slow)
- Parquet: Read-concat-write pattern too slow for real-time logging
- **KohakuVault**: Perfect balance of write performance and read efficiency

---

### Why SQLite for Metadata?

**Row-Oriented Benefits:**
- ✅ **WAL mode = non-blocking reads:** Multiple readers don't block
- ✅ **ACID transactions:** Data integrity guaranteed
- ✅ **Perfect for metadata:** Media refs, table data, step info
- ✅ **Relational queries:** Join media to steps easily
- ✅ **Battle-tested:** Mature, reliable, zero external dependencies
- ✅ **Single file:** Easy to manage and backup

**Why SQLite over alternatives:**
- More feasible for row-oriented data than columnar formats
- Better concurrency than plain file I/O
- No server process needed (unlike PostgreSQL)
- **WAL mode is key:** Writer doesn't block readers!

---

### Performance Comparison

| Operation | KohakuVault | SQLite | Best For |
|-----------|-------|--------|----------|
| **Sequential column read** | Fastest | Slow | KohakuVault ✅ |
| **Random row access** | Slow | Fast | SQLite ✅ |
| **Incremental append** | Non-blocking | Single writer | KohakuVault ✅ |
| **Concurrent reads** | Excellent | Excellent (WAL) | Both ✅ |
| **Schema evolution** | Automatic | ALTER TABLE | KohakuVault ✅ |
| **Relational queries** | ❌ | ✅ SQL | SQLite ✅ |

**Result:** Use KohakuVault where KohakuVault excels, SQLite where SQLite excels!

---

### KohakuVault for Media Storage (v0.3.0+)

**Implemented:** High-performance SQLite KV store for media binary blobs using [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault)

```mermaid
graph LR
    A[Hybrid Storage] --> B[KohakuVault<br/>Column-oriented]
    A --> C[SQLite<br/>Metadata]
    A --> D[KohakuVault<br/>Media blobs]

    B --> B1[Metrics<br/>Histograms]
    C --> C1[Media refs<br/>Tables<br/>Step info]
    D --> D1[Images<br/>Videos<br/>Audio]

```

**Design Decisions:**

**Why KohakuVault instead of filesystem:**
- Eliminates filesystem overhead (no inode limits, directory traversal)
- Single database file instead of thousands of individual files
- ACID transactions with automatic deduplication
- Easy to symlink/relocate media folder for deployment
- No size limits (dynamic growth)

**KohakuVault Features:**
- **Rust backend:** High-performance native implementation
- **Streaming operations:** Memory-efficient for multi-GB files
- **Write-back caching:** Batches writes for optimal disk I/O
- **Dict-like API:** Simple `vault[key] = data` interface
- **Zero dependencies:** Self-contained SQLite database

**Usage Example:**
```python
from kohakuvault import KVault

vault = KVault("media.db")

# Store media
vault["a1b2c3...xyz.png"] = image_bytes

# Retrieve media
data = vault.get("a1b2c3...xyz.png")

# Check existence
if "key" in vault:
    ...
```

**Implementation:**
- **Library:** [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault) (Rust + Python bindings)
- **File Location:** `{board_dir}/media/blobs.db`
- **Key Format:** `{media_hash}.{format}` (e.g., `a1b2c3d4e5f6g7h8i9j0k1.png`)
- **Benefits:** All WAL files in media/ folder for easy deployment

**Status:** ✅ Implemented in v0.3.0

---

## Message Protocol

### Message Types

```mermaid
graph LR
    A[Message Types] --> B[scalar]
    A --> C[media]
    A --> D[table]
    A --> E[histogram]
    A --> F[batch]
    A --> G[flush]

    B --> B1[step, global_step,<br/>metrics dict, timestamp]
    C --> C1[step, global_step,<br/>name, media_data,<br/>media_type, caption]
    D --> D1[step, global_step,<br/>name, table_data]
    E --> E1[step, global_step,<br/>name, values/bins/counts]
    F --> F1[step, global_step,<br/>scalars, media,<br/>tables, histograms]

```

### Scalar Message

```json
{
  "type": "scalar",
  "step": 100,
  "global_step": 5,
  "metrics": {
    "train/loss": 0.5,
    "train/lr": 0.001
  },
  "timestamp": "2025-10-29T13:00:00Z"
}
```

### Batch Message (Multiple Types)

```json
{
  "type": "batch",
  "step": 100,
  "global_step": 5,
  "timestamp": "2025-10-29T13:00:00Z",
  "scalars": {
    "loss": 0.5
  },
  "media": {
    "sample_img": {
      "media_type": "image",
      "media_data": "...",
      "caption": "Sample"
    }
  },
  "tables": {
    "results": {
      "columns": ["name", "score"],
      "rows": [...]
    }
  },
  "histograms": {
    "gradients": {
      "computed": true,
      "bins": [...],
      "counts": [...]
    }
  }
}
```

---

## Histogram Optimization

### Client-Side vs Writer-Side Computation

```mermaid
graph TB
    subgraph "Option 1: Writer-Side (Default)"
        A1[Histogram<br/>raw values] --> B1[Queue Message<br/>~100KB for 10K values]
        B1 --> C1[Writer Process]
        C1 --> D1[Compute Bins/Counts]
        D1 --> E1[Store<br/>~256 bytes]
    end

    subgraph "Option 2: Client-Side (Optimized)"
        A2[Histogram<br/>raw values] --> B2[.compute_bins]
        B2 --> C2[Queue Message<br/>~256 bytes]
        C2 --> D2[Writer Process]
        D2 --> E2[Store Directly<br/>~256 bytes]
    end

```

**Size Comparison:**
- Raw values: ~100KB for 10,000 values
- Precomputed bins/counts: ~256 bytes (64 bins × 4 bytes)
- **Reduction: 99.7%**

**Usage:**
```python
# Default: Send raw values
board.log(gradients=Histogram(param.grad))

# Optimized: Precompute bins
hist = Histogram(param.grad).compute_bins()
board.log(gradients=hist)
```

---

## Step Management

### Two-Level Step Tracking

```mermaid
graph TB
    subgraph "Step Tracking System"
        A[board.step] --> B[Increment global_step]
        C[board.log] --> D{auto_step=True?}
        D -->|Yes| E[Increment _step]
        D -->|No| F[No Increment]

        B --> G[global_step<br/>User-controlled<br/>Epoch/phase tracking]
        E --> H[_step<br/>Auto-increment<br/>Sequential ordering]
        F --> H
    end

```

**Example:**
```python
for epoch in range(100):
    board.step()  # global_step = epoch

    for batch in train_loader:
        board.log(loss=loss)  # _step auto-increments
                             # All batches share global_step=epoch
```

**Storage:**
- Both `_step` and `global_step` stored for each log entry
- `_step`: Used for deduplication and ordering
- `global_step`: Used for charting and grouping

---

## Media Handling

### Image Processing Pipeline

```mermaid
graph LR
    A[Input] --> B{Type?}
    B -->|PIL Image| C[Convert to PNG]
    B -->|NumPy Array| D[Normalize + Convert]
    B -->|Torch Tensor| E[To NumPy]
    B -->|File Path| F[Copy File]

    C --> G[SHA-256 Hash]
    D --> G
    E --> G
    F --> G

    G --> H{Already Exists?}
    H -->|Yes| I[Reuse Existing]
    H -->|No| J[Save to media/]

    I --> K[Return media_id]
    J --> K

    K --> L[Store Metadata<br/>in metadata.db]

```

**Deduplication:**
- SHA-256 hash of image data
- If hash exists, reuse existing file
- Saves disk space for repeated images

**Filename Format:**
```
{name}_{step:08d}_{hash[:8]}.{ext}
```

Example: `sample_image_00000100_a1b2c3d4.png`

---

## Histogram Storage

### Precision Modes

```mermaid
graph TB
    A[Raw Values<br/>10000 floats] --> B{Precision Mode}

    B -->|exact| C[Compute Histogram<br/>64 bins]
    B -->|compact| D[Compute Histogram<br/>64 bins]

    C --> E[int32 counts]
    D --> F[Normalize to 0-255]
    F --> G[uint8 counts]

    E --> H[Store 64 × 4 bytes<br/>= 256 bytes]
    G --> I[Store 64 × 1 bytes<br/>= 64 bytes]

```

**Storage Comparison:**
- Raw values: 10,000 × 4 bytes = **40 KB**
- Exact (int32): 64 × 4 bytes = **256 bytes** (99.4% reduction)
- Compact (uint8): 64 × 1 bytes = **64 bytes** (99.8% reduction)

**Schema:**
```
step: int64
global_step: int64
name: string
counts: list<uint8 or int32>
min: float32 (p1 percentile)
max: float32 (p99 percentile)
```

---

## Queue Management

### Queue Lifecycle

```mermaid
graph TB
    A[Queue Created<br/>capacity: 50000] --> B[Messages Added]
    B --> C{Queue Size}

    C -->|< 40000| D[Normal Operation]
    C -->|≥ 40000| E[Warning Logged<br/>80% capacity]
    C -->|≥ 50000| F[Queue Full<br/>put blocks]

    D --> G[Writer Drains]
    E --> G
    F --> H[Reduce Logging<br/>Frequency]
    H --> G

    G --> I{Stop Event?}
    I -->|No| B
    I -->|Yes| J[Final Drain]
    J --> K[Flush All]
    K --> L[Close Storage]

```

**Queue Monitoring:**
```python
# Automatic warnings
WARNING: Queue size is 40000 (80% capacity)
```

**Solutions:**
1. Reduce logging frequency (e.g., log histograms every 10 epochs instead of every epoch)
2. Precompute histograms client-side (`.compute_bins()`)
3. Use `precision="compact"` for histograms

---

## Graceful Shutdown

### Shutdown Flow

```mermaid
graph TB
    A[Shutdown Trigger] --> B{Trigger Type}

    B -->|atexit| C[Normal Exit]
    B -->|SIGINT| D[Ctrl+C]
    B -->|SIGTERM| E[Kill Signal]
    B -->|Exception| F[Uncaught Error]

    C --> G[board.finish]
    D --> H{Double Ctrl+C?}
    E --> G
    F --> G

    H -->|No| G
    H -->|Yes<br/>within 3s| I[Force Exit<br/>May lose data]

    G --> J[Stop Writer Process]
    J --> K[Drain Queue<br/>No message limit]
    K --> L[Final Flush]
    L --> M[Close Storage]
    M --> N[Restore stdout/stderr]
    N --> O[Exit Clean]

```

**Features:**
- ✅ `atexit` hook registers automatically
- ✅ Signal handlers for SIGINT and SIGTERM
- ✅ Double Ctrl+C for emergency exit
- ✅ Exception handler flushes before crash
- ✅ **No arbitrary limits** during final drain

---

## Performance Characteristics

### Throughput

| Operation | Time | Throughput |
|-----------|------|------------|
| `board.log(scalar)` | < 1µs | Non-blocking |
| Queue put | ~10µs | ~100,000/s |
| Writer batch (1000 metrics) | ~50ms | ~20,000/s |
| Histogram computation | ~5ms | ~200/s |
| Media save (PNG) | ~10ms | ~100/s |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Queue (empty) | ~1 MB |
| Queue (40k messages) | ~50-100 MB |
| Writer process | ~100-200 MB |
| KohakuVault writer cache | ~50 MB |
| SQLite cache | ~10 MB |

### Latency

```mermaid
graph LR
    A[board.log] -->|~10µs| B[Queue.put]
    B -->|~0-1s| C[Writer drains]
    C -->|~10-50ms| D[Batch write]
    D -->|~10ms| E[Flush to disk]
    E --> F[Data persisted]

```

**Typical latency:**
- Log call: < 1µs (instant return)
- Queue to disk: 10ms - 1s (adaptive)
- Total: Data on disk within 1-2 seconds

---

## Best Practices

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

### Histogram Optimization

```python
# ✅ DO: Precompute if logging many histograms
hist_data = {}
for name, param in model.named_parameters():
    hist_data[f"grad/{name}"] = Histogram(param.grad).compute_bins()
board.log(**hist_data)

# ❌ DON'T: Send raw values for 100+ histograms
for name, param in model.named_parameters():
    board.log(**{f"grad/{name}": Histogram(param.grad)})  # Large queue messages!
```

### Batching

```python
# ✅ DO: Log related values together
board.log(
    loss=0.5,
    accuracy=0.95,
    sample_img=Media(img),
    results=Table(data)
)  # Single queue message!

# ❌ DON'T: Separate calls (4 messages, 4 step increments)
board.log(loss=0.5)
board.log(accuracy=0.95)
board.log(sample_img=Media(img))
board.log(results=Table(data))
```

---

## Debugging

### Enable Debug Logging

Check `logs/writer.log` for detailed timing:

```
[DEBUG] Processed and flushed 1000 metrics in 45.2ms
[DEBUG] Histogram batch: 50 histograms in 120.5ms
[DEBUG] Auto-flush completed (5000 messages processed)
```

### Queue Monitoring

```python
# Check queue size (private API, for debugging)
print(f"Queue size: {board.queue.qsize()}")
```

---

## Summary

KohakuBoard's architecture provides:

- ✅ **Non-blocking logging** via background writer
- ✅ **Hybrid storage** combining KohakuVault (columnar) + SQLite (row-oriented)
- ✅ **Efficient batching** for mixed-type logs
- ✅ **Histogram optimization** via client-side precomputation
- ✅ **Graceful shutdown** with full queue draining
- ✅ **High throughput** (~20,000 metrics/second sustained)

The design prioritizes **training performance** (zero blocking) while maintaining **data integrity** (graceful shutdown, queue monitoring). The KohakuVault+SQLite combination leverages each database's strengths for optimal performance.
