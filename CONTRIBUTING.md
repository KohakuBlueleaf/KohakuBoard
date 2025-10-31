# Contributing to KohakuBoard

Thank you for your interest in contributing to KohakuBoard! We welcome contributions from the community.

> **Note:** KohakuBoard is a standalone sub-project under [KohakuHub](https://github.com/KohakuBlueleaf/KohakuHub). For contributions to the broader KohakuHub ecosystem, please see the main repository.

## Quick Links

- **Discord:** https://discord.gg/xWYrkyvJ2s (Best for discussions)
- **GitHub Issues:** https://github.com/KohakuBlueleaf/KohakuHub/issues (Tag with `kohakuboard`)
- **Roadmap:** See [Project Status](#project-status) below

---

## Code Conventions and Rules

### Python Code Style

**Core Principles:**
1. **Minimal solution, but you can't skip anything.** If any implementation/target/goal are too difficult, discuss first. Don't silently ignore them.
2. **Modern Python:** Use match-case instead of nested if-else, utilize native type hints (use `list[]`, `dict[]` instead of importing from `typing` unless needed)
3. **Clean code:** Try to split large functions into smaller ones
4. **Type hints recommended but not required** - No static type checking, but use type hints for documentation

**Import Order Rules:**
```python
# 1. builtin packages
import atexit
import hashlib
import multiprocessing
from datetime import datetime
from pathlib import Path

# 2. Third-party packages (alphabetical)
import duckdb
import numpy as np
import pyarrow as pa
from fastapi import APIRouter, HTTPException
from peewee import fn

# 3. Our packages (shorter paths first, then alphabetical)
from kohakuboard.client import Board
from kohakuboard.client.storage import HybridStorage
from kohakuboard.client.types import Media, Table, Histogram
from kohakuboard.client.writer import LogWriter

# Within each group:
# - `import xxx` comes before `from xxx import`
# - Shorter paths before longer paths
# - Alphabetical order
```

**Type Hints - Use Native Types:**
```python
# ✅ Good - native types (Python 3.10+)
def process_metrics(data: list[dict[str, float]]) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}
    return results

# ❌ Avoid - importing from typing
from typing import List, Dict
def process_metrics(data: List[Dict[str, float]]) -> Dict[str, List[float]]:
    pass
```

**Modern Python Patterns:**
```python
# ✅ Good - use match-case
match media_type:
    case "image":
        handle_image()
    case "video":
        handle_video()
    case _:
        handle_unknown()

# ❌ Avoid - nested if-else
if media_type == "image":
    handle_image()
elif media_type == "video":
    handle_video()
else:
    handle_unknown()

# ✅ Good - native union syntax
def get_board(board_id: str) -> Board | None:
    return Board.load(board_id)

# ❌ Avoid - Optional from typing
from typing import Optional
def get_board(board_id: str) -> Optional[Board]:
    pass
```

**No imports in functions** (except to avoid circular imports):
```python
# ✅ Good - imports at top
from kohakuboard.client.storage import HybridStorage

def create_storage(base_dir: Path):
    storage = HybridStorage(base_dir)
    return storage

# ❌ Avoid - imports in function
def create_storage(base_dir: Path):
    from kohakuboard.client.storage import HybridStorage
    storage = HybridStorage(base_dir)
    return storage
```

**Code formatting:**
- Use `black` for code formatting
- Line length: 100 characters (black default is 88, we use 100)
- Use `asyncio.gather()` for parallel async operations (NOT sequential await in loops)

---

### File Structure

KohakuBoard follows a clean, modular architecture:

```
src/kobo/
├── client/                    # Client library (training-side)
│   ├── board.py              # Main Board class
│   ├── writer.py             # Background writer process
│   ├── types/                # Data types
│   │   ├── media.py          # Media (images, videos, audio)
│   │   ├── table.py          # Structured data
│   │   └── histogram.py      # Distribution tracking
│   └── storage/              # Storage backends
│       ├── base.py           # Abstract interface
│       ├── hybrid.py         # Lance + SQLite (default)
│       ├── lance.py          # Lance-only backend
│       ├── sqlite.py         # SQLite-only backend
│       └── duckdb.py         # DuckDB backend
├── api/                      # FastAPI backend (visualization server)
│   ├── routes/
│   │   ├── boards.py         # Board listing and metadata
│   │   ├── scalars.py        # Metrics retrieval
│   │   ├── media.py          # Media retrieval
│   │   └── tables.py         # Table retrieval
│   └── utils/
│       └── storage.py        # Storage access utilities
├── cli.py                    # CLI tool (kobo command)
├── main.py                   # FastAPI server entry point
└── logger.py                 # Logging utilities
```

**Decision Rules:**

1. **Client-side code** (runs in training script):
   - Goes in `client/`
   - Must be non-blocking and high-performance
   - No FastAPI dependencies

2. **Server-side code** (visualization backend):
   - Goes in `api/`
   - Read-only operations
   - FastAPI routes and utilities

3. **Storage backends**:
   - Implement `BaseStorage` interface
   - Handle both writes (client) and reads (server)
   - Each backend in separate file

4. **Data types** (Media, Table, Histogram):
   - Self-contained in `client/types/`
   - Support serialization/deserialization
   - Work with all storage backends

---

### Frontend Code Style

**Core Principles:**
- JavaScript only (no TypeScript), use JSDoc comments for type hints
- Vue 3 Composition API with `<script setup>`
- Split reusable components
- **Always** implement dark/light mode together using `dark:` classes
- Mobile responsive design
- Use `prettier` for code formatting
- UnoCSS for styling

**Example:**
```vue
<script setup>
import { ref, computed, onMounted } from 'vue'
import { boardAPI } from '@/utils/api'

// Reactive state
const boards = ref([])
const loading = ref(false)

// Computed properties
const hasBoards = computed(() => boards.value.length > 0)

// Async operations
async function fetchBoards() {
  loading.value = true
  try {
    const response = await boardAPI.list()
    boards.value = response.data
  } catch (error) {
    console.error('Failed to fetch boards:', error)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchBoards()
})
</script>

<template>
  <!-- Always support dark mode -->
  <div class="bg-white dark:bg-gray-900 text-black dark:text-white">
    <div v-if="loading" class="text-gray-500 dark:text-gray-400">
      Loading boards...
    </div>
    <div v-else-if="hasBoards">
      <div v-for="board in boards" :key="board.id">
        {{ board.name }}
      </div>
    </div>
    <div v-else class="text-gray-500 dark:text-gray-400">
      No boards found
    </div>
  </div>
</template>
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend development)
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/KohakuBlueleaf/KohakuHub.git
cd KohakuHub

# Install KohakuBoard in editable mode
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# For frontend development
cd src/kohaku-board-ui
npm install
```

---

## Development Workflow

### Client Library Development

The client library runs inside training scripts and must be non-blocking:

```bash
# Edit client code
# src/kobo/client/board.py
# src/kobo/client/writer.py
# src/kobo/client/storage/*

# Test with example script
python examples/kohakuboard_basic.py

# Check output
ls -la ./kohakuboard/
```

**Key Performance Requirements:**
- `board.log()` must return in <1µs
- Background writer must not block training
- Storage writes should be batched for efficiency
- Graceful shutdown with full queue draining

### Backend Development

The backend serves visualization data read-only:

```bash
# Start backend server to use local dir
kobo open ./kohakuboard
# Or with live reload
kobo open ./kohakuboard --reload

# For remote mode:
kobo serve

# Backend available at:
# http://localhost:48889
# API docs: http://localhost:48889/docs
```

**Key Backend Principles:**
- Read-only operations (never modify board data)
- Efficient queries (use storage backend optimizations)
- Concurrent read support
- Fast response times (<100ms for typical queries)

### Frontend Development

```bash
# Navigate to frontend directory
cd src/kohaku-board-ui

# Start dev server (proxies API to localhost:48889)
npm run dev

# Access at http://localhost:5173
```

**Frontend Features:**
- WebGL-based charts (Plotly.js) for 100K+ points
- Smoothing algorithms (EMA, Gaussian, SMA)
- Histogram navigation
- Media viewer with image grids
- Table viewer with embedded media support
- Dark/light mode toggle
- Responsive design

---

## How to Contribute

### Reporting Bugs

Create an issue with:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment info:
  - OS and version
  - Python version
  - Storage backend used
  - Example code if possible
- Logs/error messages

**Example:**
```
Title: "Queue overflow when logging 50+ histograms per step"

Environment:
- OS: Ubuntu 22.04
- Python: 3.10.12
- Storage: hybrid (Lance + SQLite)

Steps:
1. Create board: Board(name="test")
2. Log 50 histograms: board.log(**{f"hist_{i}": Histogram(...) for i in range(50)})
3. Observe queue warning

Expected: No warning, all histograms logged
Actual: Queue size warning, some histograms dropped
```

### Suggesting Features

- Check [Project Status](#project-status) first
- Open GitHub issue with `[KohakuBoard]` prefix or discuss on Discord
- Describe use case and value
- Propose implementation approach (if you have one)

**Example:**
```
Title: "[KohakuBoard] Support for 3D point cloud logging"

Use Case:
I'm training a 3D object detection model and want to log point cloud predictions
alongside images. Currently, only 2D media is supported.

Proposed Implementation:
- Add PointCloud data type in client/types/pointcloud.py
- Store as compressed numpy arrays (.npz)
- Visualize using Three.js in frontend

Value:
Enables KohakuBoard to support 3D vision tasks (autonomous driving, robotics, etc.)
```

### Contributing Code

1. **Pick an issue** or create one for discussion
2. **Fork** the repository
3. **Create a branch**: `git checkout -b feature/my-feature`
4. **Make changes** following style guidelines
5. **Test thoroughly** (see Testing section)
6. **Submit pull request** with clear description

---

## Best Practices

### Adding a New Data Type

1. Create type class in `src/kobo/client/types/`
2. Implement serialization/deserialization
3. Add storage support in each backend
4. Update API routes for retrieval
5. Add frontend visualization
6. Document usage in examples

**Example: Adding Audio support**
```python
# src/kobo/client/types/audio.py
from pathlib import Path
import numpy as np

class Audio:
    """Audio data type for logging sound clips."""

    def __init__(
        self,
        data: np.ndarray | Path | str,
        sample_rate: int = 44100,
        caption: str = None
    ):
        # Implementation
        pass

    def to_dict(self) -> dict:
        """Serialize for storage."""
        pass

    @classmethod
    def from_dict(cls, data: dict) -> "Audio":
        """Deserialize from storage."""
        pass
```

### Adding a New Storage Backend

1. Implement `BaseStorage` interface in `src/kobo/client/storage/`
2. Support all data types (scalars, media, tables, histograms)
3. Ensure thread-safe writes and concurrent reads
4. Add tests for performance and correctness
5. Document backend characteristics

**Required Methods:**
```python
from kohakuboard.client.storage.base import BaseStorage

class MyStorage(BaseStorage):
    def write_scalars(self, step: int, global_step: int, metrics: dict):
        """Write scalar metrics."""
        pass

    def write_media(self, step: int, global_step: int, media_list: list):
        """Write media entries."""
        pass

    def write_table(self, step: int, global_step: int, name: str, table):
        """Write table data."""
        pass

    def write_histogram(self, step: int, global_step: int, name: str, histogram):
        """Write histogram data."""
        pass

    def read_scalars(self, metric_name: str) -> list:
        """Read metric time series."""
        pass

    # ... other read methods
```

### Adding a Frontend Component

1. Create component in `src/kohaku-board-ui/src/components/`
2. Use Composition API with `<script setup>`
3. Support dark/light mode
4. Make responsive (mobile-friendly)
5. Add to relevant page

**Example:**
```vue
<!-- src/kohaku-board-ui/src/components/MetricCard.vue -->
<template>
  <div class="p-4 rounded-lg bg-white dark:bg-gray-800 shadow">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
      {{ metric.name }}
    </h3>
    <p class="text-2xl font-bold text-blue-600 dark:text-blue-400">
      {{ metric.value.toFixed(4) }}
    </p>
  </div>
</template>

<script setup>
defineProps({
  metric: {
    type: Object,
    required: true
  }
})
</script>
```

---

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_board.py::test_log_scalars

# Run with coverage
python -m pytest --cov=kobo tests/
```

### Integration Tests

Test full workflows with examples:

```bash
# Test basic logging
python examples/kohakuboard_basic.py

# Test all data types
python examples/kohakuboard_all_media_types.py

# Test CIFAR-10 training (longer)
python examples/kohakuboard_cifar_training.py
```

### Performance Tests

Benchmark critical paths:

```bash
# Benchmark logging latency
python tests/benchmark_logging.py

# Benchmark storage backends
python tests/benchmark_storage.py

# Results should show:
# - board.log() < 1µs per call
# - Throughput > 20,000 metrics/sec
# - Queue handling 50,000 messages
```

### Frontend Tests

```bash
cd src/kohaku-board-ui

# Run dev server
npm run dev

# Manual testing checklist:
# - [ ] Dark/light mode toggle works
# - [ ] Charts render 100K+ points smoothly
# - [ ] Smoothing algorithms apply correctly
# - [ ] Media viewer displays images/videos
# - [ ] Table viewer shows embedded images
# - [ ] Histogram navigation works
# - [ ] Mobile layout is responsive
```

---

## Pull Request Process

1. **Before submitting:**
   - Update documentation (README.md, docs/kohakuboard/)
   - Add tests for new functionality
   - Ensure code follows style guidelines
   - Run `black src/kobo/` for Python formatting
   - Run `npm run format` for frontend formatting
   - Test with multiple storage backends (hybrid, duckdb, parquet)
   - Verify performance (no regression in logging latency)

2. **Submitting PR:**
   - Create clear, descriptive title with `[KohakuBoard]` prefix
   - Describe what changes were made and why
   - Link related issues
   - Include screenshots for UI changes
   - List any breaking changes
   - Request review from @KohakuBlueleaf

3. **After submission:**
   - Address feedback promptly
   - Keep PR focused (split large changes into multiple PRs)
   - Rebase on main if needed

---

## Project Status

*Last Updated: January 2025*

### ✅ Core Features (Complete)

**Client Library:**
- Non-blocking logging architecture (multiprocessing queue + background writer)
- Rich data types (scalars, images, videos, tables, histograms)
- Hybrid storage backend (Lance + SQLite)
- Alternative backends (DuckDB, Parquet)
- Graceful shutdown with queue draining
- Output capture (stdout/stderr)
- Content-addressed media storage (SHA-256 deduplication)
- Histogram compression (99.8% size reduction)

**Backend Server:**
- FastAPI-based REST API
- Read-only board access
- Efficient metrics queries
- Media file serving
- Table data retrieval
- Histogram data retrieval
- Board listing and metadata

**Web UI:**
- Vue 3 interface with dark/light mode
- WebGL charts (Plotly.js) for 100K+ points
- Smoothing algorithms (EMA, Gaussian, SMA)
- X-axis selection (step, global_step, any metric)
- Media viewer (images, videos)
- Table viewer with embedded media
- Histogram navigator
- Responsive design

**CLI Tool:**
- `kobo open` - Local board viewer
- `kobo serve` - Start backend server
- Browser auto-launch option

### 🚧 In Progress

- Remote server mode with authentication
- Sync protocol for uploading local boards to remote
- Project management (group related boards)
- Run comparison UI (side-by-side metrics)
- Real-time streaming (live updates while training)

### 📋 Planned Features

**Client Features:**
- PyTorch Lightning integration
- Keras callback
- Hugging Face Trainer integration
- Custom callback system

**Backend Features:**
- Multi-board comparison API
- Advanced filtering (by tags, date range)
- Export to CSV/JSON
- Aggregation queries (mean, max, min across runs)

**UI Features:**
- Diff viewer (compare two runs)
- Scatter plots (metric vs metric)
- Distribution plots (histograms across runs)
- Custom dashboards (save/load layouts)
- Annotations (mark important steps)
- Search and filter boards

**Infrastructure:**
- Docker deployment
- Kubernetes support
- Cloud storage backends (S3, GCS, Azure Blob)
- Authentication and multi-user support

---

## Development Areas

We're especially looking for help in:

### 🎨 Frontend (High Priority)
- Chart improvements (new plot types, interactions)
- UI/UX enhancements
- Mobile responsiveness
- Accessibility features
- Dashboard customization

### 🔧 Backend
- Performance optimizations (query caching, indexing)
- Alternative storage backends (ClickHouse, TimescaleDB)
- Real-time streaming protocol
- Sync protocol implementation

### 📊 Client Library
- Framework integrations (PyTorch Lightning, Keras, etc.)
- Additional data types (3D point clouds, audio, etc.)
- Performance optimizations (reduce latency, increase throughput)
- Memory usage optimization

### 📚 Documentation
- Tutorial videos
- Best practices guide
- Deployment guides
- Architecture deep-dive
- Comparison with alternatives (WandB, MLflow, TensorBoard)

### 🧪 Testing
- Unit test coverage (target >80%)
- Integration tests
- Performance benchmarks
- Stress tests (queue overflow, disk full, etc.)

---

## Community

- **Discord:** https://discord.gg/xWYrkyvJ2s (Use #kohakuboard channel)
- **GitHub Issues:** https://github.com/KohakuBlueleaf/KohakuHub/issues (Tag with `kohakuboard`)

---

## License and Copyright

By contributing to KohakuBoard, you agree to the following:

1. **License Grant**: Your contributions will be licensed under the AGPLv3 License OR **Kohaku Software License 1.0** (Non-Commercial with Trial), depends on specific module you are working on.

2. **Commercial Licensing Rights**: You grant KohakuBlueLeaf (the project owner) perpetual, irrevocable rights to:
   - Relicense your contributions under commercial terms
   - Include your contributions in commercial licenses sold to third parties
   - Use your contributions in any way necessary for the commercial operation of this project

3. **Copyright**: You retain copyright to your contributions, but grant the above license rights to the project.

**Note:** See LICENSE file for full license terms.

---

Thank you for contributing to KohakuBoard!
