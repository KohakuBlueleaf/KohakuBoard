# KohakuBoard - ML Experiment Tracking

High-performance, local-first experiment tracking for machine learning workflows.

---

## Overview

KohakuBoard is a **standalone sub-project** within KohakuHub, providing experiment tracking capabilities similar to Weights & Biases (WandB) but with better performance and local-first design.

**Status:** ✅ Core features complete and functional. Remote sync features are WIP.

---

## Key Features

- ✅ **Non-Blocking Logging** - Background writer process, zero training overhead
- ✅ **Rich Data Types** - Scalars, images, videos, tables, histograms
- ✅ **Hybrid Storage** - Lance (columnar) + SQLite (row-oriented) for optimal performance
- ✅ **No Step Inflation** - Log multiple values at same step
- ✅ **Local-First** - View experiments locally with `kobo open`, no server required
- ⏳ **Remote Sync** - Upload to shared server (WIP)

---

## Quick Start

### Installation

```bash
cd /path/to/KohakuHub
pip install -e src/kohakuboard/
```

### Your First Experiment

```python
from kohakuboard.client import Board, Histogram

# Create board
board = Board(name="my-experiment", config={"lr": 0.001})

# Training loop
for epoch in range(10):
    board.step()

    for batch in train_loader:
        loss = train_step(batch)
        board.log(loss=loss, lr=optimizer.lr)

    # Log histograms (all at same step!)
    hist_data = {
        f"grad/{name}": Histogram(param.grad).compute_bins()
        for name, param in model.named_parameters()
    }
    board.log(**hist_data)
```

### View Results

```bash
kobo open ./kohakuboard --browser
```

---

## Documentation

- **[Getting Started](../../docs/kohakuboard/getting-started.md)** - Installation and first experiment
- **[Architecture](../../docs/kohakuboard/architecture.md)** - Storage system and data flow
- **[API Reference](../../docs/kohakuboard/api.md)** - Complete Python API
- **[CLI Tools](../../docs/kohakuboard/cli.md)** - Command-line interface
- **[Examples](../../examples/)** - See `kohakuboard_cifar_training.py`

---

## Architecture

### Hybrid ColumnVault + SQLite Storage

After trying various approaches (DuckDB, Parquet), we settled on Lance+SQLite:

- **ColumnVault (Columnar)** - Metrics and histograms (column-oriented reads)
- **SQLite (Row-Oriented)** - Metadata, tables, media references

**Why this works:**
- Metrics are read as entire columns → ColumnVault excels
- Metadata needs random row access → SQLite excels
- Both support non-blocking operations with proper configuration

### Non-Blocking Architecture

```
Main Process              Background Writer
     │                           │
     ├─ board.log(...)           │
     │  └─> Queue.put()          │
     │      (instant!)           │
     │                           ├─ Queue.get()
     │                           ├─ Write to storage
     │                           └─ Flush
     │
     ├─ Continue training...
```

**Performance:** ~20,000 metrics/second sustained

---

## Project Structure

```
src/kohakuboard/
├── client/                    # Python client library
│   ├── board.py              # Main Board class
│   ├── writer.py             # Background writer process
│   ├── types/                # Data types (Media, Table, Histogram)
│   │   ├── media.py
│   │   ├── table.py
│   │   ├── columnar_histogram.py
│   │   └── media_handler.py
│   └── storage/              # Storage backends
│       ├── hybrid.py         # ColumnVault + SQLite (default)
│       ├── columnar.py          # Lance metrics storage
│       ├── sqlite.py         # SQLite metadata storage
│       └── columnar_histogram.py      # Histogram storage (Lance)
├── api/                      # FastAPI server (WIP)
├── cli.py                    # CLI tool (kobo command)
├── main.py                   # Server entry point
└── README.md                 # This file
```

---

## CLI Commands

```bash
# View local boards (fully working)
kobo open ./kohakuboard --browser

# Start server (WIP)
kobo serve --reload --port 48889

# Sync to remote (WIP)
kobo sync ./kohakuboard/{board_id} -r https://board.example.com -p my-project
```

---

## Development

### Install Dependencies

```bash
pip install -e src/kohakuboard/
cd src/kohaku-board-ui
npm install
```

### Run Development Server

```bash
# Backend (local mode)
kobo open ./kohakuboard --reload --port 48889

# Frontend
cd src/kohaku-board-ui
npm run dev
```

### Format Code

```bash
# Format both frontend and backend
python scripts/format_kobo.py
```

---

## Examples

### CIFAR-10 Training

See `examples/kohakuboard_cifar_training.py` for a complete example with:
- Gradient histograms
- Validation tables
- Sample prediction images
- Namespace organization

### Simple Training Loop

```python
from kohakuboard.client import Board, Histogram, Table, Media

board = Board(name="resnet-training", config={"lr": 0.001})

for epoch in range(100):
    board.step()

    # Training
    for batch in train_loader:
        loss = train_step(batch)
        board.log(**{"train/loss": loss})

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

## Performance

| Metric | Value |
|--------|-------|
| Log call latency | < 1µs (non-blocking) |
| Throughput | ~20,000 metrics/second |
| Queue capacity | 50,000 messages |
| Memory (writer) | ~100-200 MB |

---

## Roadmap

### ✅ Complete

- [x] Python client library
- [x] Hybrid ColumnVault+SQLite storage
- [x] Rich data types
- [x] Non-blocking async logging
- [x] Local viewer (`kobo open`)
- [x] CLI tool
- [x] Documentation

### ⏳ In Progress

- [ ] Remote server authentication
- [ ] Project management
- [ ] Sync protocol
- [ ] Frontend improvements

### 🔮 Planned

- [ ] Real-time streaming
- [ ] Run comparison UI
- [ ] PyTorch Lightning integration

---

## License

**Kohaku Software License 1.0** (Non-Commercial with Trial)

- Free for non-commercial use
- Commercial trial: 3 months OR $25k revenue/year
- After trial: Paid commercial license required

Full license: https://kblueleaf.net/documents/kohaku-code-license/?%5BYour%20Organization/Name%5D=KohakuBlueLeaf&%5BYear%5D=2025&[Your%20Contact%20Email]=kohaku@kblueleaf.net&[Your%20Jurisdiction]=Taiwan

---

## Contributing

See [../../CONTRIBUTING.md](../../CONTRIBUTING.md) for the main contributing guide.

**KohakuBoard-specific:**
- Test both local and remote modes
- Update documentation for new features
- Include examples in docstrings
- Follow the same code style as KohakuHub

---

## Support

- **Discord:** https://discord.gg/xWYrkyvJ2s
- **Issues:** https://github.com/KohakuBlueleaf/KohakuHub/issues
- **Email:** kohaku@kblueleaf.net

---

**Part of the KohakuHub Project** - https://github.com/KohakuBlueleaf/KohakuHub
