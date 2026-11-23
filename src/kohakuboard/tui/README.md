# `kohakuboard.tui`

Terminal UI integration powered by `kohakutui` (optional extra `kohakuboard[tui]`).

| File | Purpose |
| --- | --- |
| `config.py` | Pure-stdlib `TuiConfig` and `ChartSpec` declarative configuration dataclasses. |
| `helpers.py` | Formatting, metric auto-selection, namespace grouping, EMA smoothing, and histogram/KDE bar rebinning. |
| `adapter.py` | `LiveTuiHandle` live in-training monitor (`Board(tui=...)`) polling Rust snapshots at `refresh_hz`. |
| `viewer.py` | Standalone multi-run SSH/offline viewer (`kobo tui` and `kobo watch`) built on `ProjectReader` and `RunReader`. |
| `__init__.py` | Public TUI exports and CLI entry points (`run_tui`, `run_watch`). |
