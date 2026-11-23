# KohakuBoard v2 Examples

All examples run directly against the local Rust + KohakuVault 1.2 core with zero external servers.

| Script | What it demonstrates |
| --- | --- |
| `quick_start.py` | Drop-in `kobo.init()` / `kobo.log()` workflow (no `with` required) + offline `ProjectReader` / `RunReader` reads |
| `all_features_demo.py` | Explicit `board = Board(...)` multimodal tour: scalars (`NaN`/`Inf`), `Media` (dedup), `Table` (embedded `Media` + spill), `Histogram`, FFT `KernelDensity`, `TensorLog`, `Artifact`, `alert`, `progress`, FTS5 log search, SWMR live read, and `resume="must"` |
| `tui_training_demo.py` | Live in-training KohakuTUI monitor (`Board(tui=TuiConfig(...))`) with Inline ↔ Fullscreen (`ctrl+t`), high-cardinality metrics, fuzzy `Picker` (`p`), captured logs, and scrollback sealing on finish (requires `kohakuboard[tui]`) |
| `tui_ssh_viewer_demo.py` | Generates a 3-run multimodal project and launches the standalone SSH terminal viewer (`kobo tui`) with multi-run overlay charts, half-block `Media`, `Distributions`, FTS5 `Logs`, and config `Diff` |
| `benchmark.py` | Reproducible benchmark suite for scalar hot-path latency, 1,000-metric single-DB scaling, newest-first 500k-row binary chunk reads, and off-GIL rich compute |

## Running

```bash
python examples/quick_start.py
python examples/all_features_demo.py
python examples/tui_training_demo.py --mode inline --high-cardinality
python examples/tui_ssh_viewer_demo.py
python examples/benchmark.py
```

After generating runs, inspect them via CLI, terminal viewer, or local web UI:

```bash
kobo ls ./kohakuboard_examples
kobo info ./kohakuboard_examples/demo/<run_dir>
kobo logs ./kohakuboard_examples/demo/<run_dir> --search "NaN"
kobo tui ./kohakuboard_examples
kobo watch ./kohakuboard_examples/demo/<run_dir>
kobo open ./kohakuboard_examples
```
