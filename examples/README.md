# KohakuBoard v2 Examples

All examples run directly against the local Rust + KohakuVault 1.2 core with zero external servers.

| Script | What it demonstrates |
| --- | --- |
| `quick_start.py` | Drop-in `kobo.init()` / `kobo.log()` workflow (no `with` required) + offline `ProjectReader` / `RunReader` reads |
| `all_features_demo.py` | Explicit `board = Board(...)` multimodal tour: scalars (`NaN`/`Inf`), `Media` (dedup), `Table` (embedded `Media` + spill), `Histogram`, FFT `KernelDensity`, `TensorLog`, `Artifact`, `alert`, `progress`, FTS5 log search, SWMR live read, and `resume="must"` |
| `benchmark.py` | Reproducible benchmark suite for scalar hot-path latency, 1,000-metric single-DB scaling, newest-first 500k-row binary chunk reads, and off-GIL rich compute |

## Running

```bash
python examples/quick_start.py
python examples/all_features_demo.py
python examples/benchmark.py
```

After running `quick_start.py` or `all_features_demo.py`, inspect the generated runs with the `kobo` CLI:

```bash
kobo ls ./kohakuboard_examples
kobo info ./kohakuboard_examples/demo/<run_dir>
kobo logs ./kohakuboard_examples/demo/<run_dir> --search "NaN"
kobo doctor ./kohakuboard_examples --reindex
```
