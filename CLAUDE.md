# KohakuBoard v2: repo notes

Local-first, zero-overhead ML experiment tracking with a Rust logging & query core
(`kohakuboard._rs`) over KohakuVault 1.2.0 SQLite storage, a Python client/reader/CLI SDK,
and a separate Web fullstack package (`packages/kohakuboard-web/` + `apps/web/`).

Architecture plans live in `.internal/v2-plan/` (local, gitignored).

## Layout

- `src/kohakuboard-rust/`: native core (`kohakuboard._rs` with `--features python`).
  - `error/`: `BoardError` (`thiserror`) and PyErr conversion
  - `schema/`: packed binary records (`ScalarRow`, `HistHeader`, `KdeHeader`) and codecs
  - `storage/`: `RunStore` (writer), `RunReader` (read-only), `Catalog` (`.catalog.db`)
  - `compute/`: off-GIL histogram binning, FFT-KDE, blake3 hashing, PNG/WebP encoding
  - `sysmon/`: CPU/RAM (`sysinfo`) and optional dynamic NVML GPU sampler
  - `engine/`: bounded crossbeam channel, writer thread, WAL flush, live snapshot ring
  - `query/`: newest-first reverse-paged binary chunk framing and PyO3 reader bindings
- `src/kohakuboard/`: Apache-2.0 Python SDK (`client/`, `reader/`, `cli/`, optional `tui/`).
- `packages/kohakuboard-web/`: Kohaku Software License 1.0 FastAPI + Uvicorn server.
- `apps/web/`: Vue 3 + Vite 8 + UnoCSS (KohakuTerrarium v2 theme) + uPlot SPA.
- `tests/`: `static/`, `e2e/`, `integration/`, `unit/`.

## Development setup

- `uv venv`, then `uv pip install -e ".[dev]"`. maturin builds `kohakuboard._rs` into `.venv`;
  rerun after any Rust change.
- Never use `sys.path.insert` hacks in examples or tests.

## Rust rules

1. Every module is a directory whose only entry point is `mod.rs` (`tests/static/test_layout.py`
   enforces no `name.rs` beside `name/`).
2. Unit tests live in `<module>/tests.rs` (`#[cfg(test)] mod tests;` in `mod.rs`).
3. PyO3 code lives only in `<module>/python.rs` behind `#[cfg(feature = "python")]`, exposes
   `register(m)`, and is wired from `lib.rs`. Pure Rust never imports `pyo3`, so
   `cargo test --no-default-features` links zero Python.
4. Pyclasses are underscore-prefixed (`_LoggerEngine`, `_RunReader`, `_Catalog`) and wrapped by
   public Python classes.
5. `thiserror` error enum per module; never panic on Python or wire input.
6. `cargo fmt --check` (max width 100) and `cargo clippy --all-targets -- -D warnings`, both
   with and without `--features python`.

## Python rules

- Target Python 3.10+ (`abi3` wheel). Modern hints: `list`, `dict`, `X | None`.
- Prefer `match`/`case` over long `if`/`elif` chains.
- No `print()` in library code (CLI entry points under `cli/` may print).
- No imports inside functions except allowlisted optional dependencies (`kohakutui`, `torch`,
  `PIL`, `kohakuboard_web`).
- `black --line-length 100 src tests scripts` and `ruff check src tests scripts`.
- File size cap (`tests/static/test_file_sizes.py`): over 500 AST statements warns, over 1000 fails.
- Import boundaries (`scripts/check_imports.py`): `client` & `reader` never import `tui` or
  `kohakuboard_web`; `tui` never imports `kohakuboard_web`; `kohakuboard_web` never imports
  `client` writer internals or `tui`.

## Tests & Audit Loop

- Write tests **e2e -> integration -> unit** with real collaborators (real temp DBs, no fake storage).
- Audit loop: implement -> negative-case tests -> run Rust + Python suites + lint + static guards
  -> audit diff for integrity/race/behavior bugs -> make a failing test first, then fix.
