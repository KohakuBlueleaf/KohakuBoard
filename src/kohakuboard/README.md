# `kohakuboard`

Local-first, zero-overhead ML experiment tracking backed by Rust and KohakuVault 1.2.

## Subpackages
- `client`: `Board` logger, rich payload dataclasses, friendly naming, stdout/stderr capture.
- `reader`: Read-only SWMR `RunReader` and `ProjectReader`.
- `cli`: Stdlib `argparse` `kobo` CLI.
- `errors`: Typed `KohakuBoardError` hierarchy.
- `_rs`: Native PyO3 extension module built from `src/kohakuboard-rust/`.
