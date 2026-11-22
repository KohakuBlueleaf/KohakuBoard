# `kohakuboard.reader`

Read-only SWMR Python readers (`RunReader` and `ProjectReader`) over `kohakuboard._rs`.

## Boundaries
- Depends only on `kohakuboard._rs`, `kohakuboard.errors`, and `numpy`.
- Never imports `kohakuboard.client`, `kohakuboard.tui`, `kohakuboard_web`, `fastapi`, or `uvicorn`.
- Opens SQLite strictly read-only so active writers are never blocked.
