# `kohakuboard.client`

Python logging SDK (`Board`, rich data types, friendly run naming, and stdout/stderr tee capture).

## Boundaries
- Depends only on `kohakuboard._rs`, `kohakuboard.errors`, `kohakuboard.data`, and `numpy`.
- Never imports `kohakuboard_web`, `fastapi`, or `uvicorn`.
- Optional terminal UI integration is guarded behind `ConfigError` until `kohakuboard[tui]` support is enabled.
