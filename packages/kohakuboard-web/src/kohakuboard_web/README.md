# `kohakuboard-web`

Unified FastAPI + Uvicorn backend and bundled SPA assets for KohakuBoard v2.

Licensed under the Kohaku Software License 1.0 (`packages/kohakuboard-web/LICENSE`).

## Boundaries
- Reads runs exclusively through `kohakuboard.reader` (`ProjectReader`, `RunReader`) — never imports `kohakuboard.client` writer internals or `kohakuboard.tui`.
- Serves newest-first reverse-paged 8-byte-aligned binary chunk streams (`/api/projects/{p}/runs/scalars/stream`), immutable content-addressed blobs with HTTP `Range` support, and SSE live high-water-mark notifications.
