"""Access control & loopback DNS-rebinding / cross-origin guard for local serving."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _extract_host(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("["):
        end = raw.find("]")
        return raw[: end + 1] if end != -1 else raw
    return raw.split(":", 1)[0]


def is_loopback_host(host_header: str | None, *, allow_testserver: bool = False) -> bool:
    """Return True if the HTTP Host header targets a local loopback interface."""
    if not host_header:
        return True
    host = _extract_host(host_header)
    if host in _LOOPBACK_HOSTS:
        return True
    return allow_testserver and host == "testserver"


def is_loopback_origin(origin_header: str | None, *, allow_testserver: bool = False) -> bool:
    """Return True if an Origin header is absent or targets a local loopback host."""
    if not origin_header or origin_header == "null":
        return True
    parsed = urlparse(origin_header)
    host = (parsed.hostname or "").lower()
    if host in _LOOPBACK_HOSTS or f"[{host}]" in _LOOPBACK_HOSTS:
        return True
    return allow_testserver and host == "testserver"


class LoopbackGuardMiddleware(BaseHTTPMiddleware):
    """Reject DNS-rebinding Host headers and cross-origin mutating requests in local mode."""

    def __init__(self, app: object, *, allow_testserver: bool = False) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._allow_testserver = allow_testserver

    async def dispatch(self, request: Request, call_next: object) -> object:
        host = request.headers.get("host")
        if not is_loopback_host(host, allow_testserver=self._allow_testserver):
            return JSONResponse(
                {"detail": f"Rejected non-loopback Host header: {host!r}"},
                status_code=403,
            )
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and not is_loopback_origin(
            origin, allow_testserver=self._allow_testserver
        ):
            return JSONResponse(
                {"detail": f"Rejected cross-origin {request.method} from {origin!r}"},
                status_code=403,
            )
        return await call_next(request)  # type: ignore[misc]
