"""Main FastAPI application for KohakuBoard (Local Mode)

Local mode API - no authentication, no database required.
For full server with auth/DB, use kohakuboard_server instead.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kohakuboard.config import cfg
from kohakuboard.logger import logger_api
from kohakuboard.utils.board_reader import list_boards

logger_api.info("Running KohakuBoard in local mode (no authentication, no database)")

# Import routers
from kohakuboard.api import boards, projects, runs, system

# Determine static files directory (built frontend)
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="KohakuBoard API (Local Mode)",
    description="ML Experiment Tracking API - Local mode without authentication",
    version="0.1.0",
    docs_url=f"{cfg.app.api_base}/docs",
    openapi_url=f"{cfg.app.api_base}/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register project/run routers (work in local mode)
app.include_router(system.router, prefix=cfg.app.api_base, tags=["system"])
app.include_router(projects.router, prefix=cfg.app.api_base, tags=["projects"])
app.include_router(runs.router, prefix=cfg.app.api_base, tags=["runs"])

# Keep legacy boards router for backward compatibility
app.include_router(boards.router, prefix=cfg.app.api_base, tags=["boards (legacy)"])


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "mode": "local"}


# Mount static files and SPA fallback if static directory exists
if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    logger_api.info(f"Serving static frontend from {STATIC_DIR}")

    # Mount static files directories
    if (STATIC_DIR / "assets").exists():
        app.mount(
            "/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets"
        )
    if (STATIC_DIR / "images").exists():
        app.mount(
            "/images", StaticFiles(directory=STATIC_DIR / "images"), name="images"
        )
    if (STATIC_DIR / "docs").exists():
        app.mount(
            "/static-docs",
            StaticFiles(directory=STATIC_DIR / "docs"),
            name="static-docs",
        )

    @app.get("/favicon.svg")
    async def favicon():
        """Serve favicon"""
        favicon_path = STATIC_DIR / "favicon.svg"
        if favicon_path.exists():
            return FileResponse(favicon_path, media_type="image/svg+xml")
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """SPA fallback - serve index.html for all non-API routes"""
        # Try to serve the exact file if it exists
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Otherwise, serve index.html for SPA routing
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

else:
    logger_api.warning(
        f"Static frontend not found at {STATIC_DIR}. "
        "Run 'npm run build' in src/kohaku-board-ui/ to build the frontend."
    )

    @app.get("/")
    async def root():
        """Root endpoint with API info (no frontend available)"""
        try:
            boards_list = list_boards(Path(cfg.app.board_data_dir))
            board_count = len(boards_list)
        except Exception:
            board_count = 0

        return {
            "name": "KohakuBoard API (Local Mode)",
            "version": "0.1.0",
            "description": "ML Experiment Tracking - Local mode without authentication",
            "mode": "local",
            "board_data_dir": cfg.app.board_data_dir,
            "board_count": board_count,
            "docs": f"{cfg.app.api_base}/docs",
            "endpoints": {
                "system": f"{cfg.app.api_base}/system/info",
                "projects": f"{cfg.app.api_base}/projects",
            },
            "note": "Frontend not available. Build with 'npm run build' in src/kohaku-board-ui/",
        }


if __name__ == "__main__":
    import uvicorn

    logger_api.info(f"Starting KohakuBoard API on {cfg.app.host}:{cfg.app.port}")
    uvicorn.run(
        "kohakuboard.main:app", host=cfg.app.host, port=cfg.app.port, reload=True
    )
