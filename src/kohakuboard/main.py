"""Main FastAPI application for KohakuBoard (Local Mode)

Local mode API - no authentication, no database required.
For full server with auth/DB, use kohakuboard_server instead.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kohakuboard.config import cfg
from kohakuboard.logger import logger_api
from kohakuboard.utils.board_reader import list_boards

logger_api.info("Running KohakuBoard in local mode (no authentication, no database)")

# Import routers
from kohakuboard.api import boards, projects, runs, system

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


@app.get("/")
async def root():
    """Root endpoint with API info"""
    try:
        boards = list_boards(Path(cfg.app.board_data_dir))
        board_count = len(boards)
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
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "mode": "local"}


if __name__ == "__main__":
    import uvicorn

    logger_api.info(f"Starting KohakuBoard API on {cfg.app.host}:{cfg.app.port}")
    uvicorn.run(
        "kohakuboard.main:app", host=cfg.app.host, port=cfg.app.port, reload=True
    )
