"""Main FastAPI application for KohakuBoard"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kohakuboard.config import cfg
from kohakuboard.logger import logger_api

# Initialize database BEFORE importing routers (they need db to be ready)
if cfg.app.mode == "remote":
    from kohakuboard.db import init_db

    logger_api.info(f"Initializing database (mode: {cfg.app.mode})")
    logger_api.info(f"  Backend: {cfg.app.db_backend}")
    logger_api.info(f"  URL: {cfg.app.database_url}")
    init_db(cfg.app.db_backend, cfg.app.database_url)
    logger_api.info("✓ Database initialized")
else:
    logger_api.info(f"Running in {cfg.app.mode} mode (no database needed)")
    # Initialize dummy database for local mode so imports don't fail
    from kohakuboard.db import db
    from peewee import SqliteDatabase

    db.initialize(SqliteDatabase(":memory:"))

# Now import routers (after db is initialized)
from kohakuboard.api import boards, org, projects, runs, sync, system
from kohakuboard.auth import router as auth_router

app = FastAPI(
    title="KohakuBoard API",
    description=f"ML Experiment Tracking API - {cfg.app.mode.title()} mode",
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

# Register auth router (remote mode uses it, local mode ignores it)
app.include_router(auth_router, prefix=cfg.app.api_base, tags=["auth"])

# Register organization router (mounted at /org/, not /api/org/)
app.include_router(org.router, prefix="/org", tags=["organizations"])

# Register project/run routers
app.include_router(system.router, prefix=cfg.app.api_base, tags=["system"])
app.include_router(projects.router, prefix=cfg.app.api_base, tags=["projects"])
app.include_router(runs.router, prefix=cfg.app.api_base, tags=["runs"])

# Register sync router (remote mode only, but always registered for API docs)
app.include_router(sync.router, prefix=cfg.app.api_base, tags=["sync"])

# Keep legacy boards router for backward compatibility (media file serving)
app.include_router(boards.router, prefix=cfg.app.api_base, tags=["boards (legacy)"])


@app.get("/")
async def root():
    """Root endpoint with API info"""
    from pathlib import Path
    from kohakuboard.api.utils.board_reader import list_boards

    try:
        boards = list_boards(Path(cfg.app.board_data_dir))
        board_count = len(boards)
    except Exception:
        board_count = 0

    return {
        "name": "KohakuBoard API",
        "version": "0.1.0",
        "description": f"ML Experiment Tracking - {cfg.app.mode.title()} mode",
        "mode": cfg.app.mode,
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
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    logger_api.info(f"Starting KohakuBoard API on {cfg.app.host}:{cfg.app.port}")
    uvicorn.run(
        "kohakuboard.main:app", host=cfg.app.host, port=cfg.app.port, reload=True
    )
