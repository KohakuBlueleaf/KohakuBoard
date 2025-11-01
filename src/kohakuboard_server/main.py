"""Main FastAPI application for KohakuBoard Server

Full-featured server with authentication, database, and multi-user support.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kohakuboard_server.config import cfg
from kohakuboard_server.logger import logger_api
from kohakuboard_server.db import init_db

# Initialize database
logger_api.info("Initializing database for KohakuBoard Server")
logger_api.info(f"  Backend: {cfg.app.db_backend}")
logger_api.info(f"  URL: {cfg.app.database_url}")
init_db(cfg.app.db_backend, cfg.app.database_url)
logger_api.info("✓ Database initialized")

# Import routers after DB initialization
from kohakuboard_server.api import boards, org, projects, runs, sync, system
from kohakuboard_server.auth import router as auth_router

app = FastAPI(
    title="KohakuBoard Server API",
    description="ML Experiment Tracking Server - Full featured with auth & DB",
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

# Register auth router
app.include_router(auth_router, prefix=cfg.app.api_base, tags=["auth"])

# Register organization router
app.include_router(org.router, prefix=f"{cfg.app.api_base}/org", tags=["organizations"])

# Register sync router (server-only endpoints)
app.include_router(sync.router, prefix=cfg.app.api_base, tags=["sync"])

# Register project/run routers (from kohakuboard - work in remote mode)
app.include_router(system.router, prefix=cfg.app.api_base, tags=["system"])
app.include_router(projects.router, prefix=cfg.app.api_base, tags=["projects"])
app.include_router(runs.router, prefix=cfg.app.api_base, tags=["runs"])

# Keep legacy boards router for backward compatibility
app.include_router(boards.router, prefix=cfg.app.api_base, tags=["boards (legacy)"])


@app.get("/")
async def root():
    """Root endpoint with server info"""
    from kohakuboard_server.api.boards import list_boards

    try:
        boards_list = list_boards(Path(cfg.app.board_data_dir))
        board_count = len(boards_list)
    except Exception:
        board_count = 0

    return {
        "name": "KohakuBoard Server",
        "version": "0.1.0",
        "description": "ML Experiment Tracking Server - Full featured",
        "mode": "remote",
        "board_data_dir": cfg.app.board_data_dir,
        "board_count": board_count,
        "docs": f"{cfg.app.api_base}/docs",
        "endpoints": {
            "system": f"{cfg.app.api_base}/system/info",
            "projects": f"{cfg.app.api_base}/projects",
            "auth": f"{cfg.app.api_base}/auth",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "mode": "remote"}


if __name__ == "__main__":
    import uvicorn

    logger_api.info(f"Starting KohakuBoard Server on {cfg.app.host}:{cfg.app.port}")
    uvicorn.run(
        "kohakuboard_server.main:app", host=cfg.app.host, port=cfg.app.port, reload=True
    )
