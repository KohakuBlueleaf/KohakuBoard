"""System information API endpoints"""

from fastapi import APIRouter

from kohakuboard.config import cfg

router = APIRouter()


@router.get("/system/info")
async def get_system_info():
    """Get system information for frontend configuration

    Returns mode, authentication requirements, and version.
    Frontend uses this to determine UI behavior.

    NO AUTH DEPENDENCY - This endpoint must work in local mode without database.

    Returns:
        dict: System info with mode, require_auth, version
    """
    info = {
        "mode": cfg.app.mode,
        "require_auth": cfg.app.mode == "remote",
        "version": "0.1.0",
    }

    return info
