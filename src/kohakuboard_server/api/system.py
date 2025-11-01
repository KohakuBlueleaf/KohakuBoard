"""System information API endpoint for server"""

from fastapi import APIRouter

from kohakuboard_server.config import cfg

router = APIRouter()


@router.get("/system/info")
async def get_system_info():
    """Get system information

    Returns mode and version info.
    Respects no_auth flag for testing.
    """
    return {
        "mode": "remote",
        "require_auth": not cfg.app.no_auth,  # False when --no-auth flag is used
        "version": "0.1.0",
    }
