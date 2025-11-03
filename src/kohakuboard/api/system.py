"""System information API endpoints"""

from fastapi import APIRouter


router = APIRouter()


@router.get("/system/info")
async def get_system_info():
    """Get system information for frontend configuration

    Returns mode, authentication requirements, and version.
    Frontend uses this to determine UI behavior.

    Returns:
        dict: System info with mode, require_auth, version
    """
    return {
        "mode": "local",
        "require_auth": False,
        "version": "0.1.0",
    }
