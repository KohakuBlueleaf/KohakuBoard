"""Authorization and permission checking for KohakuBoard."""

from fastapi import HTTPException

from kohakuboard_server.db import Board, User
from kohakuboard_server.db_operations import get_organization, get_user_organization


def check_board_read_permission(board: Board, user: User | None = None) -> bool:
    """Check if user can read a board.

    Public boards: anyone can read
    Private boards: only owner or org members can read

    Args:
        board: The board to check
        user: The authenticated user (optional for public boards)

    Returns:
        True if user has permission

    Raises:
        HTTPException: If user doesn't have permission
    """
    # Public boards are accessible to everyone
    if not board.private:
        return True

    # Private boards require authentication
    if not user:
        raise HTTPException(
            401, detail="Authentication required to access private board"
        )

    # Check if user is the owner
    if board.owner.id == user.id:
        return True

    # Check if owner is an organization and user is a member
    if board.owner.is_org:
        membership = get_user_organization(user, board.owner)
        if membership:
            return True

    raise HTTPException(
        403, detail=f"You don't have access to private board '{board.run_id}'"
    )


def check_board_write_permission(board: Board, user: User) -> bool:
    """Check if user can modify a board (sync/update).

    Users can modify:
    - Their own boards
    - Boards in orgs where they are member/admin/super-admin

    Args:
        board: The board to check
        user: The authenticated user

    Returns:
        True if user has permission

    Raises:
        HTTPException: If user doesn't have permission
    """
    if not user:
        raise HTTPException(403, detail="Authentication required")

    # Check if user owns the board
    if board.owner.id == user.id:
        return True

    # Check if owner is an organization and user is a member
    if board.owner.is_org:
        membership = get_user_organization(user, board.owner)
        if membership and membership.role in ["member", "admin", "super-admin"]:
            return True

    raise HTTPException(
        403, detail=f"You don't have permission to modify board '{board.run_id}'"
    )


def check_project_access(
    project_owner: User, current_user: User | None, require_write: bool = False
) -> bool:
    """Check if user can access a project.

    Args:
        project_owner: Owner of the project (user or org)
        current_user: Current authenticated user
        require_write: If True, check write permission; if False, check read permission

    Returns:
        True if user has access

    Raises:
        HTTPException: If user doesn't have access
    """
    # For read access of public projects
    if not require_write and not current_user:
        # Will be checked at board level (public vs private)
        return True

    if not current_user:
        raise HTTPException(401, detail="Authentication required")

    # User's own project
    if project_owner.id == current_user.id:
        return True

    # Organization project - check membership
    if project_owner.is_org:
        membership = get_user_organization(current_user, project_owner)
        if membership:
            if require_write:
                # Write requires member+ role
                return membership.role in ["member", "admin", "super-admin"]
            else:
                # Read allowed for any member
                return True

    if require_write:
        raise HTTPException(403, detail=f"You don't have permission to modify project")
    else:
        raise HTTPException(403, detail=f"You don't have access to this project")
