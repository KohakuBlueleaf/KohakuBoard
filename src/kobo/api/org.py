"""Organization API endpoints for KohakuBoard."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from kohakuboard.auth import get_current_user
from kohakuboard.db import User, db
from kohakuboard.db_operations import (
    create_organization,
    create_user_organization,
    delete_user_organization,
    get_organization,
    get_user_by_username,
    get_user_organization,
    list_organization_members,
    list_user_organizations,
    update_user_organization,
)
from kohakuboard.logger import logger_api
from kohakuboard.utils.datetime_utils import safe_isoformat
from kohakuboard.utils.names import normalize_name

router = APIRouter()


class CreateOrganizationPayload(BaseModel):
    name: str
    description: str | None = None


class AddMemberPayload(BaseModel):
    username: str
    role: str


class UpdateMemberRolePayload(BaseModel):
    role: str


@router.post("/create")
async def create_organization_endpoint(
    payload: CreateOrganizationPayload, user: User = Depends(get_current_user)
):
    """Create a new organization.

    Args:
        payload: Organization creation data
        user: Current authenticated user

    Returns:
        Success response with organization name
    """
    logger_api.info(f"User {user.username} creating organization: {payload.name}")

    # Check if organization already exists and create atomically
    with db.atomic():
        existing_org = get_organization(payload.name)
        if existing_org:
            raise HTTPException(400, detail="Organization name already exists")

        # Check normalized name conflicts
        normalized = normalize_name(payload.name)
        existing_normalized = User.get_or_none(User.normalized_name == normalized)
        if existing_normalized:
            entity_type = "organization" if existing_normalized.is_org else "user"
            raise HTTPException(
                400,
                detail=f"Name conflicts with existing {entity_type}: {existing_normalized.username}",
            )

        # Create organization
        org = create_organization(
            name=payload.name,
            normalized_name=normalized,
            description=payload.description,
        )

        # Add creator as super-admin
        create_user_organization(user, org, "super-admin")

    logger_api.success(f"Organization created: {org.username} by {user.username}")

    return {"success": True, "name": org.username}


@router.get("/{org_name}")
async def get_organization_info(org_name: str):
    """Get organization details.

    Args:
        org_name: Organization name

    Returns:
        Organization info
    """
    org = get_organization(org_name)
    if not org:
        raise HTTPException(404, detail="Organization not found")

    return {
        "name": org.username,
        "description": org.description,
        "created_at": safe_isoformat(org.created_at),
    }


@router.post("/{org_name}/members")
async def add_member(
    org_name: str,
    payload: AddMemberPayload,
    current_user: User = Depends(get_current_user),
):
    """Add a member to an organization.

    Args:
        org_name: Organization name
        payload: Member data (username, role)
        current_user: Current authenticated user

    Returns:
        Success response
    """
    org = get_organization(org_name)
    if not org:
        raise HTTPException(404, detail="Organization not found")

    # Check if current user is admin
    user_org = get_user_organization(current_user, org)
    if not user_org or user_org.role not in ["admin", "super-admin"]:
        raise HTTPException(403, detail="Not authorized to add members")

    # Get user to add
    user_to_add = get_user_by_username(payload.username)
    if not user_to_add:
        raise HTTPException(404, detail="User not found")

    if user_to_add.is_org:
        raise HTTPException(400, detail="Cannot add organization as member")

    # Check if already a member
    existing = get_user_organization(user_to_add, org)
    if existing:
        raise HTTPException(400, detail="User is already a member of the organization")

    # Add member
    create_user_organization(user_to_add, org, payload.role)

    logger_api.success(
        f"User {user_to_add.username} added to org {org_name} as {payload.role}"
    )

    return {"success": True, "message": "Member added successfully"}


@router.delete("/{org_name}/members/{username}")
async def remove_member_endpoint(
    org_name: str,
    username: str,
    current_user: User = Depends(get_current_user),
):
    """Remove a member from an organization.

    Args:
        org_name: Organization name
        username: Username to remove
        current_user: Current authenticated user

    Returns:
        Success response
    """
    org = get_organization(org_name)
    if not org:
        raise HTTPException(404, detail="Organization not found")

    # Check if current user is admin
    user_org = get_user_organization(current_user, org)
    if not user_org or user_org.role not in ["admin", "super-admin"]:
        raise HTTPException(403, detail="Not authorized to remove members")

    # Get user to remove
    user_to_remove = get_user_by_username(username)
    if not user_to_remove:
        raise HTTPException(404, detail="User not found")

    # Get membership
    membership = get_user_organization(user_to_remove, org)
    if not membership:
        raise HTTPException(404, detail="User is not a member of the organization")

    # Remove member
    delete_user_organization(membership)

    logger_api.success(f"User {username} removed from org {org_name}")

    return {"success": True, "message": "Member removed successfully"}


@router.put("/{org_name}/members/{username}")
async def update_member_role_endpoint(
    org_name: str,
    username: str,
    payload: UpdateMemberRolePayload,
    current_user: User = Depends(get_current_user),
):
    """Update a member's role in an organization.

    Args:
        org_name: Organization name
        username: Username to update
        payload: New role
        current_user: Current authenticated user

    Returns:
        Success response
    """
    org = get_organization(org_name)
    if not org:
        raise HTTPException(404, detail="Organization not found")

    # Check if current user is admin
    user_org = get_user_organization(current_user, org)
    if not user_org or user_org.role not in ["admin", "super-admin"]:
        raise HTTPException(403, detail="Not authorized to update member roles")

    # Get user to update
    user_to_update = get_user_by_username(username)
    if not user_to_update:
        raise HTTPException(404, detail="User not found")

    # Get membership
    membership = get_user_organization(user_to_update, org)
    if not membership:
        raise HTTPException(404, detail="User is not a member of the organization")

    # Update role
    update_user_organization(membership, role=payload.role)

    logger_api.success(
        f"User {username} role updated to {payload.role} in org {org_name}"
    )

    return {"success": True, "message": "Member role updated successfully"}


@router.get("/{org_name}/members")
async def list_organization_members_endpoint(org_name: str):
    """List organization members.

    Args:
        org_name: Organization name

    Returns:
        List of members with roles
    """
    org = get_organization(org_name)
    if not org:
        raise HTTPException(404, detail="Organization not found")

    # Get all members
    members = list_organization_members(org)

    return {
        "members": [
            {
                "user": m.user.username,
                "role": m.role,
                "created_at": safe_isoformat(m.created_at),
            }
            for m in members
        ]
    }


@router.get("/users/{username}/orgs")
async def list_user_organizations_endpoint(username: str):
    """List organizations a user belongs to.

    Args:
        username: Username

    Returns:
        List of organizations with roles
    """
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(404, detail="User not found")

    if user.is_org:
        raise HTTPException(400, detail="Cannot list organizations for an organization")

    # Get user's org memberships
    orgs = list_user_organizations(user)

    return {
        "organizations": [
            {
                "name": org.organization.username,
                "description": org.organization.description,
                "role": org.role,
                "created_at": safe_isoformat(org.created_at),
            }
            for org in orgs
        ]
    }
