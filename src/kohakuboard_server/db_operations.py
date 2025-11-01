"""Synchronous database operations for KohakuBoard.

Compatible with KohakuHub schema for future integration.
All operations use db.atomic() for transaction safety.
"""

from datetime import datetime, timezone

from kohakuboard_server.db import (
    Board,
    EmailVerification,
    Session,
    Token,
    User,
    UserOrganization,
    db,
)


# ===== User operations =====


def get_user_by_id(user_id: int) -> User | None:
    """Get user by ID (works for both users and orgs)."""
    return User.get_or_none(User.id == user_id)


def get_user_by_username(username: str) -> User | None:
    """Get user by username (works for both users and orgs)."""
    return User.get_or_none(User.username == username)


def get_user_by_email(email: str) -> User | None:
    """Get user by email (only for regular users, orgs have no email)."""
    return User.get_or_none((User.email == email) & (User.is_org == False))


def create_user(
    username: str,
    normalized_name: str,
    email: str,
    password_hash: str,
    email_verified: bool = False,
    is_active: bool = True,
) -> User:
    """Create a new regular user.

    NOTE: Wrap in db.atomic() if checking existence first to prevent race conditions.
    """
    return User.create(
        username=username,
        normalized_name=normalized_name,
        is_org=False,
        email=email,
        password_hash=password_hash,
        email_verified=email_verified,
        is_active=is_active,
    )


def update_user(user: User, **fields) -> None:
    """Update user fields."""
    for key, value in fields.items():
        setattr(user, key, value)
    user.save()


def delete_user(user: User) -> None:
    """Delete a user and all related data (CASCADE via ForeignKey)."""
    user.delete_instance()


# ===== Organization operations =====


def get_organization(name: str) -> User | None:
    """Get organization by name (User with is_org=TRUE)."""
    return User.get_or_none((User.username == name) & (User.is_org == True))


def create_organization(
    name: str,
    normalized_name: str,
    description: str | None = None,
) -> User:
    """Create a new organization (User with is_org=TRUE)."""
    return User.create(
        username=name,
        normalized_name=normalized_name,
        is_org=True,
        email=None,
        password_hash=None,
        description=description,
        email_verified=False,
        is_active=True,
    )


def list_organization_members(org: User) -> list[UserOrganization]:
    """List all members of an organization using backref."""
    return list(org.members)  # Use backref from User.members


def list_user_organizations(user: User) -> list[UserOrganization]:
    """List all organizations a user belongs to using backref."""
    return list(user.org_memberships)  # Use backref from User.org_memberships


# ===== UserOrganization operations =====


def get_user_organization(user: User, organization: User) -> UserOrganization | None:
    """Get user-organization membership."""
    return UserOrganization.get_or_none(
        (UserOrganization.user == user)
        & (UserOrganization.organization == organization)
    )


def create_user_organization(
    user: User, organization: User, role: str
) -> UserOrganization:
    """Add user to organization with specified role."""
    return UserOrganization.create(
        user=user,
        organization=organization,
        role=role,
    )


def update_user_organization(membership: UserOrganization, **fields) -> None:
    """Update user-organization membership fields."""
    for key, value in fields.items():
        setattr(membership, key, value)
    membership.save()


def delete_user_organization(membership: UserOrganization) -> None:
    """Remove user from organization."""
    membership.delete_instance()


# ===== Session operations =====


def create_session(
    session_id: str,
    user: User,
    secret: str,
    expires_at: datetime,
) -> Session:
    """Create a new session."""
    return Session.create(
        session_id=session_id,
        user=user,
        secret=secret,
        expires_at=expires_at,
    )


def delete_session(session: Session) -> None:
    """Delete a session."""
    session.delete_instance()


# ===== Token operations =====


def create_token(user: User, token_hash: str, name: str) -> Token:
    """Create a new API token."""
    return Token.create(
        user=user,
        token_hash=token_hash,
        name=name,
    )


def delete_token(token_id: int) -> None:
    """Delete a token by ID."""
    Token.delete().where(Token.id == token_id).execute()


def list_user_tokens(user: User) -> list[Token]:
    """List all tokens for a user using backref."""
    return list(user.tokens)


# ===== EmailVerification operations =====


def create_email_verification(
    user: User,
    token: str,
    expires_at: datetime,
) -> EmailVerification:
    """Create email verification token."""
    return EmailVerification.create(
        user=user,
        token=token,
        expires_at=expires_at,
    )


def get_email_verification(token: str) -> EmailVerification | None:
    """Get email verification by token."""
    return EmailVerification.get_or_none(EmailVerification.token == token)


def delete_email_verification(verification: EmailVerification) -> None:
    """Delete email verification token."""
    verification.delete_instance()
