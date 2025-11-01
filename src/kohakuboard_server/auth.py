"""Authentication and authorization for KohakuBoard.

Separate from KohakuHub for licensing compliance.
Uses similar patterns but independent implementation.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Cookie, Header, HTTPException, Request

from kohakuboard_server.config import cfg
from kohakuboard_server.db import Board, Session, Token, User
from kohakuboard_server.logger import logger_api


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def hash_token(token: str) -> str:
    """Hash API token using SHA256"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_token() -> str:
    """Generate a new API token"""
    return "kobo_" + secrets.token_urlsafe(32)


def generate_session_id() -> str:
    """Generate a new session ID"""
    return secrets.token_urlsafe(32)


def get_current_user(
    request: Request,
    session_id: str | None = Cookie(None),
    authorization: str | None = Header(None),
) -> User:
    """Get current authenticated user from session or token.

    Requires authentication - raises 401 if not authenticated.

    Args:
        request: FastAPI request object
        session_id: Session cookie value
        authorization: Authorization header value

    Returns:
        User object if authenticated

    Raises:
        HTTPException: 401 if not authenticated
    """
    if cfg.app.mode != "remote":
        raise HTTPException(
            401,
            detail={"error": "Authentication only required in remote mode"},
        )

    # Try session-based auth first (web UI)
    if session_id:
        session = (
            Session.select()
            .where(
                (Session.session_id == session_id)
                & (Session.expires_at > datetime.now(timezone.utc))
            )
            .first()
        )

        if session:
            user = session.user  # Use ForeignKey relationship
            if user and user.is_active:
                logger_api.debug(f"Authenticated via session: {user.username}")
                return user

    # Try token-based auth (API)
    if authorization and authorization.startswith("Bearer "):
        token_str = authorization[7:]  # Remove "Bearer " prefix
        token_hash_val = hash_token(token_str)

        token = Token.get_or_none(Token.token_hash == token_hash_val)
        if token:
            # Update last used
            Token.update(last_used=datetime.now(timezone.utc)).where(
                Token.id == token.id
            ).execute()

            user = token.user  # Use ForeignKey relationship
            if user and user.is_active:
                logger_api.debug(f"Authenticated via token: {user.username}")
                return user

    logger_api.debug("Authentication failed - no valid session or token")
    raise HTTPException(401, detail={"error": "Not authenticated"})


def get_optional_user(
    request: Request,
    session_id: str | None = Cookie(None),
    authorization: str | None = Header(None),
) -> User | None:
    """Get current user if authenticated, None otherwise.

    Does not require authentication - returns None if not authenticated.

    Args:
        request: FastAPI request object
        session_id: Session cookie value
        authorization: Authorization header value

    Returns:
        User object if authenticated, None otherwise
    """
    if cfg.app.mode != "remote":
        return None

    try:
        return get_current_user(request, session_id, authorization)
    except HTTPException:
        return None


def check_board_read_permission(board: Board, user: User | None):
    """Check if user can read board.

    Args:
        board: Board object
        user: User object or None

    Raises:
        HTTPException: 403 if access denied
    """
    # Public boards are accessible to everyone
    if not board.private:
        return

    # Private boards require authentication
    if user is None:
        raise HTTPException(403, detail={"error": "Board is private"})

    # Owner can always access
    if board.owner.id == user.id:
        return

    # Not owner and private
    raise HTTPException(403, detail={"error": "Access denied"})


def check_board_write_permission(board: Board, user: User):
    """Check if user can write to board.

    Args:
        board: Board object
        user: User object

    Raises:
        HTTPException: 403 if access denied
    """
    if board.owner.id != user.id:
        raise HTTPException(
            403,
            detail={"error": "Only board owner can modify board"},
        )


def create_session(user_id: int) -> tuple[str, str]:
    """Create a new session for user.

    Args:
        user_id: User ID

    Returns:
        Tuple of (session_id, secret)
    """
    session_id = generate_session_id()
    secret = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=cfg.app.session_expires_days
    )

    Session.create(
        session_id=session_id,
        user_id=user_id,
        secret=secret,
        expires_at=expires_at,
    )

    return session_id, secret


def create_token(user_id: int, name: str) -> str:
    """Create a new API token for user.

    Args:
        user_id: User ID
        name: Token name

    Returns:
        Plain token string (only returned once!)
    """
    token_str = generate_token()
    token_hash_val = hash_token(token_str)

    Token.create(
        user_id=user_id,
        token_hash=token_hash_val,
        name=name,
    )

    return token_str
