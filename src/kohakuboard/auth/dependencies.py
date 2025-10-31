"""FastAPI dependencies for authentication."""

from datetime import datetime, timezone

from fastapi import Cookie, Header, HTTPException

from kohakuboard.auth.utils import hash_token
from kohakuboard.config import cfg
from kohakuboard.db import Session, Token, User
from kohakuboard.logger import logger_api


def get_current_user(
    session_id: str | None = Cookie(None),
    authorization: str | None = Header(None),
) -> User:
    """Get current authenticated user from session or token.

    In LOCAL MODE: Always raises HTTPException (no auth in local mode)
    In REMOTE MODE: Checks session/token against database
    In REMOTE MODE with no_auth=True: Returns a test user without authentication

    Args:
        session_id: Session cookie
        authorization: Bearer token in Authorization header

    Returns:
        User: Authenticated user

    Raises:
        HTTPException: 401 if not authenticated or in local mode
    """
    # Local mode: no authentication, always raise
    if cfg.app.mode == "local":
        logger_api.debug("Local mode: authentication not available")
        raise HTTPException(401, detail="Authentication not available in local mode")

    # Remote mode with no_auth: return test user
    if cfg.app.no_auth:
        logger_api.debug("No-auth mode: using test user")
        # Get or create test user
        test_user = User.get_or_none(User.username == "test")
        if not test_user:
            from kohakuboard.auth.utils import hash_password
            from kohakuboard.utils.names import normalize_name

            test_user = User.create(
                username="test",
                normalized_name=normalize_name("test"),
                email="test@example.com",
                password_hash=hash_password("test"),
                is_active=True,
                email_verified=True,
            )
            logger_api.info("Created test user for no-auth mode")
        return test_user

    session_id = str(session_id) if session_id is not None else None
    authorization = str(authorization) if authorization is not None else None

    # Extract token from Authorization header
    auth_token = None
    if authorization and authorization.startswith("Bearer "):
        auth_token = authorization.replace("Bearer ", "").strip()

    # Log authentication attempt
    auth_method = []
    if session_id:
        auth_method.append(f"session={session_id[:8]}...")
    if auth_token:
        auth_method.append("token=Bearer ***")

    if auth_method:
        logger_api.debug(f"Auth attempt: {', '.join(auth_method)}")

    # Try session-based auth first (web UI)
    if session_id:
        session = Session.get_or_none(
            (Session.session_id == session_id)
            & (Session.expires_at > datetime.now(timezone.utc))
        )
        if session:
            user = session.user  # Use ForeignKey relationship
            if user and user.is_active:
                logger_api.debug(
                    f"Authenticated via session: {user.username} (session={session_id[:8]}...)"
                )
                return user
            else:
                logger_api.warning(
                    f"Session {session_id[:8]}... found but user inactive or not found"
                )
        else:
            logger_api.debug(f"Session {session_id[:8]}... not found or expired")

    # Try token-based auth (API)
    if auth_token:
        token_hash = hash_token(auth_token)

        token = Token.get_or_none(Token.token_hash == token_hash)
        if token:
            # Update last used
            Token.update(last_used=datetime.now(timezone.utc)).where(
                Token.id == token.id
            ).execute()

            user = token.user  # Use ForeignKey relationship
            if user and user.is_active:
                logger_api.debug(
                    f"Authenticated via token: {user.username} (token_id={token.id})"
                )
                return user
            else:
                logger_api.warning(
                    f"Token {token.id} found but user inactive or not found"
                )
        else:
            logger_api.debug("Token not found or invalid")

    logger_api.debug("Authentication failed - no valid session or token")
    raise HTTPException(401, detail="Not authenticated")


def get_optional_user(
    session_id: str | None = Cookie(None),
    authorization: str | None = Header(None),
) -> User | None:
    """Get current user if authenticated, otherwise None.

    In LOCAL MODE: Always returns None (no auth in local mode)
    In REMOTE MODE: Returns user if authenticated, None otherwise

    Args:
        session_id: Session cookie
        authorization: Bearer token in Authorization header

    Returns:
        User | None: Authenticated user or None
    """
    # Local mode: no authentication, return None immediately
    if cfg.app.mode == "local":
        return None

    try:
        return get_current_user(session_id, authorization)
    except HTTPException:
        return None
