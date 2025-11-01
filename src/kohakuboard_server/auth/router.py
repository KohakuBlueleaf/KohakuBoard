"""Authentication API routes for KohakuBoard."""

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr

from kohakuboard_server.auth.dependencies import get_current_user
from kohakuboard_server.auth.email import send_verification_email
from kohakuboard_server.auth.utils import (
    generate_session_secret,
    generate_token,
    get_expiry_time,
    hash_password,
    hash_token,
    verify_password,
)
from kohakuboard_server.config import cfg
from kohakuboard_server.db import Session, Token, User, db
from kohakuboard_server.db_operations import (
    create_email_verification,
    create_session,
    create_token,
    create_user,
    delete_email_verification,
    delete_token,
    get_email_verification,
    get_user_by_email,
    get_user_by_username,
    list_user_tokens,
    update_user,
)
from kohakuboard_server.logger import logger_api
from kohakuboard.utils.datetime_utils import safe_isoformat
from kohakuboard_server.utils.names import normalize_name

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateTokenRequest(BaseModel):
    name: str


@router.post("/register")
async def register(req: RegisterRequest):
    """Register new user.

    Args:
        req: Registration request data

    Returns:
        Success response
    """
    logger_api.info(f"Registration attempt for username: {req.username}")

    # Check if username or email already exists
    with db.atomic():
        if get_user_by_username(req.username):
            logger_api.warning(
                f"Registration failed: username '{req.username}' already exists"
            )
            raise HTTPException(400, detail="Username already exists")

        if get_user_by_email(req.email):
            logger_api.warning(
                f"Registration failed: email '{req.email}' already exists"
            )
            raise HTTPException(400, detail="Email already exists")

        # Check normalized name conflicts
        normalized = normalize_name(req.username)
        existing_normalized = User.get_or_none(User.normalized_name == normalized)
        if existing_normalized:
            logger_api.warning(
                f"Registration failed: username conflicts with '{existing_normalized.username}'"
            )
            entity_type = "organization" if existing_normalized.is_org else "user"
            raise HTTPException(
                400,
                detail=f"Username conflicts with existing {entity_type}: {existing_normalized.username}",
            )

        # Create user
        user = create_user(
            username=req.username,
            normalized_name=normalized,
            email=req.email,
            password_hash=hash_password(req.password),
            email_verified=not cfg.auth.require_email_verification,
        )

    logger_api.success(f"User registered: {user.username} (id={user.id})")

    # Send verification email if required
    if cfg.auth.require_email_verification:
        token = generate_token()
        create_email_verification(
            user=user, token=token, expires_at=get_expiry_time(24)
        )

        verification_email = await asyncio.to_thread(
            send_verification_email, req.email, req.username, token
        )
        if not verification_email:
            return {
                "success": True,
                "message": "User created but failed to send verification email",
                "email_verified": False,
            }

        return {
            "success": True,
            "message": "User created. Please check your email to verify your account.",
            "email_verified": False,
        }

    return {
        "success": True,
        "message": "User created successfully",
        "email_verified": True,
    }


@router.get("/verify-email")
async def verify_email(token: str, response: Response):
    """Verify email with token and automatically log in user."""
    logger_api.info(f"Email verification attempt with token: {token[:8]}...")

    verification = get_email_verification(token)

    if not verification:
        logger_api.warning(f"Invalid verification token: {token[:8]}...")
        return RedirectResponse(
            url=f"/?error=invalid_token&message=Invalid+or+expired+verification+token",
            status_code=302,
        )

    # Check expiry
    expires_at = verification.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at <= datetime.now(timezone.utc):
        logger_api.warning(f"Expired verification token: {token[:8]}...")
        return RedirectResponse(
            url=f"/?error=invalid_token&message=Invalid+or+expired+verification+token",
            status_code=302,
        )

    # Get user
    user = verification.user
    if not user:
        logger_api.error(f"User not found for verification token: {token[:8]}...")
        return RedirectResponse(url="/?error=user_not_found", status_code=302)

    # Update user and create session atomically
    with db.atomic():
        update_user(user, email_verified=True)
        delete_email_verification(verification)

        logger_api.success(f"Email verified for user: {user.username}")

        # Create session for auto-login
        session_id = generate_token()
        session_secret = generate_session_secret()

        create_session(
            session_id=session_id,
            user=user,
            secret=session_secret,
            expires_at=get_expiry_time(cfg.auth.session_expire_hours),
        )

    # Set session cookie
    redirect_response = RedirectResponse(url="/", status_code=302)
    redirect_response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=cfg.auth.session_expire_hours * 3600,
        samesite="lax",
    )

    logger_api.success(
        f"Auto-login session created for: {user.username} (session={session_id[:8]}...)"
    )

    return redirect_response


@router.post("/login")
async def login(req: LoginRequest, response: Response):
    """Login and create session."""

    logger_api.info(f"Login attempt for user: {req.username}")

    user = get_user_by_username(req.username)

    if not user or not verify_password(req.password, user.password_hash):
        logger_api.warning(f"Failed login attempt for: {req.username}")
        raise HTTPException(401, detail="Invalid username or password")

    if not user.is_active:
        logger_api.warning(f"Login attempt for disabled account: {req.username}")
        raise HTTPException(403, detail="Account is disabled")

    if cfg.auth.require_email_verification and not user.email_verified:
        logger_api.warning(f"Login attempt with unverified email: {req.username}")
        raise HTTPException(403, detail="Please verify your email first")

    # Create session
    session_id = generate_token()
    session_secret = generate_session_secret()

    create_session(
        session_id=session_id,
        user=user,
        secret=session_secret,
        expires_at=get_expiry_time(cfg.auth.session_expire_hours),
    )

    # Set cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=cfg.auth.session_expire_hours * 3600,
        samesite="lax",
    )

    logger_api.success(f"User logged in: {user.username} (session={session_id[:8]}...)")

    return {
        "success": True,
        "message": "Logged in successfully",
        "username": user.username,
        "session_secret": session_secret,
    }


@router.post("/logout")
async def logout(response: Response, user: User = Depends(get_current_user)):
    """Logout and destroy session."""

    logger_api.info(f"Logout request for user: {user.username}")

    # Delete all user sessions
    deleted_count = Session.delete().where(Session.user == user).execute()

    # Clear cookie
    response.delete_cookie(key="session_id")

    logger_api.success(
        f"User logged out: {user.username} ({deleted_count} session(s) deleted)"
    )

    return {"success": True, "message": "Logged out successfully"}


@router.get("/me")
def get_me(user: User = Depends(get_current_user)):
    """Get current user info."""

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "email_verified": user.email_verified,
        "created_at": safe_isoformat(user.created_at),
    }


@router.get("/tokens")
async def list_tokens(user: User = Depends(get_current_user)):
    """List user's API tokens."""

    tokens = list_user_tokens(user)

    return {
        "tokens": [
            {
                "id": t.id,
                "name": t.name,
                "last_used": safe_isoformat(t.last_used),
                "created_at": safe_isoformat(t.created_at),
            }
            for t in tokens
        ]
    }


@router.post("/tokens/create")
async def create_token_endpoint(
    req: CreateTokenRequest, user: User = Depends(get_current_user)
):
    """Create new API token."""

    # Generate token
    token_str = generate_token()
    token_hash_val = hash_token(token_str)

    # Save to database
    token = create_token(user=user, token_hash=token_hash_val, name=req.name)

    # Get session secret for encryption (if in web session)
    session = Session.get_or_none(Session.user == user)
    session_secret = session.secret if session else None

    logger_api.success(f"Token created for user {user.username}: {req.name}")

    return {
        "success": True,
        "token": token_str,
        "token_id": token.id,
        "session_secret": session_secret,
        "message": "Token created. Save it securely - you won't see it again!",
    }


@router.delete("/tokens/{token_id}")
async def revoke_token(token_id: int, user: User = Depends(get_current_user)):
    """Revoke an API token."""

    token = Token.get_or_none((Token.id == token_id) & (Token.user == user))

    if not token:
        raise HTTPException(404, detail="Token not found")

    delete_token(token.id)

    logger_api.success(f"Token revoked for user {user.username}: {token.name}")

    return {"success": True, "message": "Token revoked successfully"}
