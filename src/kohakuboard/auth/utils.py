"""Authentication utilities for KohakuBoard."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt


def hash_password(password: str) -> str:
    """Hash password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def generate_token() -> str:
    """Generate random token (32 bytes = 64 hex chars)."""
    return secrets.token_hex(32)


def hash_token(token: str) -> str:
    """Hash token with SHA3-512."""
    return hashlib.sha3_512(token.encode()).hexdigest()


def generate_session_secret() -> str:
    """Generate session secret for token encryption."""
    return secrets.token_hex(16)


def get_expiry_time(hours: int) -> datetime:
    """Get expiry time from now."""
    return datetime.now(timezone.utc) + timedelta(hours=hours)
