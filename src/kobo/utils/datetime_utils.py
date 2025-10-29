"""Datetime utility functions for safe handling of database datetime fields."""

from datetime import datetime
from typing import Optional


def safe_isoformat(dt) -> str | None:
    """Safely convert datetime field to ISO format string.

    Handles both datetime objects and string timestamps from database.
    Peewee sometimes returns datetime fields as strings depending on the query.

    Args:
        dt: Either a datetime object, string timestamp, or None

    Returns:
        ISO format string or None
    """
    if dt is None:
        return None
    elif isinstance(dt, str):
        # Already a string, try to parse and re-format for consistency
        try:
            return datetime.fromisoformat(dt.replace("Z", "+00:00")).isoformat()
        except (ValueError, AttributeError):
            # If parsing fails, return as-is
            return dt
    elif isinstance(dt, datetime):
        return dt.isoformat()
    else:
        # Fallback: convert to string
        return str(dt)


def ensure_datetime(dt) -> Optional[datetime]:
    """Convert string or datetime to datetime object.

    Handles both datetime objects and string timestamps from database.
    Peewee sometimes returns datetime fields as strings depending on the query.

    Args:
        dt: Either a datetime object, string timestamp, or None

    Returns:
        datetime object or None

    Raises:
        ValueError: If string cannot be parsed as datetime
    """
    if dt is None:
        return None
    elif isinstance(dt, datetime):
        return dt
    elif isinstance(dt, str):
        # Try to parse string to datetime
        try:
            # Handle ISO format with 'Z' suffix
            return datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Cannot parse datetime string: {dt}") from e
    else:
        raise TypeError(f"Expected datetime or str, got {type(dt)}")


def safe_strftime(dt, fmt: str) -> Optional[str]:
    """Safely format datetime field using strftime.

    Handles both datetime objects and string timestamps from database.

    Args:
        dt: Either a datetime object, string timestamp, or None
        fmt: strftime format string

    Returns:
        Formatted datetime string or None
    """
    if dt is None:
        return None

    # Convert to datetime object first
    dt_obj = ensure_datetime(dt)
    return dt_obj.strftime(fmt)
