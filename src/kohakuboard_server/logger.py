"""Logger for KohakuBoard Server

Server uses stdout logging only (no file loggers that conflict with client).
"""

import sys

from loguru import logger


# Initialize server logger (stdout only)
logger.remove()  # Remove default handler

# Server log format
log_format = (
    "<cyan>{time:HH:mm:ss.SSS}</cyan> | "
    "<fg #FF00CD>{extra[api_name]: <8}</fg #FF00CD> | "
    "<level>{level: <8}</level> | "
    "{message}"
)

# Add stdout logger
logger.add(
    sys.stderr,
    format=log_format,
    level="DEBUG",
    colorize=True,
)


def get_logger(api_name: str):
    """Get logger for specific API component

    Args:
        api_name: API component name (e.g., "API", "AUTH", "DB")

    Returns:
        Logger instance
    """
    return logger.bind(api_name=api_name)


# Export common loggers
logger_api = get_logger("API")
logger_auth = get_logger("AUTH")
logger_db = get_logger("DB")
