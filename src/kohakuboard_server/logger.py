"""Logger for KohakuBoard Server

Server uses stdout logging only (no file loggers that conflict with client).
Uses native logging to avoid affecting downstream users.
"""

import logging
import sys


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_MAGENTA = "\033[95m"
    WHITE = "\033[37m"

    # Background colors
    BG_RED = "\033[41m"


# Custom log levels
SUCCESS_LEVEL = 25  # Between INFO(20) and WARNING(30)
TRACE_LEVEL = 5  # Below DEBUG(10)

logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
logging.addLevelName(TRACE_LEVEL, "TRACE")


# Level colors mapping
LEVEL_COLORS = {
    "TRACE": Colors.BRIGHT_BLACK,
    "DEBUG": Colors.BRIGHT_BLACK,
    "INFO": Colors.BRIGHT_CYAN,
    "SUCCESS": Colors.BRIGHT_GREEN,
    "WARNING": Colors.BRIGHT_YELLOW,
    "ERROR": Colors.BRIGHT_RED,
    "CRITICAL": f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}",
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""

    def format(self, record: logging.LogRecord) -> str:
        # Get API name from record
        api_name = getattr(record, "api_name", "SERVER")
        if len(api_name) > 8:
            api_name = api_name[:8]

        # Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")
        msecs = f"{record.msecs:03.0f}"

        # Get level name and color
        level_name = record.levelname
        level_color = LEVEL_COLORS.get(level_name, Colors.WHITE)

        # Colored format
        line = (
            f"{Colors.BRIGHT_CYAN}{timestamp}.{msecs}{Colors.RESET} | "
            f"{Colors.BRIGHT_MAGENTA}{api_name: <8}{Colors.RESET} | "
            f"{level_color}{level_name: <8}{Colors.RESET} | "
            f"{record.getMessage()}"
        )

        return line


class Logger:
    """Server logger wrapper"""

    def __init__(self, api_name: str):
        if "." in api_name:
            api_name = api_name.split(".")[0]
        self.api_name = api_name.upper()

        self._logger = logging.getLogger(f"kohakuboard_server.{self.api_name}")
        self._logger.setLevel(TRACE_LEVEL)
        self._logger.propagate = False

        # Add handler if not already added
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(ColoredFormatter())
            self._logger.addHandler(handler)

    def _log(self, level: int, message: str):
        self._logger.log(level, message, extra={"api_name": self.api_name})

    def trace(self, message: str):
        self._log(TRACE_LEVEL, message)

    def debug(self, message: str):
        self._log(logging.DEBUG, message)

    def info(self, message: str):
        self._log(logging.INFO, message)

    def success(self, message: str):
        self._log(SUCCESS_LEVEL, message)

    def warning(self, message: str):
        self._log(logging.WARNING, message)

    def error(self, message: str):
        self._log(logging.ERROR, message)

    def critical(self, message: str):
        self._log(logging.CRITICAL, message)


def get_logger(api_name: str) -> Logger:
    """Get logger for specific API component

    Args:
        api_name: API component name (e.g., "API", "AUTH", "DB")

    Returns:
        Logger instance
    """
    return Logger(api_name)


# Export common loggers
logger_api = get_logger("API")
logger_auth = get_logger("AUTH")
logger_db = get_logger("DB")
