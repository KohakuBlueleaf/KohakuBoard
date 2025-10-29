"""Logging configuration for KohakuBoard"""

import logging
import sys
import traceback as tb
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""

    COLORS = {
        "DEBUG": "\033[0;36m",  # Cyan
        "INFO": "\033[0;32m",  # Green
        "SUCCESS": "\033[0;92m",  # Bright Green
        "WARNING": "\033[0;33m",  # Yellow
        "ERROR": "\033[0;31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        record.name = f"\033[0;35m[{record.name}]{self.RESET}"
        return super().format(record)


class Logger:
    """Custom logger with success() and exception() methods"""

    def __init__(self, name: str):
        """Initialize logger with name.

        Args:
            name: Name of the logger (e.g., "API", "MOCK")
        """
        self.name = name
        self._logger = logging.getLogger(name)

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = ColoredFormatter("%(name)s %(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def debug(self, message: str):
        self._logger.debug(message)

    def info(self, message: str):
        self._logger.info(message)

    def success(self, message: str):
        """Log success message (custom level, logs as INFO with SUCCESS prefix)"""
        # Create a custom log record with SUCCESS level
        record = self._logger.makeRecord(
            self._logger.name,
            logging.INFO,
            "(unknown file)",
            0,
            message,
            (),
            None,
        )
        record.levelname = "SUCCESS"
        self._logger.handle(record)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str):
        self._logger.error(message)

    def critical(self, message: str):
        self._logger.critical(message)

    def exception(self, message: str, exc: Optional[Exception] = None):
        """Log exception with formatted traceback.

        Args:
            message: Error message
            exc: Exception object (if None, uses sys.exc_info())
        """
        self.error(message)
        self._print_formatted_traceback(exc)

    def _print_formatted_traceback(self, exc: Optional[Exception] = None):
        """Print formatted traceback.

        Args:
            exc: Exception object (if None, uses sys.exc_info())
        """
        if exc is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            exc_type = type(exc)
            exc_value = exc
            exc_tb = exc.__traceback__

        if exc_tb is None:
            return

        # Extract traceback frames
        frames = tb.extract_tb(exc_tb)

        # Print header
        self.debug(f"{'=' * 50}")
        self.debug("TRACEBACK")
        self.debug(f"{'=' * 50}")

        # Print stack frames
        for i, frame in enumerate(frames, 1):
            is_last = i == len(frames)
            self.debug(f"┌─ Frame #{i} {' (ERROR HERE)' if is_last else ''}")
            self.debug(f"│ File: {frame.filename}")
            self.debug(f"│ Line: {frame.lineno}")
            if frame.name:
                self.debug(f"│ In: {frame.name}()")
            if frame.line:
                self.debug(f"│ Code: {frame.line.strip()}")
            self.debug(f"└{'─' * 49}")

        # Print error details
        self.debug(" EXCEPTION DETAILS ")
        self.debug(f"┌{'─' * 49}")
        self.debug(f"│ Type: {exc_type.__name__}")
        self.debug(f"│ Message: {str(exc_value)}")
        if frames:
            last_frame = frames[-1]
            self.debug(f"│ Location: {last_frame.filename}:{last_frame.lineno}")
            if last_frame.line:
                self.debug(f"│ Code: {last_frame.line.strip()}")
        self.debug(f"└{'─' * 49}")


def get_logger(name: str) -> Logger:
    """Get a custom logger instance

    Args:
        name: Name of the logger

    Returns:
        Logger: Custom logger instance
    """
    return Logger(name)


# Pre-created loggers
logger_api = get_logger("API")
logger_mock = get_logger("MOCK")
