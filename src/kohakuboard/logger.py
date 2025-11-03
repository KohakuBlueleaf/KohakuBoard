"""Loguru-based logging implementation for KohakuBoard (similar to KohakuHub)"""

import os
import sys
import logging
import traceback as tb
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

from kohakuboard.config import cfg


class LogLevel(Enum):
    """Log levels mapping to loguru levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRACE = "TRACE"


class InterceptHandler(logging.Handler):
    """Logger Interceptor: Redirects standard library logs to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get Level Name and API name from LogRecord
        try:
            level = logger.level(record.levelname).name
            api_name = record.name.upper()
            # For XXX.YYY format, only use XXX
            if "." in api_name:
                api_name = api_name.split(".")[0]
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.bind(api_name=api_name).opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


class Logger:
    """Loguru-based logger"""

    def __init__(self, api_name: str = "APP"):
        """Initialize logger with API name.

        Args:
            api_name: Name of the API/module (e.g., "AUTH", "API", "SYNC")
        """
        # For XXX.YYY format, only use XXX
        if "." in api_name:
            api_name = api_name.split(".")[0]

        self.api_name = api_name.upper()
        self._logger = logger.bind(api_name=self.api_name)

    def _log(self, level: LogLevel, message: str):
        """Internal log method."""
        match level:
            case LogLevel.DEBUG:
                self._logger.debug(message)
            case LogLevel.INFO:
                self._logger.info(message)
            case LogLevel.SUCCESS:
                self._logger.success(message)
            case LogLevel.WARNING:
                self._logger.warning(message)
            case LogLevel.ERROR:
                self._logger.error(message)
            case LogLevel.CRITICAL:
                self._logger.critical(message)
            case LogLevel.TRACE:
                self._logger.trace(message)
            case _:
                self._logger.log(level, message)

    def debug(self, message: str):
        self._log(LogLevel.DEBUG, message)

    def info(self, message: str):
        self._log(LogLevel.INFO, message)

    def success(self, message: str):
        self._log(LogLevel.SUCCESS, message)

    def warning(self, message: str):
        self._log(LogLevel.WARNING, message)

    def error(self, message: str):
        self._log(LogLevel.ERROR, message)

    def critical(self, message: str):
        self._log(LogLevel.CRITICAL, message)

    def trace(self, message: str):
        self._log(LogLevel.TRACE, message)

    def exception(self, message: str, exc: Optional[Exception] = None):
        """Log exception with formatted traceback.

        Args:
            message: Error message
            exc: Exception object (if None, uses sys.exc_info())
        """
        self.error(message)
        self._print_formatted_traceback(exc)

    def _print_formatted_traceback(self, exc: Optional[Exception] = None):
        """Print formatted traceback as tables.

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
        self.trace(f"{'=' * 50}")
        self.trace("TRACEBACK")
        self.trace(f"{'=' * 50}")

        # Print stack frames as tables
        for i, frame in enumerate(frames, 1):
            self._print_frame_table(i, frame, is_last=(i == len(frames)))

        # Print final error table
        self._print_error_table(exc_type, exc_value, frames[-1] if frames else None)

        self.trace(f"{'=' * 50}")

    def _print_frame_table(self, index: int, frame: tb.FrameSummary, is_last: bool):
        """Print single stack frame as a table.

        Args:
            index: Frame index
            frame: Frame summary
            is_last: Whether this is the last frame (error location)
        """
        self.trace(f"┌─ Frame #{index} {' (ERROR HERE)' if is_last else ''}")
        self.trace(f"│ File: {frame.filename}")
        self.trace(f"│ Line: {frame.lineno}")
        if frame.name:
            self.trace(f"│ In: {frame.name}()")
        if frame.line:
            self.trace(f"│ Code: {frame.line.strip()}")
        self.trace(f"└{'─' * 99}")

    def _print_error_table(
        self, exc_type, exc_value, last_frame: Optional[tb.FrameSummary]
    ):
        """Print final error details as a table.

        Args:
            exc_type: Exception type
            exc_value: Exception value
            last_frame: Last stack frame (error location)
        """
        self.trace(" EXCEPTION DETAILS ")
        self.trace(f"┌{'─' * 99}")
        self.trace(f"│ Type: {exc_type.__name__}")
        self.trace(f"│ Message: {str(exc_value)}")
        if last_frame:
            self.trace(f"│ Location: {last_frame.filename}:{last_frame.lineno}")
            if last_frame.line:
                self.trace(f"│ Code: {last_frame.line.strip()}")
        self.trace(f"└{'─' * 99}")


class LoggerFactory:
    """Factory to create loguru loggers."""

    _loggers = {}
    _file_only_names = set()  # Track api_names that should not go to stdout

    @classmethod
    def init_logger_settings(
        cls, log_file: Optional[Path] = None, file_only: bool = False
    ):
        """Initialize logger settings.

        Args:
            log_file: Optional log file path (if None, no file logging)
            file_only: If True, log ONLY to file, not stdout
        """
        # Remove default handler
        logger.remove()

        # Configure colors
        logger.level("DEBUG", color="<fg #666666>")
        logger.level("INFO", color="<fg #09D0EF>")
        logger.level("SUCCESS", color="<fg #66FF00>")
        logger.level("WARNING", color="<fg #FFEB2A>")
        logger.level("ERROR", color="<fg #FF160C>")
        logger.level("CRITICAL", color="<white><bg #FF160C><bold>")
        logger.level("TRACE", color="<fg #999999>")

        # Format: | time | name | level | message
        # Fixed widths: api_name=8, level=8
        log_format = (
            "<cyan>{time:HH:mm:ss.SSS}</cyan> | "
            "<fg #FF00CD>{extra[api_name]: <8}</fg #FF00CD> | "
            "<level>{level: <8}</level> | "
            "{message}"
        )

        # Add stdout logger (unless file_only)
        # Filter out api_names that are registered as file-only
        if not file_only:
            logger.add(
                sys.stderr,
                format=log_format,
                level="DEBUG",
                colorize=True,
                filter=lambda record: record["extra"].get("api_name")
                not in cls._file_only_names,
            )

        # Add file logger if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                log_file,
                format=log_format,
                level="DEBUG",
                rotation="10 MB",
                retention="7 days",
                colorize=False,  # No ANSI codes in files
            )

        # Intercept standard library logs
        logger_name_list = [name for name in logging.root.manager.loggerDict]
        for logger_name in logger_name_list:
            _logger = logging.getLogger(logger_name)
            _logger.setLevel(logging.INFO)
            _logger.handlers = []
            if "." not in logger_name:
                _logger.addHandler(InterceptHandler())
            else:
                _logger.propagate = True

    @classmethod
    def get_logger(cls, api_name: str) -> Logger:
        """Get or create logger for API name.

        Args:
            api_name: Name of the API/module

        Returns:
            Logger instance
        """
        if api_name not in cls._loggers:
            cls._loggers[api_name] = Logger(api_name)
        return cls._loggers[api_name]


def init_logger_settings(log_file: Optional[Path] = None, file_only: bool = False):
    """Initialize logger settings.

    Args:
        log_file: Optional log file path
        file_only: If True, log ONLY to file, not stdout
    """
    LoggerFactory.init_logger_settings(log_file, file_only)


def get_logger(
    api_name: str, file_only: bool = False, log_file: Optional[Path] = None
) -> Logger:
    """Get logger for specific API.

    Args:
        api_name: Name of the API/module
        file_only: If True, log ONLY to file (no stdout)
        log_file: Log file path (required if file_only=True)

    Returns:
        Logger instance
    """
    if file_only and log_file:
        # Create a separate logger instance with file-only configuration
        return create_file_only_logger(log_file, api_name)
    else:
        # Use shared logger configuration
        return LoggerFactory.get_logger(api_name)


def create_file_only_logger(log_file: Path, api_name: str = "WORKER") -> Logger:
    """Create a logger instance that writes ONLY to file.

    Uses bind() + filter to route messages to file only.
    Also adds a filter to BLOCK this api_name from stdout.

    Args:
        log_file: Path to log file
        api_name: API name for the logger

    Returns:
        Logger instance with file-only handler
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Format (same as main logger)
    log_format = (
        "<cyan>{time:HH:mm:ss.SSS}</cyan> | "
        "<fg #FF00CD>{extra[api_name]: <8}</fg #FF00CD> | "
        "<level>{level: <8}</level> | "
        "{message}"
    )

    # Register this api_name as file-only (exclude from stdout)
    LoggerFactory._file_only_names.add(api_name)

    # Add file handler filtered to ONLY this api_name
    logger.add(
        log_file,
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        colorize=False,  # No ANSI codes in log files
        filter=lambda record: record["extra"].get("api_name") == api_name,
    )

    # Create Logger wrapper
    file_logger = Logger(api_name)

    return file_logger


# Initialize default logger settings (stdout only for server)
init_logger_settings()

# Pre-create common loggers
logger_api = get_logger("API")
logger_mock = get_logger("MOCK")
