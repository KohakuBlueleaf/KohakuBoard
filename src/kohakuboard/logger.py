"""Native logging implementation for KohakuBoard with colored output

Uses standard library logging to avoid affecting downstream users.
Provides custom colored formatting and pretty traceback printing.
"""

import logging
import re
import sys
import traceback as tb
from pathlib import Path


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_WHITE = "\033[47m"


# Custom log levels (between standard levels)
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
    """Custom formatter with colors and comprehensive information"""

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Get API name from record (default to module name)
        api_name = getattr(record, "api_name", record.name.split(".")[0].upper())
        if len(api_name) > 8:
            api_name = api_name[:8]

        # Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")
        msecs = f"{record.msecs:03.0f}"

        # Get level name and color
        level_name = record.levelname
        level_color = LEVEL_COLORS.get(level_name, Colors.WHITE)

        if self.use_color:
            # Colored format: | time | api_name | level | message
            line = (
                f"{Colors.CYAN}{timestamp}.{msecs}{Colors.RESET} | "
                f"{Colors.BRIGHT_MAGENTA}{api_name: <8}{Colors.RESET} | "
                f"{level_color}{level_name: <8}{Colors.RESET} | "
                f"{record.getMessage()}"
            )
        else:
            # Plain format for file output
            line = (
                f"{timestamp}.{msecs} | "
                f"{api_name: <8} | "
                f"{level_name: <8} | "
                f"{record.getMessage()}"
            )

        return line


class Logger:
    """Custom logger wrapper with colored output and pretty tracebacks"""

    def __init__(self, api_name: str = "APP"):
        # Normalize api_name
        if "." in api_name:
            api_name = api_name.split(".")[0]
        self.api_name = api_name.upper()

        # Get or create the underlying logger
        # Use kohakuboard namespace to avoid conflicts
        self._logger = logging.getLogger(f"kohakuboard.{self.api_name}")
        self._logger.setLevel(TRACE_LEVEL)

        # Prevent propagation to root logger
        self._logger.propagate = False

        # Track file handlers added to this logger
        self._file_handlers: dict[int, logging.Handler] = {}
        self._handler_id_counter = 0

    def add_file_handler(
        self,
        log_file: Path,
        level: str = "DEBUG",
        rotation_bytes: int | None = None,
    ) -> int:
        """Add a file handler to this logger

        Args:
            log_file: Path to log file
            level: Minimum log level for this handler
            rotation_bytes: Max file size before rotation (not implemented in stdlib)

        Returns:
            Handler ID that can be used to remove it later
        """
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")

        # Parse level
        log_level = getattr(logging, level.upper(), logging.DEBUG)
        if level.upper() == "TRACE":
            log_level = TRACE_LEVEL
        elif level.upper() == "SUCCESS":
            log_level = SUCCESS_LEVEL

        file_handler.setLevel(log_level)
        file_handler.setFormatter(ColoredFormatter(use_color=False))

        self._logger.addHandler(file_handler)

        # Track handler for later removal
        handler_id = self._handler_id_counter
        self._handler_id_counter += 1
        self._file_handlers[handler_id] = file_handler

        return handler_id

    def remove_file_handler(self, handler_id: int) -> bool:
        """Remove a file handler by ID

        Args:
            handler_id: ID returned from add_file_handler

        Returns:
            True if handler was found and removed
        """
        handler = self._file_handlers.pop(handler_id, None)
        if handler:
            self._logger.removeHandler(handler)
            handler.close()
            return True
        return False

    def _log(self, level: int, message: str):
        """Internal log method with api_name injection"""
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

    def exception(self, message: str | Exception, exc: Exception | None = None):
        """Log exception with pretty formatted traceback

        Args:
            message: Error message or Exception object
            exc: Exception object (if message is string)
        """
        # Handle case where message is the exception
        if isinstance(message, Exception):
            exc = message
            message = str(exc)

        self.error(message)
        self._print_pretty_traceback(exc)

    def _print_pretty_traceback(self, exc: Exception | None = None):
        """Print formatted traceback with pretty box formatting

        Format:
        ==========================================
        File: /path/to/file.py
        Line: 42
        Method: some_function()
        Code: x = broken_code()
        ------------------------------------------
        ...more frames...
        ==========================================
        File: /path/to/error.py
        Line: 100
        Position:     ^^^^^^^^^^^
        Error: ValueError: something went wrong
        ==========================================
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
        if not frames:
            return

        # Build traceback string
        separator = "=" * 50
        thin_sep = "-" * 50

        lines = [separator, "TRACEBACK", separator]

        # Print each frame
        for i, frame in enumerate(frames):
            is_last = i == len(frames) - 1

            lines.append(f"File: {frame.filename}")
            lines.append(f"Line: {frame.lineno}")
            if frame.name:
                lines.append(f"Method: {frame.name}()")
            if frame.line:
                lines.append(f"Code: {frame.line.strip()}")

            if not is_last:
                lines.append(thin_sep)

        # Final error section
        lines.append(separator)
        last_frame = frames[-1]
        lines.append(f"File: {last_frame.filename}")
        lines.append(f"Line: {last_frame.lineno}")

        # Try to show position indicator if we have the code line
        if last_frame.line:
            code_line = last_frame.line
            stripped = code_line.strip()

            # Try to find error position from exception message
            # This is a heuristic - works for many common errors
            position_indicator = self._get_position_indicator(
                code_line, stripped, exc_value
            )
            if position_indicator:
                lines.append(f"Code: {stripped}")
                lines.append(f"Position: {position_indicator}")
            else:
                lines.append(f"Code: {stripped}")

        lines.append(f"Error: {exc_type.__name__}: {exc_value}")
        lines.append(separator)

        # Log each line as trace
        for line in lines:
            self.trace(line)

    def _get_position_indicator(
        self, original_line: str, stripped_line: str, exc_value: BaseException | None
    ) -> str | None:
        """Try to generate a position indicator (^^^) for the error location

        Returns indicator string or None if position cannot be determined
        """
        if exc_value is None:
            return None

        error_msg = str(exc_value)

        # Try to find a variable/attribute name mentioned in error
        # Common patterns: "'xxx' is not defined", "has no attribute 'xxx'"
        patterns = [
            r"name '(\w+)' is not defined",
            r"has no attribute '(\w+)'",
            r"'(\w+)' is undefined",
            r"cannot find '(\w+)'",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                name = match.group(1)
                # Find position in stripped line
                pos = stripped_line.find(name)
                if pos >= 0:
                    indicator = " " * pos + "^" * len(name)
                    return indicator

        return None


class _NullLogger:
    """No-op logger used to suppress output"""

    def __init__(self, api_name: str = "NULL"):
        self.api_name = api_name
        self._handler_id_counter = 0

    def add_file_handler(
        self,
        log_file: Path,
        level: str = "DEBUG",
        rotation_bytes: int | None = None,
    ) -> int:
        """No-op file handler addition"""
        handler_id = self._handler_id_counter
        self._handler_id_counter += 1
        return handler_id

    def remove_file_handler(self, handler_id: int) -> bool:
        """No-op file handler removal"""
        return False

    def trace(self, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def success(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def critical(self, message: str) -> None:
        pass

    def exception(self, message: str | Exception, exc: Exception | None = None) -> None:
        pass


class LoggerFactory:
    """Factory to create and manage loggers"""

    _loggers: dict[str, Logger | _NullLogger] = {}
    _handlers_initialized: bool = False
    _console_handler: logging.Handler | None = None
    _file_handler: logging.Handler | None = None
    _file_only_names: set[str] = set()
    _dropped_names: set[str] = set()

    @classmethod
    def init_logger_settings(
        cls, log_file: Path | None = None, file_only: bool = False, level: str = "DEBUG"
    ):
        """Initialize logger settings

        Args:
            log_file: Optional log file path
            file_only: If True, log ONLY to file, not stdout
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        """
        # Get the kohakuboard parent logger
        parent_logger = logging.getLogger("kohakuboard")
        parent_logger.setLevel(TRACE_LEVEL)
        parent_logger.propagate = False

        # Remove existing handlers
        parent_logger.handlers.clear()

        # Parse level
        log_level = getattr(logging, level.upper(), logging.DEBUG)
        if level.upper() == "TRACE":
            log_level = TRACE_LEVEL
        elif level.upper() == "SUCCESS":
            log_level = SUCCESS_LEVEL

        # Add console handler (unless file_only)
        if not file_only:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(ColoredFormatter(use_color=True))

            # Add filter to exclude file-only and dropped loggers
            console_handler.addFilter(cls._create_console_filter())

            parent_logger.addHandler(console_handler)
            cls._console_handler = console_handler

        # Add file handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(ColoredFormatter(use_color=False))
            parent_logger.addHandler(file_handler)
            cls._file_handler = file_handler

        cls._handlers_initialized = True

    @classmethod
    def _create_console_filter(cls):
        """Create filter for console output"""

        class ConsoleFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                api_name = getattr(record, "api_name", "")
                if api_name in cls._file_only_names:
                    return False
                if api_name in cls._dropped_names:
                    return False
                return True

        return ConsoleFilter()

    @classmethod
    def get_logger(cls, api_name: str, *, drop: bool = False) -> Logger | _NullLogger:
        """Get or create logger for API name

        Args:
            api_name: Name of the API/module
            drop: If True, create a logger that ignores all messages

        Returns:
            Logger instance
        """
        # Initialize handlers if not done
        if not cls._handlers_initialized:
            cls.init_logger_settings()

        # Normalize name
        if "." in api_name:
            api_name = api_name.split(".")[0]
        api_name = api_name.upper()

        if drop:
            cls._dropped_names.add(api_name)
            cls._loggers[api_name] = _NullLogger(api_name)
            return cls._loggers[api_name]

        # Check if previously dropped, recreate if needed
        if api_name in cls._loggers and isinstance(cls._loggers[api_name], _NullLogger):
            cls._loggers[api_name] = Logger(api_name)
            cls._dropped_names.discard(api_name)
            return cls._loggers[api_name]

        if api_name not in cls._loggers:
            cls._loggers[api_name] = Logger(api_name)
            cls._dropped_names.discard(api_name)

        return cls._loggers[api_name]


def init_logger_settings(
    log_file: Path | None = None, file_only: bool = False, level: str = "DEBUG"
):
    """Initialize logger settings

    Args:
        log_file: Optional log file path
        file_only: If True, log ONLY to file, not stdout
        level: Minimum log level
    """
    LoggerFactory.init_logger_settings(log_file, file_only, level)


def get_logger(
    api_name: str,
    file_only: bool = False,
    log_file: Path | None = None,
    drop: bool = False,
) -> Logger | _NullLogger:
    """Get logger for specific API

    Args:
        api_name: Name of the API/module
        file_only: If True, log ONLY to file (no stdout)
        log_file: Log file path (required if file_only=True)
        drop: If True, create a null logger that drops all messages

    Returns:
        Logger instance
    """
    if drop:
        return LoggerFactory.get_logger(api_name, drop=True)

    if file_only and log_file:
        return create_file_only_logger(log_file, api_name)
    else:
        return LoggerFactory.get_logger(api_name, drop=False)


def create_file_only_logger(log_file: Path, api_name: str = "WORKER") -> Logger:
    """Create a logger instance that writes ONLY to file

    Args:
        log_file: Path to log file
        api_name: API name for the logger

    Returns:
        Logger instance with file-only handler
    """
    # Normalize name
    if "." in api_name:
        api_name = api_name.split(".")[0]
    api_name = api_name.upper()

    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Register this api_name as file-only (exclude from stdout)
    LoggerFactory._file_only_names.add(api_name)

    # Create dedicated logger for this file
    file_logger_name = f"kohakuboard.{api_name}.file"
    underlying = logging.getLogger(file_logger_name)
    underlying.setLevel(TRACE_LEVEL)
    underlying.propagate = False

    # Remove existing handlers
    underlying.handlers.clear()

    # Add file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(TRACE_LEVEL)
    file_handler.setFormatter(ColoredFormatter(use_color=False))
    underlying.addHandler(file_handler)

    # Create Logger wrapper
    logger_instance = Logger(api_name)
    logger_instance._logger = underlying

    return logger_instance


# Initialize default logger settings (stdout only)
init_logger_settings()

# Pre-create common loggers
logger_api = get_logger("API")
logger_mock = get_logger("MOCK")
