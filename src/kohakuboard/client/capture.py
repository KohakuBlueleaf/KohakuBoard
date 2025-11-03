"""Capture stdout and stderr to log file"""

import atexit
import sys
from pathlib import Path
from typing import Optional

from kohakuboard.logger import get_logger


logger = get_logger("CAPTURE")


class OutputCapture:
    """Capture stdout/stderr and redirect to file + terminal"""

    def __init__(self, log_file: Path):
        """Initialize output capture

        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.file_handle: Optional[object] = None
        self.active = False

    def start(self):
        """Start capturing output"""
        if self.active:
            logger.warning("Output capture already active")
            return

        # Open log file
        self.file_handle = open(self.log_file, "a", encoding="utf-8")

        # Create tee wrapper
        sys.stdout = TeeStream(self.original_stdout, self.file_handle)
        sys.stderr = TeeStream(
            self.original_stderr, self.file_handle, prefix="[STDERR] "
        )

        self.active = True
        logger.info(f"Started capturing stdout/stderr to {self.log_file}")

        # Register cleanup
        atexit.register(self.stop)

    def stop(self):
        """Stop capturing output"""
        if not self.active:
            return

        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Close file
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

        self.active = False
        logger.info("Stopped capturing stdout/stderr")


class TeeStream:
    """Stream that writes to multiple outputs (terminal + file)"""

    def __init__(self, stream1, stream2, prefix=""):
        """Initialize tee stream

        Args:
            stream1: First output stream (usually terminal)
            stream2: Second output stream (usually file)
            prefix: Optional prefix for each line
        """
        self.stream1 = stream1
        self.stream2 = stream2
        self.prefix = prefix
        self.current_line = ""  # Track current line for \r handling

    def write(self, data):
        """Write data to both streams"""
        # Write to terminal (with \r for tqdm)
        self.stream1.write(data)

        # For file: handle \r properly (tqdm progress bars)
        for char in data:
            if char == "\r":
                # Carriage return - discard current line, start fresh
                self.current_line = ""
            elif char == "\n":
                # Newline - write current line to file
                if self.current_line:
                    if self.prefix:
                        self.stream2.write(self.prefix + self.current_line + "\n")
                    else:
                        self.stream2.write(self.current_line + "\n")
                    self.current_line = ""
                else:
                    self.stream2.write("\n")
                # Flush immediately after each line for real-time sync
                self.stream2.flush()
            else:
                # Regular character - add to current line
                self.current_line += char

    def flush(self):
        """Flush both streams"""
        self.stream1.flush()
        self.stream2.flush()

    def isatty(self):
        """Check if stream is a TTY"""
        return self.stream1.isatty()
