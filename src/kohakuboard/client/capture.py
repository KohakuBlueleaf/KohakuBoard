"""Capture stdout and stderr to log file"""

import atexit
import re
import sys
from pathlib import Path

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
        self.file_handle: object | None = None
        self.active = False

    def start(self):
        """Start capturing output"""
        if self.active:
            logger.warning("Output capture already active")
            return

        # Open log file in read/write binary mode for seeking
        if self.log_file.exists():
            self.file_handle = open(self.log_file, "r+b")
            self.file_handle.seek(0, 2)  # Seek to end
        else:
            self.file_handle = open(self.log_file, "w+b")

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


class MemoryOutputCapture:
    """Capture stdout/stderr and forward chunks to a sink callback (no files)."""

    def __init__(self, sink):
        """Initialize in-memory capture.

        Args:
            sink: Callable accepting (stream_name: str, text_chunk: str)
        """
        self.sink = sink
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.active = False

    def start(self):
        if self.active:
            logger.warning("Output capture already active")
            return

        sys.stdout = MemoryTeeStream(self.original_stdout, self.sink, "stdout")
        sys.stderr = MemoryTeeStream(self.original_stderr, self.sink, "stderr")

        self.active = True
        atexit.register(self.stop)

    def stop(self):
        if not self.active:
            return

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.active = False


class TeeStream:
    """Stream that writes to multiple outputs (terminal + file) with ANSI and \r handling"""

    # ANSI escape sequence pattern
    ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[([0-9;]*)([ABCDEFGHJKSTfmnsulh])")

    def __init__(self, stream1, stream2, prefix=""):
        """Initialize tee stream

        Args:
            stream1: First output stream (usually terminal)
            stream2: Second output stream (usually file, binary mode)
            prefix: Optional prefix for each line
        """
        self.stream1 = stream1
        self.stream2 = stream2
        self.prefix = prefix.encode("utf-8") if prefix else b""

        self.current_line_start = None  # Where current line starts in file
        self.previous_line_start = None  # Where previous line starts (for cursor up)
        self.at_line_start = True  # Track if we're at start of line (for prefix)

    def write(self, data):
        """Write data to both streams with \r and ANSI handling"""
        # Write to terminal unchanged (tqdm works naturally with ANSI)
        self.stream1.write(data)

        # Convert to string for processing
        if isinstance(data, bytes):
            data_str = data.decode("utf-8", errors="replace")
        else:
            data_str = data

        data_str = data_str.replace("\r\n", "\n")
        # Process character by character, handling ANSI escapes
        i = 0
        while i < len(data_str):
            # Check for ANSI escape sequence
            if (
                data_str[i] == "\x1b"
                and i + 1 < len(data_str)
                and data_str[i + 1] == "["
            ):
                # Parse ANSI escape
                match = self.ANSI_ESCAPE_PATTERN.match(data_str[i:])
                if match:
                    cmd = match.group(2)

                    # Handle cursor up (go to previous line)
                    if cmd == "A" and self.previous_line_start is not None:
                        # Cursor up - set position to previous line start
                        self.stream2.seek(self.previous_line_start)
                        self.current_line_start = self.previous_line_start
                        self.at_line_start = True

                    # Skip the ANSI sequence
                    i += len(match.group(0))
                    continue

            char = data_str[i]

            if char == "\r":
                # Carriage return - go back to start of current line
                if self.current_line_start is not None:
                    self.stream2.seek(self.current_line_start)
                    self.at_line_start = True
                # If no current_line_start, we haven't written anything yet

            elif char == "\n":
                # Newline - write it and advance
                self.stream2.write(b"\n")
                self.stream2.flush()

                # Remember where previous line started (for cursor-up)
                self.previous_line_start = self.current_line_start
                self.current_line_start = None
                self.at_line_start = True

            else:
                # Regular character - write it immediately
                if self.at_line_start:
                    # First character of line - record position and write prefix
                    if self.current_line_start is None:
                        self.current_line_start = self.stream2.tell()

                    # Write prefix (only at line start)
                    if self.prefix:
                        self.stream2.write(self.prefix)

                    self.at_line_start = False

                # Write the character
                self.stream2.write(char.encode("utf-8"))

            i += 1

    def flush(self):
        """Flush both streams"""
        self.stream1.flush()
        self.stream2.flush()

    def isatty(self):
        """Check if stream is a TTY"""
        return self.stream1.isatty()

    def __del__(self):
        """Flush incomplete lines on cleanup"""
        if hasattr(self, "stream2"):
            try:
                # Ensure final newline if needed
                if hasattr(self, "at_line_start") and not self.at_line_start:
                    self.stream2.write(b"\n")
                    self.stream2.flush()
            except:
                pass  # Ignore errors during cleanup


class MemoryTeeStream:
    """Simplified tee stream that forwards chunks to a sink callback."""

    def __init__(self, primary_stream, sink, stream_name: str):
        self.primary_stream = primary_stream
        self.sink = sink
        self.stream_name = stream_name

    def write(self, data):
        self.primary_stream.write(data)
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = str(data)
        if text:
            self.sink(self.stream_name, text)

    def flush(self):
        self.primary_stream.flush()

    def isatty(self):
        return self.primary_stream.isatty()
