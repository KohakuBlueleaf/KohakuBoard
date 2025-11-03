"""Utility functions for KohakuBoard Inspector"""

import json
from pathlib import Path


def is_board_directory(path: Path) -> bool:
    """Check if path is a single board directory

    A board directory must have metadata.json

    Args:
        path: Path to check

    Returns:
        True if path is a board directory
    """
    return (path / "metadata.json").exists()


def is_boards_container(path: Path) -> bool:
    """Check if path contains multiple boards

    Args:
        path: Path to check

    Returns:
        True if path contains board subdirectories
    """
    if not path.exists() or not path.is_dir():
        return False

    # If it's already a board, it's not a container
    if is_board_directory(path):
        return False

    # Check if any subdirectories are boards
    try:
        for item in path.iterdir():
            if item.is_dir() and is_board_directory(item):
                return True
    except PermissionError:
        return False

    return False


def detect_path_type(path: Path) -> str:
    """Detect what kind of path this is

    Args:
        path: Path to check

    Returns:
        'board' - Single board directory
        'container' - Multiple boards container
        'empty' - No boards found
        'invalid' - Path doesn't exist or not a directory
    """
    if not path.exists():
        return "invalid"

    if not path.is_dir():
        return "invalid"

    if is_board_directory(path):
        return "board"

    if is_boards_container(path):
        return "container"

    return "empty"


def list_boards_in_container(container_path: Path) -> list[dict]:
    """List all boards in a container directory

    Args:
        container_path: Path to container

    Returns:
        List of board info dicts with keys:
        - path: Path to board
        - name: Board name from metadata
        - board_id: Board ID
        - created_at: Creation timestamp
        - config: Board config dict
    """
    boards = []

    try:
        for item in container_path.iterdir():
            if not item.is_dir():
                continue

            if not is_board_directory(item):
                continue

            # Read metadata
            try:
                with open(item / "metadata.json", "r") as f:
                    metadata = json.load(f)

                boards.append(
                    {
                        "path": item,
                        "name": metadata.get("name", item.name),
                        "board_id": metadata.get("board_id", item.name),
                        "created_at": metadata.get("created_at", "Unknown"),
                        "config": metadata.get("config", {}),
                    }
                )

            except Exception as e:
                # Skip boards with invalid metadata
                print(f"Warning: Failed to read metadata for {item}: {e}")
                continue

    except PermissionError as e:
        print(f"Error: Permission denied reading {container_path}: {e}")

    # Sort by creation date (newest first)
    boards.sort(key=lambda b: b["created_at"], reverse=True)

    return boards


def get_board_size(board_path: Path) -> int:
    """Get total size of board directory in bytes

    Args:
        board_path: Path to board

    Returns:
        Total size in bytes
    """
    total = 0
    try:
        for item in board_path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except:
        pass

    return total


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.2 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
