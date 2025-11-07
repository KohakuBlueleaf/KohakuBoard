"""Utility functions for KohakuBoard Inspector"""

import json
from pathlib import Path

from kohakuboard.utils.board_reader import DEFAULT_LOCAL_PROJECT


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
    """Check if path contains multiple boards (supports nested projects)."""
    if not path.exists() or not path.is_dir():
        return False

    if is_board_directory(path):
        return False

    try:
        for item in path.iterdir():
            if not item.is_dir():
                continue

            if is_board_directory(item):
                return True

            # Project-style directory
            try:
                for sub in item.iterdir():
                    if sub.is_dir() and is_board_directory(sub):
                        return True
            except PermissionError:
                continue
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
    """List all boards in a container directory (supports project folders)."""

    def read_board(board_path: Path, project: str):
        try:
            with open(board_path / "metadata.json", "r") as f:
                metadata = json.load(f)

            boards.append(
                {
                    "path": board_path,
                    "name": metadata.get("name", ""),
                    "board_id": metadata.get("board_id", board_path.name),
                    "created_at": metadata.get("created_at", "Unknown"),
                    "config": metadata.get("config", {}),
                    "project": project,
                }
            )
        except Exception as e:
            print(f"Warning: Failed to read metadata for {board_path}: {e}")

    boards: list[dict] = []

    try:
        for item in container_path.iterdir():
            if not item.is_dir():
                continue

            if is_board_directory(item):
                read_board(item, DEFAULT_LOCAL_PROJECT)
                continue

            if item.name == "users":
                try:
                    for user_dir in item.iterdir():
                        if not user_dir.is_dir():
                            continue
                        for project_dir in user_dir.iterdir():
                            if not project_dir.is_dir():
                                continue
                            for run_dir in project_dir.iterdir():
                                if run_dir.is_dir() and is_board_directory(run_dir):
                                    read_board(run_dir, project_dir.name)
                except PermissionError:
                    continue
                continue

            try:
                for run_dir in item.iterdir():
                    if run_dir.is_dir() and is_board_directory(run_dir):
                        read_board(run_dir, item.name)
            except PermissionError:
                continue

    except PermissionError as e:
        print(f"Error: Permission denied reading {container_path}: {e}")

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
