"""Board reader factory for hybrid KohakuVault + SQLite backends."""

import json
from pathlib import Path
from typing import Any, Dict, List

from kohakuboard.logger import get_logger
from kohakuboard.utils.board_reader_hybrid import HybridBoardReader
from kohakuboard.utils.run_id import split_run_dir_name

# Get logger for board reader
logger = get_logger("READER")

# Default project name when user doesn't specify one
DEFAULT_LOCAL_PROJECT = "default"


class BoardReader:
    """Read-only interface for accessing board data (factory pattern)

    v0.2.0+: Returns HybridBoardReader for all boards.
    Both legacy SQLite-only runs and modern hybrid runs share the same
    SQLite metadata structure, so a single reader works for all.

    The HybridBoardReader handles:
    - Pure SQLite metadata tables (steps, tables, tensors, etc.)
    - KohakuVault ColumnVault metric files + KVault media stores
    """

    def __new__(cls, board_dir: Path):
        """Factory method - returns HybridBoardReader for all boards

        Args:
            board_dir: Path to board directory

        Returns:
            HybridBoardReader instance

        Raises:
            FileNotFoundError: If board directory doesn't exist
        """
        board_dir = Path(board_dir)

        if not board_dir.exists():
            raise FileNotFoundError(f"Board directory not found: {board_dir}")

        # v0.2.0+: All boards use HybridBoardReader
        # Works for both 'sqlite' and 'hybrid' backends (same SQLite metadata structure)
        logger.debug(f"Loading board: {board_dir.name}")
        return HybridBoardReader(board_dir)


def list_boards(base_dir: Path) -> List[Dict[str, Any]]:
    """List all boards in base directory

    Supports both legacy layouts (boards directly inside base_dir) and
    project-based layouts (base_dir/<project>/<run_id>). Also understands
    remote storage paths (base_dir/users/<user>/<project>/<run_id>).

    Args:
        base_dir: Base directory containing boards

    Returns:
        List of dicts with board_id and metadata
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.warning(f"Board data directory does not exist: {base_dir}")
        return []

    boards: List[Dict[str, Any]] = []

    def add_board(board_dir: Path, project_name: str):
        metadata_path = board_dir / "metadata.json"
        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            updated_at = None
            try:
                reader = BoardReader(board_dir)
                latest_step = reader.get_latest_step()
                if latest_step and latest_step.get("timestamp"):
                    updated_at = latest_step["timestamp"]
            except Exception as e:
                logger.debug(f"Failed to get latest step for {board_dir.name}: {e}")

            parsed_run_id, parsed_annotation = split_run_dir_name(board_dir.name)
            annotation = metadata.get("annotation") or parsed_annotation

            boards.append(
                {
                    "board_id": board_dir.name,
                    "run_id": parsed_run_id,
                    "annotation": annotation,
                    "name": metadata.get("name", board_dir.name),
                    "created_at": metadata.get("created_at"),
                    "updated_at": updated_at,
                    "config": metadata.get("config", {}),
                    "project": project_name,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to read metadata for {board_dir.name}: {e}")

    def scan_project_dir(project_dir: Path, project_name: str):
        for run_dir in project_dir.iterdir():
            if run_dir.is_dir():
                add_board(run_dir, project_name)

    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue

        metadata_path = entry / "metadata.json"
        if metadata_path.exists():
            # Legacy layout: boards directly inside base_dir
            add_board(entry, DEFAULT_LOCAL_PROJECT)
            continue

        if entry.name == "users":
            # Remote-style layout: users/<user>/<project>/<run_id>
            for user_dir in entry.iterdir():
                if not user_dir.is_dir():
                    continue
                for project_dir in user_dir.iterdir():
                    if not project_dir.is_dir():
                        continue
                    scan_project_dir(project_dir, project_dir.name)
            continue

        scan_project_dir(entry, entry.name)

    return sorted(boards, key=lambda x: x.get("created_at", ""), reverse=True)
