"""Board reader factory for SQLite/Lance backends

v0.2.0+: Only SQLite and Hybrid (Lance+SQLite) backends are supported.
DuckDB and Parquet backends have been completely removed.

Auto-detects backend type and delegates to HybridBoardReader.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from kohakuboard.logger import get_logger
from kohakuboard.utils.board_reader_hybrid import HybridBoardReader

# Get logger for board reader
logger = get_logger("READER")


class BoardReader:
    """Read-only interface for accessing board data (factory pattern)

    v0.2.0+: Returns HybridBoardReader for all boards.
    Both 'sqlite' and 'hybrid' backends use the same reader since they share
    the same SQLite metadata structure.

    The HybridBoardReader handles:
    - Pure SQLite backends: Reads from metadata.db only
    - Hybrid backends: Reads metrics from Lance, metadata from SQLite
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

    Args:
        base_dir: Base directory containing boards

    Returns:
        List of dicts with board_id and metadata
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.warning(f"Board data directory does not exist: {base_dir}")
        return []

    boards = []
    for board_dir in base_dir.iterdir():
        if not board_dir.is_dir():
            continue

        # Check if it has metadata.json
        metadata_path = board_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Get updated_at from latest step in the database
            updated_at = None
            try:
                reader = BoardReader(board_dir)
                latest_step = reader.get_latest_step()
                if latest_step and latest_step.get("timestamp"):
                    updated_at = latest_step["timestamp"]
            except Exception as e:
                logger.debug(f"Failed to get latest step for {board_dir.name}: {e}")

            boards.append(
                {
                    "board_id": board_dir.name,
                    "name": metadata.get("name", board_dir.name),
                    "created_at": metadata.get("created_at"),
                    "updated_at": updated_at,
                    "config": metadata.get("config", {}),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to read metadata for {board_dir.name}: {e}")

    return sorted(boards, key=lambda x: x.get("created_at", ""), reverse=True)
