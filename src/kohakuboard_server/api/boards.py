"""Board list API for server"""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException

from kohakuboard_server.config import cfg
from kohakuboard_server.logger import logger_api
from kohakuboard.utils.run_id import split_run_dir_name

router = APIRouter()


def list_boards(base_dir: Path):
    """List all boards in directory

    Args:
        base_dir: Base directory to search

    Returns:
        List of board dicts with metadata
    """
    boards = []
    if not base_dir.exists():
        return boards

    for board_dir in base_dir.iterdir():
        if not board_dir.is_dir():
            continue

        metadata_file = board_dir / "metadata.json"
        if not metadata_file.exists():
            continue

        try:
            import json

            with open(metadata_file) as f:
                metadata = json.load(f)

            stored_run_id = metadata.get("run_id")
            stored_annotation = metadata.get("annotation")
            parsed_run_id, parsed_annotation = split_run_dir_name(board_dir.name)

            boards.append(
                {
                    "board_id": board_dir.name,
                    "run_id": stored_run_id or parsed_run_id,
                    "annotation": stored_annotation or parsed_annotation,
                    "name": metadata.get("name", board_dir.name),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at"),
                    "config": metadata.get("config", {}),
                }
            )
        except Exception as e:
            logger_api.warning(f"Failed to read metadata for {board_dir.name}: {e}")
            continue

    return sorted(boards, key=lambda x: x.get("created_at") or "", reverse=True)


@router.get("/boards")
async def get_boards():
    """List all boards

    Returns:
        List of boards with metadata
    """
    base_dir = Path(cfg.app.board_data_dir)

    boards = await asyncio.to_thread(list_boards, base_dir)
    return boards
