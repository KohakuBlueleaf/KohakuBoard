"""Boards API endpoints - serves real board data from file system"""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from kohakuboard.api.utils.board_reader import BoardReader, list_boards
from kohakuboard.config import cfg
from kohakuboard.logger import logger_api

router = APIRouter()


@router.get("/boards")
async def get_boards():
    """List all boards in the data directory

    Returns:
        List of boards with metadata
    """
    logger_api.info(f"Listing boards from: {cfg.app.board_data_dir}")

    try:
        boards = list_boards(Path(cfg.app.board_data_dir))
        logger_api.info(f"Found {len(boards)} boards")
        return boards
    except Exception as e:
        logger_api.error(f"Failed to list boards: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list boards: {str(e)}")


@router.get("/boards/{board_id}/metadata")
async def get_board_metadata(board_id: str):
    """Get board metadata

    Args:
        board_id: Board identifier

    Returns:
        Board metadata dict
    """
    logger_api.info(f"Fetching metadata for board: {board_id}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        return reader.get_metadata()
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger_api.error(f"Failed to get metadata for {board_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/summary")
async def get_board_summary(board_id: str):
    """Get board summary with metadata and available data

    Args:
        board_id: Board identifier

    Returns:
        Summary dict with metadata, counts, available metrics/media/tables
    """
    logger_api.info(f"Fetching summary for board: {board_id}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        return reader.get_summary()
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger_api.error(f"Failed to get summary for {board_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/scalars")
async def get_available_scalars(board_id: str):
    """Get list of available scalar metrics

    Args:
        board_id: Board identifier

    Returns:
        List of metric names
    """
    logger_api.info(f"Fetching available scalars for board: {board_id}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        metrics = reader.get_available_metrics()
        return {"metrics": metrics}
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger_api.error(f"Failed to get scalars for {board_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/scalars/{metric}")
async def get_scalar_data(
    board_id: str,
    metric: str,
    limit: Optional[int] = Query(None, description="Maximum number of data points"),
):
    """Get scalar data for a specific metric

    Args:
        board_id: Board identifier
        metric: Metric name
        limit: Optional row limit

    Returns:
        List of dicts with step, global_step, value
    """
    logger_api.info(f"Fetching scalar data for {board_id}/{metric}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        data = reader.get_scalar_data(metric, limit=limit)
        # data is now columnar format: {steps: [], global_steps: [], timestamps: [], values: []}
        return {"metric": metric, **data}
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger_api.error(f"Failed to get scalar data for {board_id}/{metric}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/media")
async def get_available_media(board_id: str):
    """Get list of available media log names

    Args:
        board_id: Board identifier

    Returns:
        List of media log names
    """
    logger_api.info(f"Fetching available media for board: {board_id}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        media_names = reader.get_available_media_names()
        return {"media": media_names}
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger_api.error(f"Failed to get media for {board_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/media/{name}")
async def get_media_data(
    board_id: str,
    name: str,
    limit: Optional[int] = Query(None, description="Maximum number of entries"),
):
    """Get media data for a specific log name

    Args:
        board_id: Board identifier
        name: Media log name
        limit: Optional row limit

    Returns:
        List of media entries with metadata
    """
    logger_api.info(f"Fetching media data for {board_id}/{name}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        data = reader.get_media_data(name, limit=limit)
        return {"name": name, "data": data}
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger_api.error(f"Failed to get media data for {board_id}/{name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/tables")
async def get_available_tables(board_id: str):
    """Get list of available table log names

    Args:
        board_id: Board identifier

    Returns:
        List of table log names
    """
    logger_api.info(f"Fetching available tables for board: {board_id}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        table_names = reader.get_available_table_names()
        return {"tables": table_names}
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger_api.error(f"Failed to get tables for {board_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/tables/{name}")
async def get_table_data(
    board_id: str,
    name: str,
    limit: Optional[int] = Query(None, description="Maximum number of entries"),
):
    """Get table data for a specific log name

    Args:
        board_id: Board identifier
        name: Table log name
        limit: Optional row limit

    Returns:
        List of table entries with parsed data
    """
    logger_api.info(f"Fetching table data for {board_id}/{name}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        data = reader.get_table_data(name, limit=limit)
        return {"name": name, "data": data}
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger_api.error(f"Failed to get table data for {board_id}/{name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/media/files/{filename}")
async def get_media_file(board_id: str, filename: str):
    """Serve media file (image/video/audio)

    Args:
        board_id: Board identifier
        filename: Media filename

    Returns:
        Media file response
    """
    logger_api.info(f"Serving media file: {board_id}/{filename}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id
        reader = BoardReader(board_dir)
        file_path = reader.get_media_file_path(filename)

        if not file_path:
            raise HTTPException(status_code=404, detail="Media file not found")

        # Determine media type from extension
        suffix = file_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
        }

        media_type = media_types.get(suffix, "application/octet-stream")

        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
        )
    except FileNotFoundError as e:
        logger_api.warning(f"Board not found: {board_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger_api.error(f"Failed to serve media file {board_id}/{filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/boards/{board_id}/media/by-id/{media_id}")
async def get_media_by_id(board_id: str, media_id: int):
    """Get media file by database ID

    Used to resolve <media id=123> tags from tables.
    Serves the actual media file content.

    Args:
        board_id: Board identifier
        media_id: Media database ID (SQLite auto-increment ID)

    Returns:
        Media file response (image/video/audio)
    """
    logger_api.info(f"Resolving media ID {media_id} for board {board_id}")

    try:
        board_dir = Path(cfg.app.board_data_dir) / board_id

        if not board_dir.exists():
            raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")

        # Initialize board reader
        reader = BoardReader(board_dir)

        # Query database for media ID
        media_info = reader.get_media_by_id(media_id)

        if not media_info:
            raise HTTPException(
                status_code=404,
                detail=f"Media ID {media_id} not found in board '{board_id}'",
            )

        # Get filename and file path
        filename = media_info["filename"]
        file_path = reader.get_media_file_path(filename)

        if not file_path or not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Media file not found: {filename}"
            )

        # Determine media type from format
        format_ext = media_info["format"]
        media_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "mp4": "video/mp4",
            "webm": "video/webm",
            "avi": "video/avi",
            "mov": "video/quicktime",
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }

        media_type = media_types.get(format_ext, "application/octet-stream")

        # Serve the actual file
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger_api.error(f"Error fetching media by ID {media_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
