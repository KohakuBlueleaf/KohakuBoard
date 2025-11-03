"""Sync client for uploading boards to remote server"""

import json
from pathlib import Path

import requests
from kohakuboard.logger import get_logger


logger = get_logger("CLIENT")


class SyncClient:
    """Client for syncing local boards to remote server

    Handles file uploads with multipart/form-data encoding.
    """

    def __init__(self, base_url: str, token: str):
        """Initialize sync client

        Args:
            base_url: Remote server base URL (e.g., https://board.example.com)
            token: Authentication token
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def sync_board(
        self,
        board_dir: Path,
        project: str,
        private: bool = True,
        progress_callback=None,
    ) -> dict:
        """Upload board to remote server

        Args:
            board_dir: Path to local board directory
            project: Project name on remote server
            private: Whether board should be private
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            dict: Response from server with run_id, url, status

        Raises:
            FileNotFoundError: If required files not found
            requests.HTTPError: If upload fails
        """
        board_dir = Path(board_dir)

        # Read metadata
        metadata_file = board_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {board_dir}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Update metadata with sync settings
        metadata["project"] = project
        metadata["private"] = private
        metadata["run_id"] = metadata.get("board_id") or board_dir.name

        logger.info(f"Syncing board: {metadata['run_id']}")
        logger.info(f"  Project: {project}")
        logger.info(f"  Name: {metadata.get('name', 'Unnamed')}")

        # Check DuckDB file exists
        duckdb_file = board_dir / "data" / "board.duckdb"
        if not duckdb_file.exists():
            raise FileNotFoundError(f"board.duckdb not found at {duckdb_file}")

        # Collect media files
        media_dir = board_dir / "media"
        media_files = []
        if media_dir.exists():
            media_files = list(media_dir.glob("*"))
            logger.info(f"  Found {len(media_files)} media files")

        # Prepare multipart upload
        files = {}
        data = {"metadata": json.dumps(metadata)}

        # Update progress
        if progress_callback:
            progress_callback(10)

        # Open DuckDB file
        duckdb_handle = open(duckdb_file, "rb")
        files["duckdb_file"] = (
            "board.duckdb",
            duckdb_handle,
            "application/octet-stream",
        )

        # Update progress
        if progress_callback:
            progress_callback(20)

        # Open media files
        media_handles = []
        for media_file in media_files:
            handle = open(media_file, "rb")
            media_handles.append(handle)
            files[f"media_files"] = (
                media_file.name,
                handle,
                "application/octet-stream",
            )

        # Update progress
        if progress_callback:
            progress_callback(30)

        try:
            # Upload
            url = f"{self.base_url}/api/projects/{project}/sync"
            logger.info(f"Uploading to: {url}")

            response = self.session.post(
                url,
                data=data,
                files=files,
                timeout=300,
            )

            # Update progress
            if progress_callback:
                progress_callback(90)

            response.raise_for_status()

            # Update progress
            if progress_callback:
                progress_callback(100)

            result = response.json()
            logger.info(f"Sync completed: {result['status']}")
            return result

        except requests.HTTPError as e:
            logger.error(f"Upload failed: {e}")
            logger.error(f"Response: {e.response.text}")
            raise

        finally:
            # Close all file handles
            duckdb_handle.close()
            for handle in media_handles:
                handle.close()
