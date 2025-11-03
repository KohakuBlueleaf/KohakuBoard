"""Data service - wraps HybridBoardReader for inspector"""

from pathlib import Path

from kohakuboard.utils.board_reader_hybrid import HybridBoardReader


class DataService:
    """Data access service for a single board

    Wraps HybridBoardReader with inspector-friendly API
    """

    def __init__(self, board_path: Path):
        """Initialize data service

        Args:
            board_path: Path to board directory
        """
        self.board_path = Path(board_path)
        self.reader = HybridBoardReader(board_path)

    def get_board_metadata(self) -> dict:
        """Get board metadata

        Returns:
            Metadata dict from metadata.json
        """
        return self.reader.get_metadata()

    def get_board_summary(self) -> dict:
        """Get board summary with counts

        Returns:
            Summary dict with metrics_count, media_count, etc.
        """
        return self.reader.get_summary()

    def get_available_metrics(self) -> list[str]:
        """Get list of available metrics

        Returns:
            List of metric names
        """
        return self.reader.get_available_metrics()

    def get_metric_data(self, metric: str, limit: int | None = None) -> dict[str, list]:
        """Get metric data

        Args:
            metric: Metric name
            limit: Optional row limit

        Returns:
            Dict with keys: steps, global_steps, timestamps, values
        """
        return self.reader.get_scalar_data(metric, limit)

    def get_available_media_names(self) -> list[str]:
        """Get list of available media log names

        Returns:
            List of media log names
        """
        return self.reader.get_available_media_names()

    def get_media_data(self, name: str, limit: int | None = None) -> list[dict]:
        """Get media log entries

        Args:
            name: Media log name
            limit: Optional limit

        Returns:
            List of media entry dicts
        """
        return self.reader.get_media_data(name, limit)

    def get_media_binary(self, filename: str) -> bytes | None:
        """Get media binary data

        Args:
            filename: Media filename

        Returns:
            Binary data or None
        """
        return self.reader.get_media_data(filename)

    def get_available_table_names(self) -> list[str]:
        """Get list of available table log names

        Returns:
            List of table log names
        """
        return self.reader.get_available_table_names()

    def get_table_data(self, name: str, limit: int | None = None) -> list[dict]:
        """Get table log entries

        Args:
            name: Table log name
            limit: Optional limit

        Returns:
            List of table entry dicts
        """
        return self.reader.get_table_data(name, limit)

    def get_available_histogram_names(self) -> list[str]:
        """Get list of available histogram names

        Returns:
            List of histogram names
        """
        return self.reader.get_available_histogram_names()

    def get_histogram_data(self, name: str, limit: int | None = None) -> list[dict]:
        """Get histogram data

        Args:
            name: Histogram name
            limit: Optional limit

        Returns:
            List of histogram entry dicts
        """
        return self.reader.get_histogram_data(name, limit)

    def close(self):
        """Cleanup resources"""
        # HybridBoardReader doesn't need explicit cleanup
        pass
