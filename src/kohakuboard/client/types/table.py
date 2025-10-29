"""Table data structure for logging tabular data"""

from typing import Any, Dict, List, Optional, Union

from .media import Media, is_media


class Table:
    """Table data structure for logging structured tabular data

    Examples:
        >>> # From list of dicts
        >>> table = Table([
        ...     {"name": "Alice", "score": 95, "pass": True},
        ...     {"name": "Bob", "score": 87, "pass": True},
        ... ])

        >>> # With explicit columns and types
        >>> table = Table(
        ...     columns=["Sample", "Precision", "Recall"],
        ...     column_types=["text", "number", "number"],
        ...     rows=[
        ...         ["Cat", 0.85, 0.80],
        ...         ["Dog", 0.88, 0.85],
        ...     ]
        ... )
    """

    def __init__(
        self,
        data: Optional[Union[List[Dict[str, Any]], List[List[Any]]]] = None,
        columns: Optional[List[str]] = None,
        column_types: Optional[List[str]] = None,
        rows: Optional[List[List[Any]]] = None,
    ):
        """Initialize table

        Args:
            data: List of dicts (each dict is a row) or list of lists
            columns: Column names (inferred from data if not provided)
            column_types: Column types ('text', 'number', 'image', 'boolean')
            rows: Rows data (used with explicit columns)
        """
        if data is not None:
            if isinstance(data[0], dict):
                # List of dicts
                self.columns = list(data[0].keys())
                self.rows = [[row.get(col) for col in self.columns] for row in data]
            else:
                # List of lists
                self.columns = columns or [f"col_{i}" for i in range(len(data[0]))]
                self.rows = data
        elif columns and rows:
            self.columns = columns
            self.rows = rows
        else:
            raise ValueError("Must provide either data or (columns + rows)")

        # Infer column types if not provided
        if column_types is None:
            self.column_types = self._infer_types()
        else:
            self.column_types = column_types

    def _infer_types(self) -> List[str]:
        """Infer column types from data"""
        types = []
        for col_idx in range(len(self.columns)):
            # Sample first non-None value
            sample_val = None
            for row in self.rows:
                if row[col_idx] is not None:
                    sample_val = row[col_idx]
                    break

            if sample_val is None:
                types.append("text")
            elif is_media(sample_val):
                types.append("media")
            elif isinstance(sample_val, bool):
                types.append("boolean")
            elif isinstance(sample_val, (int, float)):
                types.append("number")
            else:
                types.append("text")

        return types

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization

        Returns:
            Dict with columns, column_types, rows, and media_objects
            Media objects in rows are replaced with placeholder strings
        """
        # Extract media objects and replace with placeholders
        media_objects = {}  # {row_idx: {col_idx: Media}}
        serialized_rows = []

        for row_idx, row in enumerate(self.rows):
            new_row = []
            for col_idx, value in enumerate(row):
                if is_media(value):
                    # Store media object for separate processing
                    if row_idx not in media_objects:
                        media_objects[row_idx] = {}
                    media_objects[row_idx][col_idx] = value
                    # Replace with placeholder
                    new_row.append(f"__media_{row_idx}_{col_idx}__")
                else:
                    new_row.append(value)
            serialized_rows.append(new_row)

        return {
            "columns": self.columns,
            "column_types": self.column_types,
            "rows": serialized_rows,
            "media_objects": media_objects,  # Separate media objects
        }

    def __repr__(self) -> str:
        return f"Table(columns={self.columns}, rows={len(self.rows)})"
