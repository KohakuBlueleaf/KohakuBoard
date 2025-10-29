"""Media type definitions for logging"""

from pathlib import Path
from typing import Any, Union


class Media:
    """Media wrapper for images/videos

    Can be used directly with board.log() or inside tables.

    Examples:
        >>> # Direct logging
        >>> img = Media(image_array, caption="Sample")
        >>> board.log(sample_image=img)  # Equivalent to board.log_images("sample_image", image_array)

        >>> # In tables
        >>> table = Table([
        ...     {"name": "Cat", "image": Media(cat_img), "score": 0.95},
        ...     {"name": "Dog", "image": Media(dog_img), "score": 0.87},
        ... ])
        >>> board.log_table("results", table)
    """

    def __init__(
        self,
        data: Any,
        caption: str = "",
        media_type: str = "image",
    ):
        """Initialize Media

        Args:
            data: Image/video data (PIL Image, numpy array, torch Tensor, or file path)
            caption: Optional caption
            media_type: Type of media ("image" or "video")
        """
        self.data = data
        self.caption = caption
        self.media_type = media_type

    def __repr__(self) -> str:
        return f"Media(type={self.media_type}, caption={self.caption!r})"


def is_media(value: Any) -> bool:
    """Check if value is a Media object"""
    return isinstance(value, Media)
