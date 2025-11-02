"""Media handling utilities for images, videos, and audio"""

import io
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from kohakuvault import KVault

from kohakuboard.client.utils.media_hash import generate_media_hash
from kohakuboard.logger import get_logger


class MediaHandler:
    """Handle media file storage and conversion

    Supports:
    - Images: PIL Image, numpy array, torch Tensor, file paths (png, jpg, gif, webp, etc.)
    - Videos: file paths (mp4, avi, mov, mkv, webm, etc.)
    - Audio: file paths (mp3, wav, flac, ogg, etc.)
    """

    # Supported extensions by type
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

    def __init__(self, media_dir: Path, kv_storage: KVault):
        """Initialize media handler

        Args:
            media_dir: Directory to store media files (deprecated, kept for logging path)
            kv_storage: KVault storage instance for media binary data
        """
        self.media_dir = media_dir
        self.kv_storage = kv_storage

        # Setup file-only logger for media
        log_file = media_dir.parent / "logs" / "media.log"
        self.logger = get_logger("MEDIA", file_only=True, log_file=log_file)

    def process_media(
        self, media: Any, name: str, step: int, media_type: str = "image"
    ) -> dict:
        """Process and save media with content-addressable storage

        New behavior (v0.2.0+):
        - Filename: {hash}.{ext} (no step, no name)
        - Deduplication: Check if file exists before writing
        - Returns metadata dict with hash and filename (media_id assigned by DB)

        Args:
            media: Media data (PIL Image, numpy array, torch Tensor, or file path)
            name: Name for this media log (for metadata only, not in filename)
            step: Step number (for metadata only, not in filename)
            media_type: Type of media ("image", "video", "audio", "auto")

        Returns:
            dict with media metadata:
                - media_hash: 22-char base-36 hash
                - format: file extension (e.g., "png", "mp4", "wav")
                - type: "image", "video", or "audio"
                - size_bytes: file size
                - width, height: image dimensions (if applicable)
                - deduplicated: True if file already existed
            Note: Filename is derived as {media_hash}.{format}, not returned here
        """
        # Auto-detect type from file extension if type is "auto"
        if media_type == "auto" and isinstance(media, (str, Path)):
            ext = Path(media).suffix.lower()
            if ext in self.IMAGE_EXTS:
                media_type = "image"
            elif ext in self.VIDEO_EXTS:
                media_type = "video"
            elif ext in self.AUDIO_EXTS:
                media_type = "audio"

        # Convert media to bytes and determine format
        if media_type == "image":
            content_bytes, ext = self._prepare_image(media)
        elif media_type == "video":
            content_bytes, ext = self._prepare_video(media)
        elif media_type == "audio":
            content_bytes, ext = self._prepare_audio(media)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")

        # Generate content hash for key
        media_hash = generate_media_hash(content_bytes)
        key = f"{media_hash}.{ext}"

        # Deduplication: Only write if key doesn't exist in KVault
        already_exists = key in self.kv_storage
        if not already_exists:
            # Use cache for efficient bulk writes (cache is shared across calls)
            with self.kv_storage.cache(64 * 1024 * 1024):  # 64MB cache
                self.kv_storage[key] = content_bytes
            self.logger.debug(f"Saved new {media_type} to KVault: {key}")
        else:
            self.logger.debug(
                f"Deduplicated {media_type}: {key} (already exists in KVault)"
            )

        # Get file metadata
        file_size = len(content_bytes)

        # Get dimensions for images
        width, height = None, None
        if media_type == "image":
            try:
                from PIL import Image

                with Image.open(io.BytesIO(content_bytes)) as img:
                    width, height = img.size
            except Exception:
                pass  # Skip if we can't read dimensions

        # Return metadata (media_id will be assigned by database)
        # Key is stored in SQLite KV as {media_hash}.{format}
        return {
            "media_hash": media_hash,
            "format": ext,
            "type": media_type,
            "size_bytes": file_size,
            "width": width,
            "height": height,
            "deduplicated": already_exists,
        }

    def process_images(self, images: list[Any], name: str, step: int) -> list[dict]:
        """Process multiple images

        Args:
            images: List of images
            name: Name for this media log
            step: Step number

        Returns:
            List of media metadata dicts
        """
        results = []
        for idx, img in enumerate(images):
            metadata = self.process_media(
                img, f"{name}_{idx}", step, media_type="image"
            )
            results.append(metadata)
        return results

    def _prepare_image(self, image: Any) -> tuple[bytes, str]:
        """Prepare image data for storage

        Args:
            image: PIL Image, numpy array, torch Tensor, or file path

        Returns:
            tuple of (content_bytes, extension)
        """
        try:
            # If it's a file path, read the file directly
            if isinstance(image, (str, Path)):
                source_path = Path(image)
                if not source_path.exists():
                    raise FileNotFoundError(f"Image file not found: {source_path}")

                with open(source_path, "rb") as f:
                    content_bytes = f.read()

                ext = source_path.suffix.lstrip(".").lower()
                if not ext:
                    ext = "png"

                return content_bytes, ext

            # Otherwise convert to PIL and save as PNG
            pil_image = self._to_pil(image)

            # Convert to bytes
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG", optimize=True)
            content_bytes = buf.getvalue()

            return content_bytes, "png"

        except Exception as e:
            self.logger.error(f"Failed to prepare image: {e}")
            raise

    def _prepare_video(self, video: str | Path) -> tuple[bytes, str]:
        """Prepare video data for storage

        Args:
            video: Video file path

        Returns:
            tuple of (content_bytes, extension)
        """
        source_path = Path(video)

        if not source_path.exists():
            raise FileNotFoundError(f"Video file not found: {video}")

        with open(source_path, "rb") as f:
            content_bytes = f.read()

        ext = source_path.suffix.lstrip(".").lower()
        if not ext:
            ext = "mp4"

        return content_bytes, ext

    def _prepare_audio(self, audio: str | Path) -> tuple[bytes, str]:
        """Prepare audio data for storage

        Args:
            audio: Audio file path

        Returns:
            tuple of (content_bytes, extension)
        """
        source_path = Path(audio)

        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        with open(source_path, "rb") as f:
            content_bytes = f.read()

        ext = source_path.suffix.lstrip(".").lower()
        if not ext:
            ext = "wav"

        return content_bytes, ext

    def _to_pil(self, image: Any):
        """Convert various image formats to PIL Image"""
        try:
            from PIL import Image

            if isinstance(image, Image.Image):
                return image

            # File path
            if isinstance(image, (str, Path)):
                return Image.open(image)

            # Numpy array
            if hasattr(image, "__array__"):

                arr = np.array(image)

                # Normalize to 0-255 uint8
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)

                # Handle channel dimensions (C, H, W) -> (H, W, C)
                if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                    arr = np.transpose(arr, (1, 2, 0))

                # Remove single channel dimension
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr[:, :, 0]

                return Image.fromarray(arr)

            # Torch tensor
            if hasattr(image, "cpu"):
                tensor = image.detach().cpu().numpy()
                return self._to_pil(tensor)

        except ImportError:
            pass

        raise ValueError(f"Unsupported image type: {type(image)}")
