"""Media handling utilities for images, videos, and audio"""

import hashlib
import io
import shutil
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from loguru import logger


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

    def __init__(self, media_dir: Path):
        """Initialize media handler

        Args:
            media_dir: Directory to store media files
        """
        self.media_dir = media_dir
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def process_media(
        self, media: Any, name: str, step: int, media_type: str = "image"
    ) -> dict:
        """Process and save media (image/video/audio)

        Args:
            media: Media data (PIL Image, numpy array, torch Tensor, or file path)
            name: Name for this media log
            step: Step number
            media_type: Type of media ("image", "video", "audio", "auto")

        Returns:
            dict with media metadata
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

        if media_type == "image":
            return self._process_image(media, name, step)
        elif media_type == "video":
            return self._process_video(media, name, step)
        elif media_type == "audio":
            return self._process_audio(media, name, step)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")

    def process_images(self, images: List[Any], name: str, step: int) -> List[dict]:
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
            metadata = self._process_image(img, f"{name}_{idx}", step)
            results.append(metadata)
        return results

    def _process_image(self, image: Any, name: str, step: int) -> dict:
        """Process and save image

        Supports: PIL Image, numpy array, torch Tensor, file path (any image format)
        """
        try:
            # If it's a file path, copy it directly (preserves GIF animation, etc.)
            if isinstance(image, (str, Path)):
                return self._copy_file(image, name, step, "image")

            # Otherwise convert to PIL and save as PNG
            pil_image = self._to_pil(image)

            # Generate filename and hash
            # Replace "/" with "__" in name to avoid subdirectory issues
            safe_name = name.replace("/", "__")
            image_hash = self._hash_media(pil_image)
            ext = "png"
            filename = f"{safe_name}_{step:08d}_{image_hash[:8]}.{ext}"
            filepath = self.media_dir / filename

            # Save image
            pil_image.save(filepath, format="PNG", optimize=True)

            logger.debug(f"Saved image: {filepath}")

            return {
                "media_id": image_hash,
                "type": "image",
                "filename": filename,
                "path": str(filepath),
                "width": pil_image.width,
                "height": pil_image.height,
            }

        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise

    def _process_video(self, video: Union[str, Path], name: str, step: int) -> dict:
        """Process and save video

        Only supports file paths (videos must already be encoded)
        """
        return self._copy_file(video, name, step, "video")

    def _process_audio(self, audio: Union[str, Path], name: str, step: int) -> dict:
        """Process and save audio

        Only supports file paths (audio must already be encoded)
        """
        return self._copy_file(audio, name, step, "audio")

    def _copy_file(
        self, source: Union[str, Path], name: str, step: int, media_type: str
    ) -> dict:
        """Copy media file to storage

        Args:
            source: Source file path
            name: Name for this media log
            step: Step number
            media_type: Type of media

        Returns:
            dict with media metadata
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Media file not found: {source}")

        # Calculate hash for deduplication
        media_hash = self._hash_file(source_path)

        # Preserve original extension
        # Replace "/" with "__" in name to avoid subdirectory issues
        safe_name = name.replace("/", "__")
        ext = source_path.suffix.lstrip(".")
        filename = f"{safe_name}_{step:08d}_{media_hash[:8]}.{ext}"
        dest_path = self.media_dir / filename

        # Copy file if it doesn't exist (deduplication)
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            logger.debug(f"Copied {media_type}: {dest_path}")
        else:
            logger.debug(f"Deduplicated {media_type}: {dest_path}")

        # Get file size and metadata
        file_size = dest_path.stat().st_size

        metadata = {
            "media_id": media_hash,
            "type": media_type,
            "filename": filename,
            "path": str(dest_path),
            "size_bytes": file_size,
            "format": ext,
        }

        # Add dimensions for images (if we can read them)
        if media_type == "image":
            try:
                from PIL import Image

                with Image.open(dest_path) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
            except Exception:
                pass  # Skip if we can't read dimensions

        return metadata

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

    def _hash_media(self, pil_image) -> str:
        """Generate hash for image deduplication (also used as media ID)"""
        # Convert to bytes
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Generate hash (this becomes the media_id)
        return hashlib.sha256(image_bytes).hexdigest()

    def _hash_file(self, filepath: Path) -> str:
        """Generate hash for file deduplication (also used as media ID)"""
        sha256 = hashlib.sha256()

        with open(filepath, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()
