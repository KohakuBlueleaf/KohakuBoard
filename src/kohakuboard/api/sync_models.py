"""Pydantic models for incremental sync API

Defines request/response schemas for the new sync endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SyncRange(BaseModel):
    """Step range being synced"""

    start_step: int = Field(..., description="Start step (inclusive)")
    end_step: int = Field(..., description="End step (inclusive)")


class StepData(BaseModel):
    """Step metadata"""

    step: int = Field(..., description="Step number")
    global_step: Optional[int] = Field(None, description="Global step number")
    timestamp: Optional[int] = Field(None, description="Timestamp in milliseconds")


class ScalarPoint(BaseModel):
    """Single scalar data point"""

    step: int = Field(..., description="Step number")
    value: float = Field(..., description="Metric value")


class MediaMetadata(BaseModel):
    """Media metadata from SQLite"""

    id: int = Field(..., description="Media ID from SQLite")
    media_hash: str = Field(..., description="22-char content hash")
    format: str = Field(..., description="File format (png, jpg, mp4, etc.)")
    step: int = Field(..., description="Step number")
    global_step: Optional[int] = Field(None, description="Global step number")
    name: str = Field(..., description="Media log name")
    caption: Optional[str] = Field(None, description="Caption")
    type: str = Field(..., description="Media type (image, video, audio)")
    width: Optional[int] = Field(None, description="Image width (if applicable)")
    height: Optional[int] = Field(None, description="Image height (if applicable)")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")


class TableData(BaseModel):
    """Table data"""

    step: int = Field(..., description="Step number")
    global_step: Optional[int] = Field(None, description="Global step number")
    name: str = Field(..., description="Table name")
    columns: List[str] = Field(..., description="Column names")
    column_types: List[str] = Field(..., description="Column types")
    rows: List[List[Any]] = Field(..., description="Table rows")


class HistogramData(BaseModel):
    """Histogram data"""

    step: int = Field(..., description="Step number")
    global_step: Optional[int] = Field(None, description="Global step number")
    name: str = Field(..., description="Histogram name")
    bins: List[float] = Field(..., description="Bin edges (K+1 values)")
    counts: List[int] = Field(..., description="Bin counts (K values)")
    precision: str = Field(..., description="Precision (i32 or u8)")


class LogSyncRequest(BaseModel):
    """Request payload for incremental log sync

    Contains all new data since last sync, organized by type.
    """

    sync_range: SyncRange = Field(..., description="Range of steps being synced")
    steps: List[StepData] = Field(
        default_factory=list, description="Step metadata entries"
    )
    scalars: Dict[str, List[ScalarPoint]] = Field(
        default_factory=dict,
        description="Scalar metrics (metric_name -> list of points)",
    )
    media: List[MediaMetadata] = Field(
        default_factory=list, description="Media metadata entries"
    )
    tables: List[TableData] = Field(default_factory=list, description="Table entries")
    histograms: List[HistogramData] = Field(
        default_factory=list, description="Histogram entries"
    )


class LogSyncResponse(BaseModel):
    """Response from log sync endpoint"""

    status: str = Field(..., description="Sync status (synced, partial, error)")
    synced_range: SyncRange = Field(..., description="Range that was synced")
    missing_media: List[str] = Field(
        default_factory=list,
        description="List of media hashes that need to be uploaded",
    )
    errors: Optional[List[str]] = Field(
        None, description="List of errors (if status is partial or error)"
    )


class MediaUploadResponse(BaseModel):
    """Response from media upload endpoint"""

    status: str = Field(..., description="Upload status")
    uploaded_count: int = Field(..., description="Number of files uploaded")
    uploaded_hashes: List[str] = Field(
        default_factory=list, description="List of uploaded media hashes"
    )
    skipped_count: int = Field(
        default=0, description="Number of files skipped (already exist)"
    )
