"""
Pydantic schemas for API request/response models
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


# === Common Types ===
from enum import Enum

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


# === Process API ===
class ProcessRequest(BaseModel):
    """Request for creating a processing task"""
    s3_key: Optional[str] = Field(None, description="S3 key for uploaded video")
    filename: Optional[str] = Field(None, description="Original filename")
    size: Optional[int] = Field(None, description="File size in bytes")


class ProcessResponse(BaseModel):
    """Response for task creation"""
    task_id: str = Field(..., description="Unique task identifier")


# === Result API ===
class ResultResponse(BaseModel):
    """Response for task status"""
    status: TaskStatus = Field(..., description="Current task status")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    message: Optional[str] = Field(None, description="Status message or error details")


# === Payload API ===
class VideoMetadata(BaseModel):
    """Video metadata"""
    s3_key: str = Field(..., description="S3 path to video file")
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    duration: float = Field(..., description="Video duration in seconds")
    fps: Optional[float] = Field(None, description="Frames per second")


class TaskMetadata(BaseModel):
    """Task metadata"""
    task_id: str = Field(..., description="Task identifier")
    video: VideoMetadata = Field(..., description="Video information")
    generated_at: str = Field(..., description="ISO timestamp")
    model_versions: Optional[Dict[str, str]] = Field(None, description="Model versions used")


class NormalizedBox(BaseModel):
    """Normalized bounding box coordinates"""
    x1: float = Field(..., ge=0, le=1, description="Left coordinate (0-1)")
    y1: float = Field(..., ge=0, le=1, description="Top coordinate (0-1)")
    x2: float = Field(..., ge=0, le=1, description="Right coordinate (0-1)")
    y2: float = Field(..., ge=0, le=1, description="Bottom coordinate (0-1)")


class PIIDetection(BaseModel):
    """PII detection result"""
    id: str = Field(..., description="Unique detection ID")
    label: str = Field(..., description="PII type (NAME, EMAIL, PHONE, etc.)")
    text: Optional[str] = Field(None, description="Detected text content")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Detection confidence")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    norm_box: NormalizedBox = Field(..., description="Normalized bounding box")
    source: Optional[str] = Field(None, description="Detection source (ocr, asr, ner, face, fusion)")


class FaceBox(BaseModel):
    """Face detection box"""
    id: Optional[str] = Field(None, description="Face ID")
    x1: float = Field(..., ge=0, le=1, description="Left coordinate (0-1)")
    y1: float = Field(..., ge=0, le=1, description="Top coordinate (0-1)")
    x2: float = Field(..., ge=0, le=1, description="Right coordinate (0-1)")
    y2: float = Field(..., ge=0, le=1, description="Bottom coordinate (0-1)")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Detection confidence")


class FaceFrame(BaseModel):
    """Face detection frame"""
    id: Optional[str] = Field(None, description="Frame identifier")
    t: float = Field(..., description="Timestamp in seconds")
    end: Optional[float] = Field(None, description="End timestamp in seconds")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Frame-level confidence")
    frame_range: Optional[str] = Field(None, description="Frame index range (e.g., '0-749')")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata from detector")
    norm_boxes: List[FaceBox] = Field(..., description="Face boxes in this frame")


class FaceDetections(BaseModel):
    """Face detection results"""
    frames: List[FaceFrame] = Field(..., description="Face detection frames")


class CaptionWord(BaseModel):
    """Caption word"""
    id: str = Field(..., description="Word ID")
    text: str = Field(..., description="Word text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    is_PII: Optional[int] = Field(0, description="PII flag: 0=not PII, 1=PII")


class CaptionSegment(BaseModel):
    """Caption segment"""
    id: str = Field(..., description="Segment ID")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: Optional[str] = Field(None, description="Segment text")
    words: Optional[List[CaptionWord]] = Field(None, description="Word-level timestamps")


class Captions(BaseModel):
    """Caption data"""
    segments: List[CaptionSegment] = Field(..., description="Caption segments")


class PayloadResponse(BaseModel):
    """Full payload response"""
    metadata: TaskMetadata = Field(..., description="Task metadata")
    pii_detections: List[PIIDetection] = Field(..., description="PII detection results")
    face_detections: FaceDetections = Field(..., description="Face detection results")
    captions: Captions = Field(..., description="Caption data")


# === Review API ===
class ManualBox(BaseModel):
    """Manually added detection box"""
    id: str = Field(..., description="Manual box ID")
    label: str = Field(..., description="Box label")
    norm_box: NormalizedBox = Field(..., description="Normalized coordinates")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    source: str = Field("manual_review", description="Source identifier")


class MuteRange(BaseModel):
    """Caption mute range"""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class ReviewCommitRequest(BaseModel):
    """Review commit request"""
    task_id: str = Field(..., description="Task ID")
    deleted_ids: Optional[List[str]] = Field(None, description="IDs to delete")
    manual_boxes: Optional[List[ManualBox]] = Field(None, description="Manual boxes")
    captions_muted: Optional[List[MuteRange]] = Field(None, description="Mute ranges")
    message: Optional[str] = Field(None, description="Commit message")


class ReviewCommitResponse(BaseModel):
    """Review commit response"""
    ok: bool = Field(..., description="Success status")
    review_version: int = Field(..., description="New version number")


# === Finalize API ===
class FinalizeRequest(BaseModel):
    """Finalize request"""
    task_id: str = Field(..., description="Task ID")
    use_version: int = Field(..., description="Review version to use")


class FinalizeResponse(BaseModel):
    """Finalize response"""
    ok: bool = Field(..., description="Success status")
    final_task_id: str = Field(..., description="Final task ID")
    status: str = Field("queued", description="Initial status")


# === Final Result API ===
class FinalResultResponse(BaseModel):
    """Final result response"""
    status: TaskStatus = Field(..., description="Processing status")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    final_video_url: Optional[str] = Field(None, description="Final video URL")
    message: Optional[str] = Field(None, description="Status message")


# === Commits API ===
class CommitStats(BaseModel):
    """Commit statistics"""
    deleted: Optional[int] = Field(None, description="Number of deleted items")
    manual: Optional[int] = Field(None, description="Number of manual items")
    mute_ranges: Optional[int] = Field(None, description="Number of mute ranges")


class CommitItem(BaseModel):
    """Commit item"""
    review_version: int = Field(..., description="Version number")
    created_at: str = Field(..., description="Creation timestamp")
    message: Optional[str] = Field(None, description="Commit message")
    stats: Optional[CommitStats] = Field(None, description="Commit statistics")


class CommitsListResponse(BaseModel):
    """Commits list response"""
    items: List[CommitItem] = Field(..., description="List of commits")


# === History API ===
class HistoryItem(BaseModel):
    """History item"""
    id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    filename: str = Field(..., description="Original filename")
    redacted_video_url: Optional[str] = Field(None, description="Redacted video URL")
    created_at: str = Field(..., description="Creation timestamp")


class HistoryResponse(BaseModel):
    """History response"""
    items: List[HistoryItem] = Field(..., description="List of history items")
