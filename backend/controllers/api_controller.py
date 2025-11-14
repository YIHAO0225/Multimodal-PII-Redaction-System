"""
API Controller for PII Detection and Redaction System
Handles all API endpoints and business logic
"""
import os
import json
import uuid
import time
import asyncio
import math
import cv2
from typing import Optional, Dict, Any, List
from ..Function import s3_tool
from datetime import datetime, timezone
from fastapi import HTTPException, UploadFile
import boto3
from botocore.config import Config
from fastapi.responses import RedirectResponse

from backend.models.schemas import (
    ProcessRequest, ProcessResponse, ResultResponse, PayloadResponse,
    ReviewCommitRequest, ReviewCommitResponse, FinalizeRequest, FinalizeResponse,
    FinalResultResponse, CommitsListResponse, HistoryResponse,
    TaskStatus, TaskMetadata, VideoMetadata, PIIDetection, FaceDetections,
    FaceFrame, FaceBox, Captions, CaptionSegment, CaptionWord,
    NormalizedBox, ManualBox, MuteRange, CommitItem, CommitStats, HistoryItem
)
from backend.Function.upload import handle
from backend.Function.AWSTEST import pii_text_detector
from backend.Function.video_stage1 import stage1
from backend.Function.stage_a import stageA
from backend.Function.audio_detector import audio_detector
from backend.Function.s3_tool import push_AWS_S3, pull_AWS_S3
from backend.Function.video_mosaic_processor import video_mosaic_processor
from backend.Function.audio_dedaction_processor import process_dedaction
from backend.Function.mux_video import mux_text_audio


class TaskManager:
    """In-memory task manager (in production, use Redis or database)"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.final_tasks: Dict[str, Dict[str, Any]] = {}
        self.reviews: Dict[str, List[Dict[str, Any]]] = {}  # task_id -> list of review versions
        self.annotations: Dict[str, List[Dict[str, Any]]] = {}  # task_id -> list of manual annotations
        self.history: List[Dict[str, Any]] = []
    
    def create_task(self, task_id: str, s3_key: str, filename: str, size: int) -> None:
        """Create a new task"""
        print(f"[TaskManager] Creating task: {task_id}")
        self.tasks[task_id] = {
            "task_id": task_id,
            "s3_key": s3_key,
            "filename": filename,
            "size": size,
            "status": TaskStatus.QUEUED,
            "progress": 0,
            "message": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": None,
            "payload": None,
            "video_metadata": None
        }
        self.reviews[task_id] = []
        print(f"[TaskManager] Task created and stored. Total tasks: {len(self.tasks)}")
        print(f"[TaskManager] Task details: {self.tasks[task_id]}")
    
    def update_task_status(self, task_id: str, status, progress: int = None, message: str = None) -> None:
        """Update task status"""
        print(f"[TaskManager] Updating task {task_id} status to {status} (type: {type(status)})")
        if task_id in self.tasks:
            # Convert status to string if it's a TaskStatus enum
            if hasattr(status, 'value'):
                status_value = status.value
                print(f"[TaskManager] Converted enum {status} to string: {status_value}")
            else:
                status_value = status
                print(f"[TaskManager] Using status as-is: {status_value}")
            
            self.tasks[task_id]["status"] = status_value
            if progress is not None:
                self.tasks[task_id]["progress"] = progress
            if message is not None:
                self.tasks[task_id]["message"] = message
            print(f"[TaskManager] Task {task_id} updated successfully with status: {status_value}")
        else:
            print(f"[TaskManager] ERROR: Task {task_id} not found for status update!")
            print(f"[TaskManager] Available tasks: {list(self.tasks.keys())}")
    
    def set_task_payload(self, task_id: str, payload: Dict[str, Any]) -> None:
        """Set task payload"""
        print(f"[TaskManager] Setting payload for task {task_id}")
        if task_id in self.tasks:
            self.tasks[task_id]["payload"] = payload
            self.tasks[task_id]["metadata"] = payload.get("metadata")
            print(f"[TaskManager] Payload set successfully for task {task_id}")
        else:
            print(f"[TaskManager] ERROR: Task {task_id} not found for payload setting!")
            print(f"[TaskManager] Available tasks: {list(self.tasks.keys())}")
    
    def update_video_metadata(self, task_id: str, video_metadata: Dict[str, Any]) -> None:
        """Update cached video metadata for a task"""
        print(f"[TaskManager] Updating video metadata for task {task_id}: {video_metadata}")
        if task_id in self.tasks:
            self.tasks[task_id]["video_metadata"] = video_metadata
        else:
            print(f"[TaskManager] WARNING: Task {task_id} not found while setting video metadata")
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        print(f"[TaskManager] Looking for task: {task_id}")
        print(f"[TaskManager] Available tasks: {list(self.tasks.keys())}")
        result = self.tasks.get(task_id)
        print(f"[TaskManager] Task found: {result is not None}")
        return result
    
    def create_final_task(self, final_task_id: str, task_id: str, use_version: int) -> None:
        """Create final processing task"""
        self.final_tasks[final_task_id] = {
            "final_task_id": final_task_id,
            "task_id": task_id,
            "use_version": use_version,
            "status": TaskStatus.QUEUED,
            "progress": 0,
            "message": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "final_video_url": None
        }
    
    def update_final_task(self, final_task_id: str, status: str, progress: int = None, 
                         message: str = None, final_video_url: str = None) -> None:
        """Update final task status"""
        if final_task_id in self.final_tasks:
            self.final_tasks[final_task_id]["status"] = status
            if progress is not None:
                self.final_tasks[final_task_id]["progress"] = progress
            if message is not None:
                self.final_tasks[final_task_id]["message"] = message
            if final_video_url is not None:
                self.final_tasks[final_task_id]["final_video_url"] = final_video_url
    
    def get_final_task(self, final_task_id: str) -> Optional[Dict[str, Any]]:
        """Get final task by ID"""
        return self.final_tasks.get(final_task_id)
    
    def add_review(self, task_id: str, review_data: Dict[str, Any]) -> int:
        """Add review version"""
        if task_id not in self.reviews:
            self.reviews[task_id] = []
        
        version = len(self.reviews[task_id]) + 1
        review_data["review_version"] = version
        review_data["created_at"] = datetime.now(timezone.utc).isoformat()
        self.reviews[task_id].append(review_data)
        return version
    
    def get_reviews(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all reviews for a task"""
        return self.reviews.get(task_id, [])
    
    def add_to_history(self, task_data: Dict[str, Any]) -> None:
        """Add completed task to history (persist to S3 JSON file)"""
        history_s3_path = "aws-uni-output2/history/db.json"
        print(f"[TaskManager] Adding to history: {task_data.get('id')}")
        
        try:
            # Download existing history file
            try:
                local_history_path = pull_AWS_S3(history_s3_path, region="ap-southeast-2")
                print(f"[TaskManager] Downloaded history file from S3: {local_history_path}")
                with open(local_history_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                # Ensure it's a list
                if not isinstance(history_data, list):
                    history_data = []
            except Exception as e:
                # File doesn't exist yet, start with empty list
                print(f"[TaskManager] History file doesn't exist yet, creating new: {str(e)}")
                history_data = []
                local_history_path = "history_db.json"  # Temporary local file
            
            # Append new task data
            history_data.append(task_data)
            print(f"[TaskManager] Added task to history. Total items: {len(history_data)}")
            
            # Write back to local file
            with open(local_history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            # Upload back to S3
            push_AWS_S3(history_s3_path, local_history_path, region="ap-southeast-2")
            print(f"[TaskManager] History file uploaded to S3: {history_s3_path}")
            
            # Clean up local file
            if os.path.exists(local_history_path) and local_history_path != "history_db.json":
                os.remove(local_history_path)
                print(f"[TaskManager] Local history file cleaned up")
            
            # Also update in-memory cache for faster access
            self.history.append(task_data)
            
        except Exception as e:
            print(f"[TaskManager] ERROR: Failed to persist history to S3: {str(e)}")
            # Fallback: still add to in-memory cache
            self.history.append(task_data)
            import traceback
            print(f"[TaskManager] Traceback: {traceback.format_exc()}")
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get task history"""
        """Get task history (read from S3 JSON file)"""
        history_s3_path = "aws-uni-output2/history/db.json"
        print(f"[TaskManager] Getting history (limit: {limit})")
        
        try:
            # Download history file from S3
            local_history_path = pull_AWS_S3(history_s3_path, region="ap-southeast-2")
            print(f"[TaskManager] Downloaded history file from S3: {local_history_path}")
            
            with open(local_history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            # Ensure it's a list
            if not isinstance(history_data, list):
                print(f"[TaskManager] WARNING: History file is not a list, converting...")
                history_data = []
            
            # Sort by created_at descending and limit
            sorted_history = sorted(history_data, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
            print(f"[TaskManager] Returning {len(sorted_history)} history items (from {len(history_data)} total)")
            
            # Update in-memory cache
            self.history = history_data
            
            # Clean up local file
            if os.path.exists(local_history_path):
                os.remove(local_history_path)
                print(f"[TaskManager] Local history file cleaned up")
            
            return sorted_history
            
        except Exception as e:
            print(f"[TaskManager] ERROR: Failed to load history from S3: {str(e)}")
            # Fallback: return in-memory cache
            print(f"[TaskManager] Falling back to in-memory history ({len(self.history)} items)")
            return sorted(self.history, key=lambda x: x.get("created_at", ""), reverse=True)[:limit][TaskManager]
            print(f"[TaskManager] Available tasks: {list(self.tasks.keys())}")
    
    def add_annotation(self, task_id: str, annotation: Dict[str, Any]) -> None:
        """Add manual annotation to task"""
        print(f"[TaskManager] Adding annotation to task: {task_id}")
        if task_id not in self.annotations:
            self.annotations[task_id] = []
        
        # Add timestamp and ensure required fields
        annotation["created_at"] = datetime.now(timezone.utc).isoformat()
        annotation["source"] = "manual_review"
        
        self.annotations[task_id].append(annotation)
        print(f"[TaskManager] Annotation added. Total annotations for {task_id}: {len(self.annotations[task_id])}")
        print(f"[TaskManager] Annotation details: {annotation}")
    
    def get_annotations(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all manual annotations for a task"""
        print(f"[TaskManager] Getting annotations for task: {task_id}")
        annotations = self.annotations.get(task_id, [])
        print(f"[TaskManager] Found {len(annotations)} annotations for {task_id}")
        return annotations
    
    def clear_annotations(self, task_id: str) -> None:
        """Clear all annotations for a task"""
        print(f"[TaskManager] Clearing annotations for task: {task_id}")
        if task_id in self.annotations:
            count = len(self.annotations[task_id])
            self.annotations[task_id] = []
            print(f"[TaskManager] Cleared {count} annotations for {task_id}")
        else:
            print(f"[TaskManager] No annotations found for {task_id}")


# Global task manager instance
task_manager = TaskManager()


class APIController:
    """Main API controller"""
    
    @staticmethod
    async def create_task(request: Optional[ProcessRequest] = None, file: Optional[UploadFile] = None) -> ProcessResponse:
        """Create a new processing task"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        print(f"[API] Creating task: {task_id}")
        
        if request and request.s3_key:
            # Method A: Direct S3 upload (preferred)
            print(f"[API] Method A: Direct S3 upload - {request.s3_key}")
            task_manager.create_task(
                task_id=task_id,
                s3_key=request.s3_key,
                filename=request.filename or "unknown.mp4",
                size=request.size or 0
            )
            
            # Start background processing
            print(f"[API] Starting background task for {task_id}")
            background_task = asyncio.create_task(APIController._process_task_background(task_id, request.s3_key))
            print(f"[API] Background task created: {background_task}")
            
            # Add a callback to track task completion
            def task_done_callback(task):
                print(f"[API] Background task completed: {task_id}, exception: {task.exception()}")
            
            background_task.add_done_callback(task_done_callback)
            
        elif file:
            # Method B: File upload
            print(f"[API] Method B: File upload - {file.filename}")
            try:
                # Upload file to S3
                workdir = f"/tmp/work_{task_id}"
                os.makedirs(workdir, exist_ok=True)
                
                upload_result = await handle(file, workdir)
                s3_key = upload_result["s3_path"]
                print(f"[API] File uploaded to S3: {s3_key}")
                
                task_manager.create_task(
                    task_id=task_id,
                    s3_key=s3_key,
                    filename=file.filename or "unknown.mp4",
                    size=file.size or 0
                )

                local_video_path = upload_result.get("local_path")
                local_video_metadata = APIController._extract_video_metadata_from_local(local_video_path)
                if local_video_metadata:
                    task_manager.update_video_metadata(task_id, local_video_metadata)
                
                # Start background processing
                print(f"[API] Starting background task for {task_id}")
                background_task = asyncio.create_task(APIController._process_task_background(task_id, s3_key))
                print(f"[API] Background task created: {background_task}")
                
                # Add a callback to track task completion
                def task_done_callback(task):
                    print(f"[API] Background task completed: {task_id}, exception: {task.exception()}")
                    if task.exception():
                        print(f"[API] Background task failed with exception: {task.exception()}")
                        import traceback
                        print(f"[API] Exception traceback: {traceback.format_exc()}")
                
                background_task.add_done_callback(task_done_callback)
                
            except Exception as e:
                print(f"[API] File upload failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
        else:
            print(f"[API] Error: No request data or file provided")
            raise HTTPException(status_code=400, detail="Either s3_key or file must be provided")
        
        print(f"[API] Task created successfully: {task_id}")
        print(f"[API] Task manager now has {len(task_manager.tasks)} tasks: {list(task_manager.tasks.keys())}")
        
        # Verify the task exists
        stored_task = task_manager.get_task(task_id)
        if stored_task:
            print(f"[API] Task verification successful: {stored_task}")
        else:
            print(f"[API] ERROR: Task was not stored properly!")
            print(f"[API] TaskManager ID: {id(task_manager)}")
            print(f"[API] TaskManager tasks dict: {task_manager.tasks}")
        
        # Add a small delay to see if the task persists
        async def check_task_persistence():
            await asyncio.sleep(1)  # Wait 1 second
            task_after_delay = task_manager.get_task(task_id)
            if task_after_delay:
                print(f"[API] Task {task_id} still exists after 1 second delay")
            else:
                print(f"[API] ERROR: Task {task_id} lost after 1 second delay!")
                print(f"[API] Available tasks after delay: {list(task_manager.tasks.keys())}")
        
        # Schedule the check (don't await it)
        asyncio.create_task(check_task_persistence())
        
        return ProcessResponse(task_id=task_id)
    
    @staticmethod
    def _extract_video_metadata_from_local(video_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Extract basic metadata (width, height, fps, duration) from a local video file."""
        if not video_path:
            print("[API] _extract_video_metadata_from_local: No video path provided")
            return None
        
        if not os.path.exists(video_path):
            print(f"[API] _extract_video_metadata_from_local: Path does not exist - {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[API] _extract_video_metadata_from_local: Failed to open video - {video_path}")
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        duration = (frame_count / fps) if fps and fps > 0 else 0.0
        metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": frame_count,
            "duration": duration
        }
        print(f"[API] _extract_video_metadata_from_local: {metadata}")
        return metadata
    
    @staticmethod
    async def _process_task_background(task_id: str, s3_key: str) -> None:
        """Background task processing"""
        print(f"[API] ===== STARTING BACKGROUND PROCESSING =====")
        print(f"[API] Task ID: {task_id}")
        print(f"[API] S3 Key: {s3_key}")
        print(f"[API] ===== PROCESSING STEPS =====")
        
        # Verify task exists before processing
        task = task_manager.get_task(task_id)
        if not task:
            print(f"[API] ERROR: Task {task_id} not found before processing!")
            return
        print(f"[API] Task found before processing: {task}")
        
        try:
            print(f"[API] About to update status to PROCESSING with progress 10...")
            task_manager.update_task_status(task_id, TaskStatus.PROCESSING, 10, "Starting processing...")
            print(f"[API] Step 1/5: Starting processing... Status updated to PROCESSING")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            
            # Verify task still exists after status update
            task_after_update = task_manager.get_task(task_id)
            if not task_after_update:
                print(f"[API] ERROR: Task {task_id} lost after status update!")
                return
            print(f"[API] Task still exists after status update: {task_after_update}")
            
            # Step 1: Audio detection
            print(f"[API] Step 2/5: Processing audio...")
            task_manager.update_task_status(task_id, TaskStatus.PROCESSING, 20, "Processing audio...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            try:
                audio_result = audio_detector(s3_key, region="ap-southeast-2")
                print(f"[API] ✅ Audio detection completed")
            except Exception as e:
                print(f"[API] ❌ Audio detection failed: {str(e)}")
                raise e
            
            # Step 2: Video stage 1
            print(f"[API] Step 3/5: Processing video frames...")
            task_manager.update_task_status(task_id, TaskStatus.PROCESSING, 40, "Processing video frames...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            video_output_dir = None
            video_metadata = None
            try:
                video_stage1_result = stage1(s3_key, region="ap-southeast-2")
                video_output_dir = video_stage1_result["output_dir"]
                video_metadata = video_stage1_result.get("video_info")
                print(f"[API] ✅ Video stage 1 completed: {video_output_dir}")
                if video_metadata:
                    print(f"[API] Video metadata: {video_metadata}")
                    task_manager.update_video_metadata(task_id, video_metadata)
            except Exception as e:
                print(f"[API] ❌ Video stage 1 failed: {str(e)}")
                raise e
            
            if not video_output_dir:
                raise RuntimeError("Stage1 did not return an output directory")
            
            # Step 3: Face detection
            print(f"[API] Step 4/5: Detecting faces...")
            task_manager.update_task_status(task_id, TaskStatus.PROCESSING, 60, "Detecting faces...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            try:
                face_result = stageA(video_output_dir, region="ap-southeast-2")
                print(f"[API] ✅ Face detection completed: {face_result}")
            except Exception as e:
                print(f"[API] ❌ Face detection failed: {str(e)}")
                raise e
            
            # Step 4: PII detection
            print(f"[API] Step 5/5: Detecting PII...")
            task_manager.update_task_status(task_id, TaskStatus.PROCESSING, 80, "Detecting PII...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            try:
                pii_result = pii_text_detector(video_output_dir, region="ap-southeast-2")
                print(f"[API] ✅ PII detection completed: {pii_result}")
            except Exception as e:
                print(f"[API] ❌ PII detection failed: {str(e)}")
                raise e
            
            # Step 5: Load and format results
            print(f"[API] Final step: Formatting results...")
            task_manager.update_task_status(task_id, TaskStatus.PROCESSING, 90, "Formatting results...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            try:
                payload = await APIController._load_and_format_results(task_id, s3_key, pii_result)
                task_manager.set_task_payload(task_id, payload)
                print(f"[API] ✅ Results formatted and stored")
            except Exception as e:
                print(f"[API] ❌ Results formatting failed: {str(e)}")
                raise e
            
            task_manager.update_task_status(task_id, TaskStatus.DONE, 100, "Processing completed")
            print(f"[API] ===== PROCESSING COMPLETED SUCCESSFULLY =====")
            print(f"[API] Task {task_id}: Status set to DONE")
            
            # Verify task still exists after completion
            final_task = task_manager.get_task(task_id)
            if final_task:
                print(f"[API] Task {task_id} still exists after completion: {final_task}")
            else:
                print(f"[API] ERROR: Task {task_id} lost after completion!")
                print(f"[API] Available tasks: {list(task_manager.tasks.keys())}")
            
        except Exception as e:
            print(f"[API] ===== PROCESSING FAILED =====")
            print(f"[API] Task {task_id}: Error - {str(e)}")
            print(f"[API] Error type: {type(e).__name__}")
            import traceback
            print(f"[API] Traceback: {traceback.format_exc()}")
            
            # Verify task still exists before updating status
            task_before_update = task_manager.get_task(task_id)
            if not task_before_update:
                print(f"[API] ERROR: Task {task_id} lost before error status update!")
                print(f"[API] Available tasks: {list(task_manager.tasks.keys())}")
                return
            
            task_manager.update_task_status(task_id, TaskStatus.FAILED, 0, f"Processing failed: {str(e)}")
            print(f"[API] Task {task_id}: Status set to FAILED")
            
            # Verify task still exists after error status update
            task_after_error = task_manager.get_task(task_id)
            if task_after_error:
                print(f"[API] Task {task_id} still exists after error status update: {task_after_error}")
            else:
                print(f"[API] ERROR: Task {task_id} lost after error status update!")
                print(f"[API] Available tasks: {list(task_manager.tasks.keys())}")
    
    @staticmethod
    async def _load_and_format_results(task_id: str, s3_key: str, pii_result: Dict[str, Any]) -> Dict[str, Any]:
        """Load and format detection results"""
        print(f"[API] Loading and formatting results for task: {task_id}")
        try:
            # Load unified detection JSON
            unified_json_path = pii_result.get("output_path", "") + pii_result.get("unified_json", "")
            print(f"[API] Loading JSON from: {unified_json_path}")
            
            # Download and parse unified JSON
            local_json_path = pull_AWS_S3(unified_json_path, region="ap-southeast-2")
            print(f"[API] Downloaded JSON to: {local_json_path}")
            
            with open(local_json_path, 'r', encoding='utf-8') as f:
                unified_data = json.load(f)
            
            print(f"[API] JSON loaded successfully. Keys: {list(unified_data.keys())}")
            print(f"[API] PII detections: {len(unified_data.get('pii_detections', []))}")
            print(f"[API] Face detections: {len(unified_data.get('face_detections', {}).get('frames', []))}")
            print(f"[API] Captions: {len(unified_data.get('captions', {}).get('segments', []))}")
            
            # Debug: Show sample PII detection structure
            if unified_data.get('pii_detections'):
                sample_pii = unified_data['pii_detections'][0]
                print(f"[API] Sample PII structure:")
                print(f"  - Keys: {list(sample_pii.keys())}")
                print(f"  - Raw confidence: {sample_pii.get('confidence')} (type: {type(sample_pii.get('confidence'))})")
                print(f"  - Raw start_time: {sample_pii.get('start_time')} (type: {type(sample_pii.get('start_time'))})")
                print(f"  - Raw end_time: {sample_pii.get('end_time')} (type: {type(sample_pii.get('end_time'))})")
            
            # Debug: Show sample face detection structure
            if unified_data.get('face_detections', {}).get('frames'):
                sample_frame = unified_data['face_detections']['frames'][0]
                print(f"[API] Sample Face Frame structure:")
                print(f"  - Keys: {list(sample_frame.keys())}")
                print(f"  - Raw confidence: {sample_frame.get('confidence')} (type: {type(sample_frame.get('confidence'))})")
                print(f"  - Raw start_time: {sample_frame.get('start_time')} (type: {type(sample_frame.get('start_time'))})")
                print(f"  - Raw end_time: {sample_frame.get('end_time')} (type: {type(sample_frame.get('end_time'))})")
            
            # Clean up local file
            os.remove(local_json_path)
            
            task_data = task_manager.get_task(task_id)
            cached_video_metadata = {}
            if task_data:
                cached_video_metadata = task_data.get("video_metadata") or {}
            
            video_width = cached_video_metadata.get("width")
            video_height = cached_video_metadata.get("height")
            video_duration = cached_video_metadata.get("duration")
            video_fps = cached_video_metadata.get("fps")

            # Format for API response
            metadata = TaskMetadata(
                task_id=task_id,
                video=VideoMetadata(
                    s3_key=s3_key,
                    width=int(video_width) if isinstance(video_width, (int, float)) and video_width and video_width > 0 and not math.isnan(float(video_width)) else 1920,
                    height=int(video_height) if isinstance(video_height, (int, float)) and video_height and video_height > 0 and not math.isnan(float(video_height)) else 1080,
                    duration=float(video_duration) if isinstance(video_duration, (int, float)) and not math.isnan(float(video_duration)) else 60.0,
                    fps=float(video_fps) if isinstance(video_fps, (int, float)) and video_fps and video_fps > 0 and not math.isnan(float(video_fps)) else None
                ),
                generated_at=datetime.now(timezone.utc).isoformat(),
                model_versions={
                    "ocr": "v2.1.0",
                    "ner": "v1.6.3", 
                    "asr": "whisper-large-v3",
                    "face": "retinaface-0.0.2"
                }
            )
            
            # Convert PII detections
            pii_detections = []
            print(f"[API] Processing {len(unified_data.get('pii_detections', []))} PII detections")
            for item in unified_data.get("pii_detections", []):
                # Debug: Show available fields in the JSON item
                print(f"[API] Available fields in PII item: {list(item.keys())}")
                
                # Extract confidence, start time, and end time with test output
                confidence = item.get("confidence")
                start_time = item.get("start_time", 0.0)
                end_time = item.get("end_time", 0.0)
                
                # Handle confidence conversion - ensure it's a float or 0
                if confidence is not None:
                    try:
                        confidence = float(confidence)
                    except (ValueError, TypeError):
                        print(f"[API] WARNING: Invalid confidence value '{confidence}', setting to 0")
                        confidence = 0.0
                else:
                    # Replace None confidence with 0
                    print(f"[API] WARNING: Confidence is None for PII {item.get('id', 'unknown')}, setting to 0")
                    confidence = 0.0
                
                # Handle start_time and end_time conversion
                try:
                    start_time = float(start_time) if start_time is not None else 0.0
                except (ValueError, TypeError):
                    start_time = 0.0
                    
                try:
                    end_time = float(end_time) if end_time is not None else 0.0
                except (ValueError, TypeError):
                    end_time = 0.0
                
                # Test output for debugging
                print(f"[API] PII Detection - ID: {item.get('id', f'pii_{item.get('index', 0)}')}")
                print(f"  - Label: {item.get('label', 'UNKNOWN')}")
                print(f"  - Text: {item.get('text')}")
                print(f"  - Confidence: {confidence} (type: {type(confidence)})")
                print(f"  - Start Time: {start_time} (type: {type(start_time)})")
                print(f"  - End Time: {end_time} (type: {type(end_time)})")
                print(f"  - Source: {item.get('source', 'ocr')}")
                
                pii_detections.append(PIIDetection(
                    id=item.get("id", f"pii_{item.get('index', 0)}"),
                    label=item.get("label", "UNKNOWN"),
                    text=item.get("text"),
                    confidence=confidence,
                    start=start_time,
                    end=end_time,
                    norm_box=NormalizedBox(
                        x1=item.get("normalized_coordinates", {}).get("x_min", 0.0),
                        y1=item.get("normalized_coordinates", {}).get("y_min", 0.0),
                        x2=item.get("normalized_coordinates", {}).get("x_max", 0.0),
                        y2=item.get("normalized_coordinates", {}).get("y_max", 0.0)
                    ),
                    source=item.get("source", "ocr")
                ))
            
            # Convert face detections
            face_frames = []
            for frame in unified_data.get("face_detections", {}).get("frames", []):
                frame_time = frame.get("start_time", 0.0)
                frame_confidence = frame.get("confidence")  # Confidence is at frame level, not box level
                
                # Handle None confidence for face frames
                if frame_confidence is None:
                    print(f"[API] WARNING: Face frame confidence is None, setting to 0")
                    frame_confidence = 0.0
                else:
                    try:
                        frame_confidence = float(frame_confidence)
                    except (ValueError, TypeError):
                        print(f"[API] WARNING: Invalid face frame confidence value '{frame_confidence}', setting to 0")
                        frame_confidence = 0.0
                
                print(f"[API] Face Frame - Time: {frame_time}s, Confidence: {frame_confidence}")
                
                face_boxes = []
                for box in frame.get("boxes", []):
                    print(f"  - Face Box ID: {box.get('id')}")
                    print(f"    - Frame Confidence: {frame_confidence}")  # Use frame confidence for all boxes
                    print(f"    - Coordinates: ({box.get('normalized_coordinates', {}).get('x_min', 0.0)}, {box.get('normalized_coordinates', {}).get('y_min', 0.0)}) to ({box.get('normalized_coordinates', {}).get('x_max', 0.0)}, {box.get('normalized_coordinates', {}).get('y_max', 0.0)})")
                    
                    face_boxes.append(FaceBox(
                        id=box.get("id"),
                        x1=box.get("normalized_coordinates", {}).get("x_min", 0.0),
                        y1=box.get("normalized_coordinates", {}).get("y_min", 0.0),
                        x2=box.get("normalized_coordinates", {}).get("x_max", 0.0),
                        y2=box.get("normalized_coordinates", {}).get("y_max", 0.0),
                        confidence=frame_confidence  # Use frame confidence for all boxes
                    ))
                
                face_frames.append(FaceFrame(
                    id=frame.get("id"),
                    t=frame_time,
                    end=frame.get("end_time"),
                    confidence=frame_confidence,
                    frame_range=frame.get("frame_range"),
                    extra_metadata=frame.get("extra_metadata"),
                    norm_boxes=face_boxes
                ))
            
            # Convert captions from audio processing (separate S3 file)
            task = task_data or task_manager.get_task(task_id)
            video_key = task["s3_key"]
            video_filename = os.path.basename(video_key)
            video_basename = os.path.splitext(video_filename)[0]
            captions_path = f"aws-uni-output1/audio/{video_filename}/transcript_segments.json"
            print(f"[API] Loading captions from S3 path: {captions_path}")
            
            try:
                # pull_AWS_S3 returns a local file path, not the content
                local_file_path = pull_AWS_S3(captions_path,region="ap-southeast-2")
                print(f"[API] Downloaded captions to local path: {local_file_path}")
                
                if not local_file_path or not os.path.exists(local_file_path):
                    print(f"[API] WARNING: Captions file not found at {local_file_path}")
                    captions_data = {"segments": []}
                else:
                    # Read the local file content
                    with open(local_file_path, 'r', encoding='utf-8') as f:
                        captions_raw = f.read()
                    
                    print(f"[API] Captions file size: {len(captions_raw)} characters")
                    print(f"[API] Captions file preview: {captions_raw[:200]}...")
                    
                    if captions_raw.strip() == "":
                        print(f"[API] WARNING: Empty captions file")
                        captions_data = {"segments": []}
                    else:
                        captions_data = json.loads(captions_raw)
                        print(f"[API] Successfully loaded captions data")
            except json.JSONDecodeError as e:
                print(f"[API] JSON ERROR: Failed to parse captions JSON: {e}")
                print(f"[API] Raw data that failed to parse: {captions_raw[:500] if 'captions_raw' in locals() else 'None'}")
                captions_data = {"segments": []}
            except Exception as e:
                print(f"[API] ERROR: Failed to load captions from {captions_path}: {e}")
                captions_data = {"segments": []}
            
            print(f"[API] Processing {len(captions_data.get('segments', []))} caption segments")
            caption_segments = []
            for seg_idx, seg in enumerate(captions_data.get("segments", [])):
                # Debug: Show segment structure
                print(f"[API] Caption Segment [{seg_idx}]:")
                print(f"  - ID: {seg.get('id', 'unknown')}")
                print(f"  - Start Time: {seg.get('start_time', 0.0)}")
                print(f"  - End Time: {seg.get('end_time', 0.0)}")
                print(f"  - Text: {seg.get('text', '')[:50]}...")
                print(f"  - Words Count: {len(seg.get('words', []))}")
                
                # Convert words from audio detector format to frontend format
                words = []
                for i, word_data in enumerate(seg.get("words", [])):
                    word_id = word_data.get("id", f"w_{seg.get('id', 0)}_{i}")
                    word_text = word_data.get("word", "")
                    word_start = word_data.get("start_time", 0.0)
                    word_end = word_data.get("end_time", 0.0)
                    word_is_PII = word_data.get("is_PII", 0)
                    
                    words.append(CaptionWord(
                        id=word_id,
                        text=word_text,
                        start=word_start,
                        end=word_end,
                        is_PII=word_is_PII  # Default to 0 if not present
                    ))
                
                caption_segments.append(CaptionSegment(
                    id=str(seg.get("id", 0)),
                    start=seg.get("start_time", 0.0),
                    end=seg.get("end_time", 0.0),
                    text=seg.get("text", ""),
                    words=words
                ))
            
            captions = Captions(segments=caption_segments)
            print(f"[API] Captions Summary:")
            print(f"  - Total Segments: {len(caption_segments)}")
            print(f"  - Total Words: {sum(len(seg.words) for seg in caption_segments)}")
            if caption_segments:
                print(f"  - First Segment: '{caption_segments[0].text[:30]}...' ({len(caption_segments[0].words)} words)")
                print(f"  - Last Segment: '{caption_segments[-1].text[:30]}...' ({len(caption_segments[-1].words)} words)")
            
            result = {
                "metadata": metadata.dict(),
                "pii_detections": [pii.dict() for pii in pii_detections],
                "face_detections": {"frames": [frame.dict() for frame in face_frames]},
                "captions": captions.dict()
            }
            
            # Summary test output
            print(f"[API] ===== DETECTION SUMMARY =====")
            print(f"[API] Total PII Detections: {len(result['pii_detections'])}")
            print(f"[API] Total Face Frames: {len(result['face_detections']['frames'])}")
            print(f"[API] Total Caption Segments: {len(result['captions']['segments'])}")
            
            # Show PII detection statistics
            if result['pii_detections']:
                pii_with_confidence = [pii for pii in result['pii_detections'] if pii.get('confidence') is not None and pii.get('confidence') > 0]
                pii_with_zero_confidence = [pii for pii in result['pii_detections'] if pii.get('confidence') == 0]
                pii_with_timing = [pii for pii in result['pii_detections'] if pii.get('start', 0) > 0 or pii.get('end', 0) > 0]
                print(f"[API] PII with confidence scores: {len(pii_with_confidence)}")
                print(f"[API] PII with zero confidence (converted from None): {len(pii_with_zero_confidence)}")
                print(f"[API] PII with timing data: {len(pii_with_timing)}")
                
                # Show confidence range if available
                confidences = [pii.get('confidence') for pii in result['pii_detections'] if pii.get('confidence') is not None]
                if confidences:
                    print(f"[API] Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            
            # Show face detection statistics
            total_face_boxes = sum(len(frame.get('norm_boxes', [])) for frame in result['face_detections']['frames'])
            print(f"[API] Total Face Boxes: {total_face_boxes}")
            
            print(f"[API] ===== END SUMMARY =====")
            return result
            
        except Exception as e:
            print(f"[API] Error loading results: {str(e)}")
            # Return empty results on error
            return {
                "metadata": {
                    "task_id": task_id,
                    "video": {
                        "s3_key": s3_key,
                        "width": 1280,
                        "height": 720,
                        "duration": 60.0,
                        "fps": 29.97
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model_versions": {}
                },
                "pii_detections": [],
                "face_detections": {"frames": []},
                "captions": {"segments": []}
            }
    
    @staticmethod
    def get_result(task_id: str) -> ResultResponse:
        """Get task processing status"""
        print(f"[API] Getting result for task: {task_id}")
        print(f"[API] TaskManager ID: {id(task_manager)}")
        print(f"[API] Task manager has {len(task_manager.tasks)} tasks: {list(task_manager.tasks.keys())}")
        print(f"[API] TaskManager tasks dict: {task_manager.tasks}")
        task = task_manager.get_task(task_id)
        if not task:
            print(f"[API] Task not found: {task_id}")
            print(f"[API] Available tasks: {list(task_manager.tasks.keys())}")
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Get status value - ensure it's a TaskStatus enum
        status_value = task["status"]
        print(f"[API] Raw status from task: {status_value} (type: {type(status_value)})")
        
        # Convert to TaskStatus enum if it's a string
        if isinstance(status_value, str):
            try:
                status_enum = TaskStatus(status_value)
                print(f"[API] Converted string '{status_value}' to TaskStatus enum: {status_enum}")
            except ValueError:
                print(f"[API] ERROR: Invalid status string '{status_value}', using QUEUED")
                status_enum = TaskStatus.QUEUED
        elif isinstance(status_value, TaskStatus):
            status_enum = status_value
            print(f"[API] Status is already TaskStatus enum: {status_enum}")
        else:
            print(f"[API] ERROR: Unknown status type {type(status_value)}, using QUEUED")
            status_enum = TaskStatus.QUEUED
        
        progress_value = task.get("progress")
        message_value = task.get("message")
        print(f"[API] Task {task_id} final status: {status_enum}, progress: {progress_value}%, message: {message_value}")
        print(f"[API] Status enum type: {type(status_enum)}, value: {status_enum.value if hasattr(status_enum, 'value') else status_enum}")
        
        response = ResultResponse(
            status=status_enum,
            progress=progress_value,
            message=message_value
        )
        print(f"[API] ResultResponse created: status={response.status}, progress={response.progress}, message={response.message}")
        print(f"[API] ResultResponse status type: {type(response.status)}")
        return response
    
    @staticmethod
    def get_payload(task_id: str, use_version: Optional[int] = None) -> PayloadResponse:
        """Get full payload with optional version preview"""
        print(f"[API] Getting payload for task: {task_id}, version: {use_version}")
        task = task_manager.get_task(task_id)
        if not task:
            print(f"[API] Task not found: {task_id}")
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task["status"] != TaskStatus.DONE:
            print(f"[API] Task {task_id} not completed, status: {task['status']}")
            raise HTTPException(status_code=400, detail="Task not completed")
        
        payload = task.get("payload")
        if not payload:
            print(f"[API] No payload available for task: {task_id}")
            raise HTTPException(status_code=404, detail="Payload not available")
        
        print(f"[API] Payload found: {len(payload.get('pii_detections', []))} PII, {len(payload.get('face_detections', {}).get('frames', []))} face frames")
        
        # Test output: Show raw payload structure
        print(f"[API] ===== PAYLOAD DEBUG INFO =====")
        print(f"[API] Raw payload keys: {list(payload.keys())}")
        
        # Test output: Show PII detections details
        pii_detections = payload.get('pii_detections', [])
        print(f"[API] PII Detections ({len(pii_detections)} total):")
        for i, pii in enumerate(pii_detections[:3]):  # Show first 3 for debugging
            print(f"  [{i}] ID: {pii.get('id', 'N/A')}")
            print(f"      Label: {pii.get('label', 'N/A')}")
            print(f"      Text: {pii.get('text', 'N/A')}")
            print(f"      Confidence: {pii.get('confidence', 'N/A')} (type: {type(pii.get('confidence'))})")
            print(f"      Start: {pii.get('start', 'N/A')} (type: {type(pii.get('start'))})")
            print(f"      End: {pii.get('end', 'N/A')} (type: {type(pii.get('end'))})")
            print(f"      Source: {pii.get('source', 'N/A')}")
        
        if len(pii_detections) > 3:
            print(f"  ... and {len(pii_detections) - 3} more PII detections")
        
        # Test output: Show face detections details
        face_detections = payload.get('face_detections', {})
        face_frames = face_detections.get('frames', [])
        print(f"[API] Face Detections ({len(face_frames)} frames):")
        for i, frame in enumerate(face_frames[:2]):  # Show first 2 frames for debugging
            print(f"  Frame [{i}] Time: {frame.get('t', 'N/A')}s")
            boxes = frame.get('norm_boxes', [])
            print(f"      Boxes: {len(boxes)}")
            for j, box in enumerate(boxes[:2]):  # Show first 2 boxes per frame
                print(f"        Box [{j}] ID: {box.get('id', 'N/A')}")
                print(f"              Confidence: {box.get('confidence', 'N/A')} (type: {type(box.get('confidence'))})")
                print(f"              Coords: ({box.get('x1', 0):.3f}, {box.get('y1', 0):.3f}) to ({box.get('x2', 0):.3f}, {box.get('y2', 0):.3f})")
        
        if len(face_frames) > 2:
            print(f"  ... and {len(face_frames) - 2} more face frames")
        
        # Apply review version if specified
        if use_version:
            reviews = task_manager.get_reviews(task_id)
            if use_version > len(reviews):
                print(f"[API] Review version {use_version} not found for task {task_id}")
                raise HTTPException(status_code=404, detail="Review version not found")
            
            # Apply review changes to payload
            payload = APIController._apply_review_version(payload, reviews[use_version - 1])
            print(f"[API] Applied review version {use_version}")
        
        # Convert dict to PayloadResponse object
        result = PayloadResponse(
            metadata=TaskMetadata(**payload["metadata"]),
            pii_detections=[PIIDetection(**pii) for pii in payload["pii_detections"]],
            face_detections=FaceDetections(**payload["face_detections"]),
            captions=Captions(**payload["captions"])
        )
        
        # Test output: Show final result structure
        print(f"[API] ===== FINAL RESULT DEBUG =====")
        print(f"[API] Final PII detections: {len(result.pii_detections)}")
        for i, pii in enumerate(result.pii_detections[:2]):  # Show first 2 final results
            print(f"  Final PII [{i}]:")
            print(f"    - ID: {pii.id}")
            print(f"    - Label: {pii.label}")
            print(f"    - Text: {pii.text}")
            print(f"    - Confidence: {pii.confidence} (type: {type(pii.confidence)})")
            print(f"    - Start: {pii.start} (type: {type(pii.start)})")
            print(f"    - End: {pii.end} (type: {type(pii.end)})")
            print(f"    - Source: {pii.source}")
        
        print(f"[API] Final face frames: {len(result.face_detections.frames)}")
        for i, frame in enumerate(result.face_detections.frames[:1]):  # Show first frame
            print(f"  Final Face Frame [{i}]:")
            print(f"    - Time: {frame.t}s")
            print(f"    - Boxes: {len(frame.norm_boxes)}")
            for j, box in enumerate(frame.norm_boxes[:1]):  # Show first box
                print(f"      Box [{j}]:")
                print(f"        - ID: {box.id}")
                print(f"        - Confidence: {box.confidence} (type: {type(box.confidence)})")
                print(f"        - Coords: ({box.x1:.3f}, {box.y1:.3f}) to ({box.x2:.3f}, {box.y2:.3f})")
        
        print(f"[API] ===== END PAYLOAD DEBUG =====")
        print(f"[API] Returning payload with {len(result.pii_detections)} PII detections")
        return result
    
    @staticmethod
    def _apply_review_version(payload: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
        """Apply review version to payload"""
        # This is a simplified implementation
        # In production, you'd need to properly merge the review changes
        return payload
    
    @staticmethod
    def commit_review(request: Dict[str, Any]) -> ReviewCommitResponse:
        """Commit review changes"""
        print(f"[API] ===== COMMIT REVIEW =====")
        print(f"[API] Request: {request}")
        
        # Extract task_id from dictionary
        task_id = request.get('task_id')
        if not task_id:
            print(f"[API] ERROR: Missing task_id in request")
            raise HTTPException(status_code=400, detail="task_id is required")
        
        print(f"[API] Task ID: {task_id}")
        
        task = task_manager.get_task(task_id)
        if not task:
            print(f"[API] ERROR: Task {task_id} not found")
            raise HTTPException(status_code=404, detail="Task not found")
        
        print(f"[API] Task found: {task['task_id']}")
        
        # Create review data from dictionary
        review_data = {
            "deleted_ids": request.get('deleted_ids', []),
            "manual_boxes": request.get('manual_boxes', []),
            "captions_muted": request.get('captions_muted', []),
            "message": request.get('message')
        }
        
        video_key = task["s3_key"]
        video_filename = os.path.basename(video_key)
        video_basename = os.path.splitext(video_filename)[0]
        text_s3_path = f"aws-uni-output1/text_framesandjson_output/{video_basename}/{video_basename}.json"
        audio_s3_path = f"aws-uni-output1/audio/{video_filename}/transcript_segments.json"
        text_s3_output_path = f"aws-uni-output1/text_framesandjson_output/{video_basename}/{video_basename}_reviewed.json"
        audio_s3_output_path = f"aws-uni-output1/audio/{video_filename}/transcript_segments_reviewed.json"

        # Read and update text PII JSON file
        print(f"[API] Updating text PII JSON file: {text_s3_path}")
        try:
            # Download the current text PII JSON file
            local_text_json_path = pull_AWS_S3(text_s3_path, region="ap-southeast-2")
            print(f"[API] Downloaded text JSON to: {local_text_json_path}")
            
            local_audio_json_path = pull_AWS_S3(audio_s3_path, region="ap-southeast-2")
            print(f"[API] Downloaded audio JSON to: {local_audio_json_path}")
            # Read the current JSON content
            with open(local_text_json_path, 'r', encoding='utf-8') as f:
                text_json_data = json.load(f)
            with open(local_audio_json_path, 'r', encoding='utf-8') as f:
                audio_json_data = json.load(f)
            
            print(f"[API] Original text JSON structure: {list(text_json_data.keys())}")
            print(f"[API] Original PII detections: {len(text_json_data.get('pii_detections', []))}")
            
            # Get the review data for this version
            deleted_ids = review_data.get('deleted_ids', [])
            deleted_set = set(deleted_ids)
            manual_boxes = review_data.get('manual_boxes', [])
            fps = task.get('fps', 25.0)
            captions_muted = review_data.get('captions_muted', [])
            
            print(f"[API] Review changes:")
            print(f"  - Deleted IDs: {len(deleted_ids)}")
            print(f"  - Manual boxes: {len(manual_boxes)}")
            
            # Remove PIIs with IDs in deleted_ids
            if deleted_ids:
                marked = 0
                for pii in text_json_data.get('pii_detections', []):
                    if pii.get('id') in deleted_set:
                        pii['is_pii'] = 0
                        marked += 1
                print(f"[API] Marked {marked} detections as non-PII based on deleted IDs")
            
            # Add manual boxes as new PII detections
            if manual_boxes:
                print(f"[API] Adding {len(manual_boxes)} manual PII detections")
                
                # Get the next available index
                existing_indices = [pii.get('index', 0) for pii in text_json_data.get('pii_detections', [])]
                next_index = max(existing_indices, default=0) + 1
                
                for i, manual_box in enumerate(manual_boxes):
                    # Create a new PII detection from manual box

                    start_t = float(manual_box.get("start", 0.0) or 0.0)
                    end_t = float(manual_box.get("end", 0.0) or 0.0)
                    frames_start = max(0, int(math.floor(start_t * fps)))
                    frames_end = max(frames_start, int(math.ceil(end_t * fps)))
                    frame_range = f"{frames_start}-{frames_end}"
                    new_pii_detection = {
                        "index": next_index + i,
                        "id": f"manual_{next_index + i}",
                        "source_image": "manual_review",
                        "text": manual_box.get('text', 'Manual Detection'),
                        "label": manual_box.get('label', 'MANUAL'),
                        "confidence": 1.0,  # Manual detections have full confidence
                        "is_pii": 1,
                        "coordinates": {
                            "x_min": int(manual_box.get('pixel_box', {}).get('x', 0)),
                            "y_min": int(manual_box.get('pixel_box', {}).get('y', 0)),
                            "x_max": int(manual_box.get('pixel_box', {}).get('x', 0) + manual_box.get('pixel_box', {}).get('w', 0)),
                            "y_max": int(manual_box.get('pixel_box', {}).get('y', 0) + manual_box.get('pixel_box', {}).get('h', 0))
                        },
                        "normalized_coordinates": {
                            "x_min": manual_box.get('norm_box', {}).get('x1', 0.0),
                            "y_min": manual_box.get('norm_box', {}).get('y1', 0.0),
                            "x_max": manual_box.get('norm_box', {}).get('x2', 0.0),
                            "y_max": manual_box.get('norm_box', {}).get('y2', 0.0)
                        },
                        "start_time": start_t,
                        "end_time": end_t,
                        "frame_range": frame_range,
                        "confidence": 1.0,
                        "source": "manual_review"
                    }
                    
                    text_json_data['pii_detections'].append(new_pii_detection)
                    print(f"[API] Added manual PII detection: {new_pii_detection['id']} - '{new_pii_detection['text']}'")
            
            if captions_muted: 
                # Mark words as PII if they overlap with any muted period
                print(f"[API] Processing {len(captions_muted)} muted periods")
                for segment in audio_json_data.get('segments', []):
                    # Initialize all words in segment to non-PII first
                    for word in segment.get('words', []):
                        word['is_PII'] = 0
                    
                    # Check each word against all muted periods
                    for word in segment.get('words', []):
                        word_start = float(word.get('start_time', 0.0))
                        word_end = float(word.get('end_time', 0.0))
                        
                        # Check if word overlaps with any muted period
                        # Two ranges overlap if: word_start < muted_end AND word_end > muted_start
                        for muted_period in captions_muted:
                            muted_start = float(muted_period.get('start', 0.0))
                            muted_end = float(muted_period.get('end', 0.0))
                            
                            # Check for overlap
                            if word_start <= muted_end and word_end >= muted_start: 
                                word['is_PII'] = 1
                                break  # No need to check other muted periods for this word
                
                # Count words marked as PII
                total_pii_words = sum(
                    sum(1 for word in seg.get('words', []) if word.get('is_PII', 0) == 1)
                    for seg in audio_json_data.get('segments', [])
                )
                print(f"[API] Marked {total_pii_words} words as PII based on muted periods")
            

            # Update metadata
            if 'metadata' in text_json_data:
                text_json_data['metadata']['total_pii_detections'] = len(text_json_data['pii_detections'])
                text_json_data['metadata']['last_updated'] = datetime.now(timezone.utc).isoformat()

            
            print(f"[API] Final PII detections count: {len(text_json_data['pii_detections'])}")
            
            # Write the updated JSON back to local file
            with open(local_text_json_path, 'w', encoding='utf-8') as f:
                json.dump(text_json_data, f, indent=2, ensure_ascii=False)
            
            with open(local_audio_json_path, 'w', encoding='utf-8') as f:
                json.dump(audio_json_data, f, indent=2, ensure_ascii=False)
            
            # Upload the updated JSON back to S3
            push_AWS_S3(text_s3_output_path, local_text_json_path, region="ap-southeast-2")
            print(f"[API] Updated text PII JSON uploaded to S3: {text_s3_output_path}")
            push_AWS_S3(audio_s3_output_path, local_audio_json_path, region="ap-southeast-2")
            print(f"[API] Updated audio PII JSON uploaded to S3: {audio_s3_output_path}")
            
            # Clean up local file
            os.remove(local_text_json_path)
            print(f"[API] Local text JSON file cleaned up")
            os.remove(local_audio_json_path)
            print(f"[API] Local audio JSON file cleaned up")

        except Exception as e:
            print(f"[API] WARNING: Failed to update text PII JSON: {str(e)}")
            print(f"[API] Continuing with original text JSON file...")
        
        print(f"[API] Review data:")
        print(f"  - Deleted IDs: {len(review_data['deleted_ids'])}")
        print(f"  - Manual boxes: {len(review_data['manual_boxes'])}")
        print(f"  - Captions muted: {len(review_data['captions_muted'])}")
        print(f"  - Message: {review_data['message']}")
        
        # Add review version
        version = task_manager.add_review(task_id, review_data)
        
        print(f"[API] Review version {version} created successfully")
        print(f"[API] ===== END COMMIT REVIEW =====")
        
        return ReviewCommitResponse(ok=True, review_version=version)
    
    @staticmethod
    async def finalize(request: Dict[str, Any]) -> FinalizeResponse:
        """Trigger final processing"""
        print(f"[API] ===== FINALIZE =====")
        print(f"[API] Request: {request}")
        
        # Extract task_id and use_version from dictionary
        task_id = request.get('task_id')
        use_version = request.get('use_version')
        
        if not task_id:
            print(f"[API] ERROR: Missing task_id in request")
            raise HTTPException(status_code=400, detail="task_id is required")
        
        if use_version is None:
            print(f"[API] ERROR: Missing use_version in request")
            raise HTTPException(status_code=400, detail="use_version is required")
        
        print(f"[API] Task ID: {task_id}")
        print(f"[API] Use Version: {use_version}")
        
        task = task_manager.get_task(task_id)
        if not task:
            print(f"[API] ERROR: Task {task_id} not found")
            raise HTTPException(status_code=404, detail="Task not found")
        
        print(f"[API] Task found: {task['task_id']}")
        print(f"[API] Task status: {task['status']}")
        
        if task["status"] != TaskStatus.DONE:
            print(f"[API] ERROR: Task {task_id} not completed (status: {task['status']})")
            raise HTTPException(status_code=400, detail="Task not completed")
        
        # Create final task
        final_task_id = f"final_{uuid.uuid4().hex[:8]}"
        task_manager.create_final_task(final_task_id, task_id, use_version)
        
        print(f"[API] Final task created: {final_task_id}")
        
        # Start background final processing
        asyncio.create_task(APIController._finalize_background(final_task_id, task_id, use_version))
        
        print(f"[API] Background processing started")
        print(f"[API] ===== END FINALIZE =====")
        
        return FinalizeResponse(ok=True, final_task_id=final_task_id, status="queued")
    
    @staticmethod
    async def _finalize_background(final_task_id: str, task_id: str, use_version: int) -> None:
        """Background final processing"""
        try:
            print(f"[API] ===== STARTING FINAL BACKGROUND PROCESSING =====")
            print(f"[API] Final task ID: {final_task_id}")
            print(f"[API] Task ID: {task_id}")
            print(f"[API] Use Version: {use_version}")
            
            task_manager.update_final_task(final_task_id, TaskStatus.PROCESSING, 10, "Starting final processing...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            
            # Get task and review data
            task = task_manager.get_task(task_id)
            reviews = task_manager.get_reviews(task_id)
            
            if use_version > len(reviews):
                raise Exception(f"Review version {use_version} not found")
            
            # Apply review changes
            print(f"[API] Step 1/4: Applying review changes...")
            task_manager.update_final_task(final_task_id, TaskStatus.PROCESSING, 20, "Applying review changes...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            
            video_key = task["s3_key"]
            print(f"[API] Video key: {video_key}")
            
            # Extract filename from video_key (remove any path prefix)
            video_filename = os.path.basename(video_key)
            video_basename = os.path.splitext(video_filename)[0]
            print(f"[API] Video filename: {video_filename}")
            print(f"[API] Video basename: {video_basename}")
            
            audio_s3_path = f"aws-uni-output1/audio/{video_filename}/transcript_segments_reviewed.json"
            original_video_S3_path = f"aws-uni-input1/{video_filename}"
            text_s3_path = f"aws-uni-output1/text_framesandjson_output/{video_basename}/{video_basename}_reviewed.json"
            
            print(f"[API] Audio S3 path: {audio_s3_path}")
            print(f"[API] Original video S3 path: {original_video_S3_path}")
            print(f"[API] Text S3 path: {text_s3_path}")

            # Step 2: Video mosaic processing
            print(f"[API] Step 2/4: Processing video mosaic...")
            task_manager.update_final_task(final_task_id, TaskStatus.PROCESSING, 40, "Processing video mosaic...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            video_with_redact = video_mosaic_processor(text_s3_path, original_video_S3_path, region="ap-southeast-2")
            print(f"[API] ✅ Video mosaic completed: {video_with_redact}")
            
            # Step 3: Audio dedaction processing
            print(f"[API] Step 3/4: Processing audio redaction...")
            task_manager.update_final_task(final_task_id, TaskStatus.PROCESSING, 60, "Processing audio redaction...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            audio_with_redact = process_dedaction(audio_s3_path, original_video_S3_path, region="ap-southeast-2")
            print(f"[API] ✅ Audio redaction completed: {audio_with_redact}")
            
            # Step 4: Mux video and audio
            print(f"[API] Step 4/4: Muxing video and audio...")
            task_manager.update_final_task(final_task_id, TaskStatus.PROCESSING, 80, "Muxing video and audio...")
            await asyncio.sleep(0.1)  # Small delay to ensure status update is visible
            final_result_path = mux_text_audio(video_with_redact, audio_with_redact)
            print(f"[API] ✅ Muxing completed: {final_result_path}")
            #redacted_video_url = pull_AWS_S3(final_result_path, region="ap-southeast-2")
            
            task_manager.update_final_task(final_task_id, TaskStatus.DONE, 100, "Final processing completed", final_result_path)
            print(f"[API] ===== FINAL PROCESSING COMPLETED SUCCESSFULLY =====")
            
            # Add to history
            task_manager.add_to_history({
                "id": task_id,
                "status": "ready",
                "filename": task["filename"],
                "redacted_video_url": final_result_path,
                "created_at": task["created_at"]
            })
            
        except Exception as e:
            print(f"[API] Final processing failed: {str(e)}")
            task_manager.update_final_task(final_task_id, TaskStatus.FAILED, 0, f"Final processing failed: {str(e)}")
    
    @staticmethod
    def get_final_result(final_task_id: str) -> FinalResultResponse:
        """Get final processing result"""
        print(f"[API] Getting final result for: {final_task_id}")
        final_task = task_manager.get_final_task(final_task_id)
        if not final_task:
            print(f"[API] Final task not found: {final_task_id}")
            raise HTTPException(status_code=404, detail="Final task not found")
        
        # Get status value - ensure it's a TaskStatus enum
        status_value = final_task["status"]
        print(f"[API] Raw final status from task: {status_value} (type: {type(status_value)})")
        
        # Convert to TaskStatus enum if it's a string
        if isinstance(status_value, str):
            try:
                status_enum = TaskStatus(status_value)
                print(f"[API] Converted string '{status_value}' to TaskStatus enum: {status_enum}")
            except ValueError:
                print(f"[API] ERROR: Invalid status string '{status_value}', using QUEUED")
                status_enum = TaskStatus.QUEUED
        elif isinstance(status_value, TaskStatus):
            status_enum = status_value
            print(f"[API] Status is already TaskStatus enum: {status_enum}")
        else:
            print(f"[API] ERROR: Unknown status type {type(status_value)}, using QUEUED")
            status_enum = TaskStatus.QUEUED
        
        progress_value = final_task.get("progress")
        message_value = final_task.get("message")
        final_video_url_value = final_task.get("final_video_url")
        
        print(f"[API] Final task {final_task_id}: status={status_enum}, progress={progress_value}%, message={message_value}")
        
        response = FinalResultResponse(
            status=status_enum,
            progress=progress_value,
            final_video_url=final_video_url_value,
            message=message_value
        )
        print(f"[API] FinalResultResponse created: status={response.status}, progress={response.progress}, message={response.message}")
        return response
    
    @staticmethod
    def list_commits(task_id: str) -> CommitsListResponse:
        """List review versions for a task"""
        reviews = task_manager.get_reviews(task_id)
        
        items = []
        for review in reviews:
            stats = CommitStats(
                deleted=len(review.get("deleted_ids", [])),
                manual=len(review.get("manual_boxes", [])),
                mute_ranges=len(review.get("captions_muted", []))
            )
            
            items.append(CommitItem(
                review_version=review["review_version"],
                created_at=review["created_at"],
                message=review.get("message"),
                stats=stats
            ))
        
        return CommitsListResponse(items=items)

    @staticmethod
    def list_history(limit: int = 100) -> HistoryResponse:
        """List processing history"""
        history = task_manager.get_history(limit)
        print(f"[API] History: {history}")

        items = []
        for item in history:
            # Handle redacted_video_url: might be a list due to data corruption
            redacted_url = item.get("redacted_video_url")
            if isinstance(redacted_url, list):
                # Extract first element if it's a list, or None if empty
                redacted_url = redacted_url[0] if redacted_url else None
            elif not isinstance(redacted_url, str) and redacted_url is not None:
                # Convert to string if it's some other type, or None if invalid
                redacted_url = str(redacted_url) if redacted_url else None
            
            items.append(HistoryItem(
                id=item["id"],
                status=item["status"],
                filename=item["filename"],
                redacted_video_url=redacted_url,
                created_at=item["created_at"]
            ))

        return HistoryResponse(items=items)
    
    def get_download_url(task_id: str) -> str:
        """Get download URL for a task"""
        s3 = boto3.client("s3", region_name="ap-southeast-2")
        history_s3_path = "aws-uni-output2/history/db.json"
        local_history_path = pull_AWS_S3(history_s3_path, region="ap-southeast-2")
        with open(local_history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        task = next((t for t in history_data if t["id"] == task_id), None)

        video_filename = task["filename"]
        video_basename = os.path.splitext(video_filename)[0]
        out_bucket = "aws-uni-output2"
        out_video_key = f"{video_basename}_redaction.mp4"
        presigned_download_url = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": out_bucket,
                "Key": out_video_key,
                "ResponseContentType": "video/mp4",
                "ResponseContentDisposition": "attachment",
                #"filename": f"{video_basename}_redaction.mp4"
            },
            ExpiresIn=3600  # valid for 1 hour
        )
        return RedirectResponse(presigned_download_url)

    @staticmethod
    def get_play_url(task_id: str) -> str:
        """Get play URL for a task"""
        s3 = boto3.client("s3", region_name="ap-southeast-2")
        history_s3_path = "aws-uni-output2/history/db.json"
        local_history_path = pull_AWS_S3(history_s3_path, region="ap-southeast-2")
        with open(local_history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        task = next((t for t in history_data if t["id"] == task_id), None)

        video_filename = task["filename"]
        video_basename = os.path.splitext(video_filename)[0]
        out_bucket = "aws-uni-output2"
        out_video_key = f"{video_basename}_redaction.mp4"
        presigned_download_url = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": out_bucket,
                "Key": out_video_key,
                "ResponseContentType": "video/mp4",
            },
            ExpiresIn=3600  # valid for 1 hour
        )
        return RedirectResponse(presigned_download_url)

    @staticmethod
    def get_annotations(task_id: str) -> List[Dict[str, Any]]:
        """
        Get all manual annotations for a task
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of manual annotations
        """
        print(f"[API] ===== GET ANNOTATIONS =====")
        print(f"[API] Task ID: {task_id}")
        
        # Validate task exists
        task = task_manager.get_task(task_id)
        if not task:
            print(f"[API] ERROR: Task {task_id} not found")
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        print(f"[API] Task found: {task['task_id']}")
        print(f"[API] Task status: {task['status']}")
        
        # Get annotations from TaskManager
        annotations = task_manager.get_annotations(task_id)
        
        print(f"[API] Found {len(annotations)} annotations for task {task_id}")
        
        # Test output: Show annotation details
        for i, annotation in enumerate(annotations):
            print(f"[API] Annotation [{i}]:")
            print(f"  - ID: {annotation.get('id', 'N/A')}")
            print(f"  - Label: {annotation.get('label', 'N/A')}")
            print(f"  - Start Time: {annotation.get('start', 'N/A')}")
            print(f"  - End Time: {annotation.get('end', 'N/A')}")
            print(f"  - Source: {annotation.get('source', 'N/A')}")
            print(f"  - Created At: {annotation.get('created_at', 'N/A')}")
            if 'norm_box' in annotation:
                box = annotation['norm_box']
                print(f"  - Box: ({box.get('x1', 0):.3f}, {box.get('y1', 0):.3f}) to ({box.get('x2', 0):.3f}, {box.get('y2', 0):.3f})")
            if 'pixel_box' in annotation:
                pixel_box = annotation['pixel_box']
                print(f"  - Pixel Box: ({pixel_box.get('x', 0)}, {pixel_box.get('y', 0)}) w={pixel_box.get('w', 0)} h={pixel_box.get('h', 0)}")
        
        print(f"[API] ===== END GET ANNOTATIONS =====")
        
        # Return annotations directly as array (matches client.ts expectation)
        return annotations
    
    @staticmethod
    def post_annotations(task_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save manual annotations for a task
        
        Args:
            task_id: Task identifier
            request: Annotation request containing:
                - target_space: 'processed' or 'original'
                - ref_width: Reference width for coordinate conversion
                - ref_height: Reference height for coordinate conversion
                - annotations: List of annotation objects
                
        Returns:
            Success response
        """
        print(f"[API] ===== POST ANNOTATIONS =====")
        print(f"[API] Task ID: {task_id}")
        print(f"[API] Request: {request}")
        
        # Validate task exists
        task = task_manager.get_task(task_id)
        if not task:
            print(f"[API] ERROR: Task {task_id} not found")
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        print(f"[API] Task found: {task['task_id']}")
        print(f"[API] Task status: {task['status']}")
        
        # Extract request parameters
        target_space = request.get('target_space', 'original')
        ref_width = request.get('ref_width', 1280)
        ref_height = request.get('ref_height', 720)
        annotations = request.get('annotations', [])
        
        print(f"[API] Target space: {target_space}")
        print(f"[API] Reference dimensions: {ref_width}x{ref_height}")
        print(f"[API] Number of annotations: {len(annotations)}")
        
        # Process each annotation
        processed_annotations = []
        for i, annotation in enumerate(annotations):
            print(f"[API] Processing annotation [{i}]:")
            print(f"  - Raw annotation: {annotation}")
            
            # Validate required fields
            required_fields = ['id', 'label', 'pixel_box', 'start', 'end']
            for field in required_fields:
                if field not in annotation:
                    print(f"[API] ERROR: Missing required field '{field}' in annotation {i}")
                    raise HTTPException(status_code=400, detail=f"Missing required field '{field}' in annotation {i}")
            
            # Convert pixel coordinates to normalized coordinates
            pixel_box = annotation['pixel_box']
            x = pixel_box['x']
            y = pixel_box['y']
            w = pixel_box['w']
            h = pixel_box['h']
            
            # Calculate normalized coordinates
            norm_box = {
                'x1': x / ref_width,
                'y1': y / ref_height,
                'x2': (x + w) / ref_width,
                'y2': (y + h) / ref_height
            }
            
            # Ensure coordinates are within valid range [0, 1]
            norm_box = {
                'x1': max(0, min(1, norm_box['x1'])),
                'y1': max(0, min(1, norm_box['y1'])),
                'x2': max(0, min(1, norm_box['x2'])),
                'y2': max(0, min(1, norm_box['y2']))
            }
            
            # Create processed annotation
            processed_annotation = {
                'id': annotation['id'],
                'label': annotation['label'],
                'start': float(annotation['start']),
                'end': float(annotation['end']),
                'source': 'manual_review',
                'norm_box': norm_box,
                'pixel_box': pixel_box,  # Keep original pixel box for reference
                'target_space': target_space,
                'ref_width': ref_width,
                'ref_height': ref_height
            }
            
            processed_annotations.append(processed_annotation)
            
            print(f"  - Processed annotation:")
            print(f"    - ID: {processed_annotation['id']}")
            print(f"    - Label: {processed_annotation['label']}")
            print(f"    - Time: {processed_annotation['start']:.2f}s - {processed_annotation['end']:.2f}s")
            print(f"    - Pixel Box: ({x}, {y}) w={w} h={h}")
            print(f"    - Normalized Box: ({norm_box['x1']:.3f}, {norm_box['y1']:.3f}) to ({norm_box['x2']:.3f}, {norm_box['y2']:.3f})")
        
        # Clear existing annotations and add new ones
        print(f"[API] Clearing existing annotations for task {task_id}")
        task_manager.clear_annotations(task_id)
        
        # Add each processed annotation
        for annotation in processed_annotations:
            task_manager.add_annotation(task_id, annotation)
        
        print(f"[API] Successfully saved {len(processed_annotations)} annotations for task {task_id}")
        print(f"[API] ===== END POST ANNOTATIONS =====")
        
        # Return format that matches client.ts expectation: { ok: boolean }
        return {
            "ok": True
        }
