"""
API Router for PII Detection and Redaction System
Defines all API endpoints and routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
from .api_controller import APIController

router = APIRouter()

# 1) POST /upload - Upload video and start processing
@router.post("/upload")
async def upload(file: UploadFile = File(...)):

    return await APIController.create_task(file=file)

# 1b) POST /process - Alternative endpoint for frontend compatibility
@router.post("/process")
async def process(file: UploadFile = File(...)):

    return await APIController.create_task(file=file)

# 2) GET /result - Get processing status
@router.get("/result")
def get_result(task_id: str = Query(..., description="Task ID")):

    result = APIController.get_result(task_id)
    print(f"[ROUTER] Returning result: status={result.status}, progress={result.progress}, message={result.message}")
    print(f"[ROUTER] Result status type: {type(result.status)}, value: {result.status}")
    return result

# 3) GET /payload - Get full payload data
@router.get("/payload")
def get_payload(
    task_id: str = Query(..., description="Task ID"),
    use_version: Optional[int] = Query(None, description="Review version to preview")
):

    result = APIController.get_payload(task_id, use_version)
    
    # Test output: Show what's being returned to frontend
    print(f"[ROUTER] ===== RETURNING TO FRONTEND =====")
    print(f"[ROUTER] Task ID: {task_id}")
    print(f"[ROUTER] PII detections count: {len(result.pii_detections)}")
    for i, pii in enumerate(result.pii_detections[:3]):  # Show first 3 PII detections
        print(f"[ROUTER] PII [{i}] - ID: {pii.id}")
        print(f"  - Label: {pii.label}")
        print(f"  - Text: {pii.text}")
        print(f"  - Confidence: {pii.confidence} (type: {type(pii.confidence)})")
        print(f"  - Start: {pii.start} (type: {type(pii.start)})")
        print(f"  - End: {pii.end} (type: {type(pii.end)})")
        print(f"  - Source: {pii.source}")
    print(f"[ROUTER] Face frames count: {len(result.face_detections.frames)}")
    for i, frame in enumerate(result.face_detections.frames[:2]):  # Show first 2 face frames
        print(f"[ROUTER] Face Frame [{i}] - Time: {frame.t}s")
        print(f"  - Boxes: {len(frame.norm_boxes)}")
        for j, box in enumerate(frame.norm_boxes[:1]):  # Show first box of each frame
            print(f"    Box [{j}] - ID: {box.id}")
            print(f"      - Confidence: {box.confidence} (type: {type(box.confidence)})")
            print(f"      - Coords: ({box.x1}, {box.y1}) to ({box.x2}, {box.y2})")
    print(f"[ROUTER] ===== END FRONTEND RETURN =====")
    
    return result

# 3b) GET /annotations - Get manual annotations
@router.get("/annotations")
def get_annotations(task_id: str = Query(..., description="Task ID")):

    return APIController.get_annotations(task_id)

# 3c) POST /annotations - Save manual annotations
@router.post("/annotations")
def post_annotations(request: dict):

    task_id = request.get('task_id')
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    
    return APIController.post_annotations(task_id, request)

# 4) POST /review/commit - Commit review changes
@router.post("/review/commit")
def commit_review(request: dict):

    return APIController.commit_review(request)

# 5) POST /finalize - Trigger final processing
@router.post("/finalize")
async def finalize(request: dict):

    return await APIController.finalize(request)

# 6) GET /final/result - Get finalization status
@router.get("/final/result")
def get_final_result(final_task_id: str = Query(..., description="Final task ID")):

    return APIController.get_final_result(final_task_id)

# 7) GET /history - Get processing history
@router.get("/history")
def get_history():

    return APIController.list_history()

# 8) GET /download - Get download URL
@router.get("/download")
def get_download_url(task_id: str = Query(..., description="Task ID")):

    return APIController.get_download_url(task_id)

# 8b) GET /video/{task_id}/download - Get download URL (alternative path for frontend)
@router.get("/video/{task_id}/download")
def get_download_url_path(task_id: str):

    return APIController.get_download_url(task_id)
    
# 8c) GET /video/{task_id}/play - Get play URL
@router.get("/video/{task_id}/play")
def get_play_url(task_id: str):

    return APIController.get_play_url(task_id)
