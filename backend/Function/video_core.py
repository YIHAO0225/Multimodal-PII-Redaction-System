# backend/Function/video_core.py - 50 THREADS + ORIGINAL QUALITY
import boto3
import cv2
import os
import json
import tempfile
import shutil
import time
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# Global parameters
THRESHOLD = 0.70
MAX_DURATION_SEC = 6
MIN_INTERVAL_FRAMES = 25
MAX_DEPTH = 8
S3_BUCKET = "aws-uni-input1"
MAX_UPLOAD_WORKERS = 50


def textract_ocr_image(bucket, key, region):
    """Execute Textract OCR on a single frame"""
    textract_client = boto3.client('textract', region_name=region)
    
    for attempt in range(4):
        try:
            response = textract_client.detect_document_text(
                Document={'S3Object': {'Bucket': bucket, 'Name': key}}
            )
            text = " ".join([b['Text'] for b in response.get('Blocks', []) if b['BlockType'] == 'LINE'])
            
            if not text.strip():
                print(f"[OCR_DEBUG] Frame {key}: NO TEXT DETECTED")
            else:
                print(f"[OCR_DEBUG] Frame {key}: '{text[:50]}...'")
            
            return text
        except ClientError as e:
            if e.response["Error"]["Code"] == "ProvisionedThroughputExceededException":
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"[OCR] Failed for {key}: {e}")
                return ""
    print(f"[OCR] Failed after 4 retries: {key}")
    return ""


def extract_frames(video_path, output_dir):
    """Extract all frames from video - ORIGINAL QUALITY"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []

    os.makedirs(output_dir, exist_ok=True)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        idx += 1

    cap.release()
    print(f"[EXTRACT] Extracted {len(frames)} frames, FPS={fps}, Total={frame_count}, Size={width}x{height}")
    return frames, fps, frame_count, width, height


def upload_frame_to_s3(frame_path, s3_key, bucket, region):
    """Upload a single frame to S3 with retry"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            s3_client = boto3.client("s3", region_name=region)
            s3_client.upload_file(frame_path, bucket, s3_key)
            return s3_key, True, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
                continue
            return s3_key, False, str(e)


def upload_frames_parallel(frames, s3_output_prefix, bucket, region, max_workers=50):
    """Upload frames to S3 in parallel using 50 threads"""
    print(f"[UPLOAD] Starting parallel upload with {max_workers} workers...")
    
    frame_s3_keys = [None] * len(frames)
    upload_count = 0
    failed_count = 0
    lock = Lock()
    
    tasks = []
    for idx, frame in enumerate(frames):
        fname = os.path.basename(frame)
        s3_key = f"{s3_output_prefix}/{fname}"
        tasks.append((idx, frame, s3_key))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(upload_frame_to_s3, frame, s3_key, bucket, region): (idx, s3_key)
            for idx, frame, s3_key in tasks
        }
        
        for future in as_completed(future_to_task):
            idx, expected_key = future_to_task[future]
            try:
                s3_key, success, error = future.result()
                
                with lock:
                    if success:
                        frame_s3_keys[idx] = s3_key
                        upload_count += 1
                        
                        if upload_count % 100 == 0:
                            print(f"[UPLOAD] Uploaded {upload_count}/{len(frames)} frames")
                    else:
                        failed_count += 1
                        print(f"[UPLOAD] Failed to upload {s3_key}: {error}")
                        
            except Exception as e:
                with lock:
                    failed_count += 1
                    print(f"[UPLOAD] Exception uploading frame {idx}: {e}")
    
    print(f"[UPLOAD] Completed: {upload_count} success, {failed_count} failed")
    
    if failed_count > 0:
        raise Exception(f"Failed to upload {failed_count} frames")
    
    return frame_s3_keys


def recursive_split_on_demand(frame_s3_keys, start_idx, end_idx, depth, fps, 
                               ocr_cache, embedder, bucket, region):
    """Recursively split video frames with on-demand OCR"""
    
    if depth > MAX_DEPTH:
        return [(start_idx, end_idx)]

    if (end_idx - start_idx) <= MIN_INTERVAL_FRAMES:
        return [(start_idx, end_idx)]
    
    mid_idx = (start_idx + end_idx) // 2
    if mid_idx == start_idx or mid_idx == end_idx:
        return [(start_idx, end_idx)]

    segment_duration = (end_idx - start_idx) / fps

    if segment_duration > MAX_DURATION_SEC and depth < MAX_DEPTH:
        print(f"[FORCED_SPLIT] Frames ({start_idx}, {end_idx}): duration={segment_duration:.1f}s > {MAX_DURATION_SEC}s, depth={depth}")
        
        left = recursive_split_on_demand(
            frame_s3_keys, start_idx, mid_idx, depth + 1, 
            fps, ocr_cache, embedder, bucket, region
        )
        right = recursive_split_on_demand(
            frame_s3_keys, mid_idx, end_idx, depth + 1, 
            fps, ocr_cache, embedder, bucket, region
        )
        return left + right

    def get_text(idx):
        if idx not in ocr_cache:
            s3_key = frame_s3_keys[idx]
            ocr_cache[idx] = textract_ocr_image(bucket, s3_key, region)
            if len(ocr_cache) % 10 == 0:
                print(f"[OCR] On-demand OCR progress: {len(ocr_cache)} frames processed")
        return ocr_cache[idx]

    text_start = get_text(start_idx)
    text_mid = get_text(mid_idx)
    text_end = get_text(end_idx)

    if not text_start.strip():
        text_start = f"[EMPTY_FRAME_{start_idx}]"
    if not text_mid.strip():
        text_mid = f"[EMPTY_FRAME_{mid_idx}]"
    if not text_end.strip():
        text_end = f"[EMPTY_FRAME_{end_idx}]"

    emb_start = embedder.encode([text_start])
    emb_mid = embedder.encode([text_mid])
    emb_end = embedder.encode([text_end])
    
    sim_left = cosine_similarity(emb_start, emb_mid)[0][0]
    sim_right = cosine_similarity(emb_mid, emb_end)[0][0]
    
    print(f"[SPLIT_DEBUG] Frames ({start_idx}, {mid_idx}, {end_idx}): "
          f"sim_left={sim_left:.3f}, sim_right={sim_right:.3f}, "
          f"duration={segment_duration:.1f}s, depth={depth}")

    if sim_left < THRESHOLD and (mid_idx - start_idx) >= MIN_INTERVAL_FRAMES:
        print(f"[SPLIT_DEBUG]   └─ Left dissimilar, split")
        left_segments = recursive_split_on_demand(
            frame_s3_keys, start_idx, mid_idx, depth + 1, 
            fps, ocr_cache, embedder, bucket, region
        )
    else:
        left_segments = [(start_idx, mid_idx)]
    
    if sim_right < THRESHOLD and (end_idx - mid_idx) >= MIN_INTERVAL_FRAMES:
        print(f"[SPLIT_DEBUG]   └─ Right dissimilar, split")
        right_segments = recursive_split_on_demand(
            frame_s3_keys, mid_idx, end_idx, depth + 1, 
            fps, ocr_cache, embedder, bucket, region
        )
    else:
        right_segments = [(mid_idx, end_idx)]
    
    return left_segments + right_segments


def process_video(video_path, s3_output_prefix, region="ap-southeast-2"):
    """Main processing pipeline with 50-thread parallel upload"""
    
    print("=" * 80)
    print(f"[PROCESS_VIDEO] Starting video processing (50 Threads + Original Quality)")
    print(f"  Input: {video_path}")
    print(f"  S3 Output: s3://{S3_BUCKET}/{s3_output_prefix}/")
    print(f"  Params: THRESHOLD={THRESHOLD}, MAX_DURATION={MAX_DURATION_SEC}s, MAX_DEPTH={MAX_DEPTH}")
    print(f"  Upload Workers: {MAX_UPLOAD_WORKERS}")
    print("=" * 80)

    process_start = time.time()

    s3 = boto3.client("s3", region_name=region)
    embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Step 1: Extract frames (original quality)
    extract_start = time.time()
    tmp_dir = tempfile.mkdtemp(prefix="frames_")
    frames, fps, frame_count, width, height = extract_frames(video_path, tmp_dir)
    extract_time = time.time() - extract_start
    print(f"[TIMING] Frame extraction: {extract_time:.3f}s")

    video_info = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": frame_count,
        "duration": (frame_count / fps) if fps and fps > 0 else 0.0,
    }
    print(f"[PROCESS_VIDEO] Video info: {video_info}")

    # Step 2: Parallel upload with 50 threads
    upload_start = time.time()
    frame_s3_keys = upload_frames_parallel(
        frames, s3_output_prefix, S3_BUCKET, region, max_workers=MAX_UPLOAD_WORKERS
    )
    upload_time = time.time() - upload_start
    
    print(f"[TIMING] Upload {len(frames)} frames to S3: {upload_time:.3f}s ({upload_time/len(frames)*1000:.1f}ms per frame)")
    print(f"[SPEEDUP] vs original (131s): {130.970/upload_time:.1f}x faster!")
    print(f"[SPEEDUP] vs 20-thread (68s): {67.878/upload_time:.1f}x faster!")

    # Step 3: Scene detection
    print(f"[SPLIT] Starting on-demand OCR + recursive split")
    
    split_start = time.time()
    ocr_cache = {}
    
    segments = recursive_split_on_demand(
        frame_s3_keys, 0, len(frames) - 1, 0, fps, ocr_cache, embedder, S3_BUCKET, region
    )
    split_time = time.time() - split_start
    
    print(f"[SPLIT] Completed, generated {len(segments)} segments")
    print(f"[TIMING] Scene detection (OCR + Analysis): {split_time:.3f}s")
    print(f"[OCR] Total frames: {len(frames)}")
    print(f"[OCR] Actually OCR'd: {len(ocr_cache)} frames ({len(ocr_cache)/len(frames)*100:.1f}%)")

    # Step 4: Generate and upload JSON
    mapping_json = [{"start_frame": s, "end_frame": e} for s, e in segments]
    frame_timestamps_json = [
        {"frame": frame_idx, "timestamp": round(frame_idx / fps, 4)}
        for frame_idx in range(frame_count)
    ]

    mapping_path = os.path.join(tmp_dir, "mapping.json")
    frame_ts_path = os.path.join(tmp_dir, "frame_timestamps.json")

    with open(mapping_path, "w") as f:
        json.dump(mapping_json, f, indent=2)
    with open(frame_ts_path, "w") as f:
        json.dump(frame_timestamps_json, f, indent=2)

    s3.upload_file(mapping_path, S3_BUCKET, f"{s3_output_prefix}/mapping.json")
    s3.upload_file(frame_ts_path, S3_BUCKET, f"{s3_output_prefix}/frame_timestamps.json")

    print(f"[OUTPUT] Uploaded mapping.json and frame_timestamps.json")

    # Step 5: Cleanup
    try:
        shutil.rmtree(tmp_dir)
        print(f"[CLEANUP] Deleted temp directory: {tmp_dir}")
    except Exception as e:
        print(f"[CLEANUP] Failed to delete temp directory: {e}")

    output_dir = f"s3://{S3_BUCKET}/{s3_output_prefix}/"
    process_time = time.time() - process_start
    
    print("=" * 80)
    print("[TIMING SUMMARY]")
    print(f"  Extract frames:    {extract_time:.3f}s")
    print(f"  Upload frames:     {upload_time:.3f}s  ← 50 Threads")
    print(f"  Scene detection:   {split_time:.3f}s")
    print(f"  TOTAL:             {process_time:.3f}s")
    print(f"  Speedup vs v1 (177s): {177.087/process_time:.1f}x faster!")
    print(f"  Speedup vs v2 (114s): {113.920/process_time:.1f}x faster!")
    print("=" * 80)
    print(f"[PROCESS_VIDEO] Processing completed!")
    print(f"  Output directory: {output_dir}")
    print(f"  Segments: {len(segments)}")
    print("=" * 80)

    return {
        "output_dir": output_dir,
        "video_info": video_info
    }
