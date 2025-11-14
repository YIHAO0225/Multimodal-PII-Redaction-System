"""
Video Mosaic Processing Module - File Path Version
- Receives specific file paths instead of folder paths
- Fixed face detection extraction logic
- Added detailed debugging information and runtime statistics
"""
import os
import json
import boto3
import cv2
import numpy as np
from typing import List, Dict
import tempfile
import time
from datetime import timedelta


def video_mosaic_processor(
    json_s3_path: str,
    video_s3_path: str,
    region: str = "ap-southeast-2",
    mosaic_size: int = 40
) -> dict:
    """
    Video mosaic processing main function

    Features:
    1. Read specified JSON detection result file from S3
    2. Read specified video file from S3
    3. Add fixed-style mosaics based on timestamps and coordinates in JSON
    4. Upload processed video to S3

    Args:
        json_s3_path: Full S3 path of JSON file
            Example: "aws-uni-output1/text_framesandjson_output/1min_test.json"
            Or: "s3://aws-uni-output1/text_framesandjson_output/1min_test.json"

        video_s3_path: Full S3 path of video file
            Example: "aws-uni-input1/1min_test.mp4"
            Or: "s3://aws-uni-input1/1min_test.mp4"

        region: AWS region, default ap-southeast-2
        mosaic_size: Mosaic block size (pixels), default 15

    Returns:
        {
            "status": "success",
            "output_path": "s3://bucket/processed_videos/video_basename/video_name.mp4",
            "video_name": "1min_test.mp4",
            "processed_detections": 15,
            "total_time_seconds": 49.40
        }
    """

    # Record total start time
    total_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"🎬 Video Mosaic Processor Started (File Path Mode)")
    print(f"{'='*70}")
    print(f"📍 Region: {region}")
    print(f"📄 JSON File: {json_s3_path}")
    print(f"🎥 Video File: {video_s3_path}")
    print(f"🔲 Mosaic Size: {mosaic_size}px")
    print(f"⏰ Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    temp_dir = None

    try:
        # ========== Initialize AWS Client ==========
        session = boto3.Session(region_name=region)
        s3 = session.client("s3")

        # ========== Parse JSON Path ==========
        json_bucket, json_key = _parse_s3_path(json_s3_path)
        json_filename = os.path.basename(json_key)
        json_basename = os.path.splitext(json_filename)[0]

        # ========== Parse Video Path ==========
        video_bucket, video_key = _parse_s3_path(video_s3_path)
        video_filename = os.path.basename(video_key)
        video_basename = os.path.splitext(video_filename)[0]

        # Output configuration
        output_bucket = os.environ.get('OUTPUT_BUCKET', 'aws-uni-input2')
        output_prefix = os.environ.get('OUTPUT_PREFIX', 'text_processed_videos/')

        print(f"\n📁 Parsed:")
        print(f"   JSON: s3://{json_bucket}/{json_key}")
        print(f"   Video: s3://{video_bucket}/{video_key}")
        print(f"   Output: s3://{output_bucket}/{output_prefix}")

        # ========== Step 1: Read JSON File ==========
        step1_start = time.time()
        print(f"\n📥 Step 1: Reading JSON file...")
        print(f"   📄 Filename: {json_filename}")

        # Verify JSON file exists
        try:
            s3.head_object(Bucket=json_bucket, Key=json_key)
        except:
            return {"status": "error", "message": f"JSON file does not exist: s3://{json_bucket}/{json_key}"}

        json_data = _download_and_read_json(s3, json_bucket, json_key)
        if not json_data:
            return {"status": "error", "message": "Failed to read JSON file"}

        detections = _extract_detections_from_json(json_data)
        if not detections:
            return {"status": "error", "message": "No sensitive information found in JSON"}

        print(f"   ✅ Found {len(detections)} sensitive information items")
        step1_time = time.time() - step1_start
        print(f"   ⏱️  Duration: {step1_time:.2f}s")

        # ========== Step 2: Verify Video File ==========
        step2_start = time.time()
        print(f"\n🎥 Step 2: Verifying video file...")
        print(f"   🎬 Filename: {video_filename}")

        # Verify video file exists
        try:
            video_head = s3.head_object(Bucket=video_bucket, Key=video_key)
            video_size_mb = video_head['ContentLength'] / (1024 * 1024)
            print(f"   ✅ Video file exists ({video_size_mb:.2f} MB)")
        except:
            return {"status": "error", "message": f"Video file does not exist: s3://{video_bucket}/{video_key}"}

        step2_time = time.time() - step2_start
        print(f"   ⏱️  Duration: {step2_time:.2f}s")

        # ========== Use Memory Filesystem (if available) ==========
        if os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK):
            temp_dir = tempfile.mkdtemp(dir='/dev/shm')
            print(f"\n💾 Using memory filesystem")
        else:
            temp_dir = tempfile.mkdtemp()
            print(f"\n📁 Using temporary directory")

        # ========== Step 3: Process Video ==========
        step3_start = time.time()
        print(f"\n🔲 Step 3: Processing video and adding mosaics...")

        local_video_path = os.path.join(temp_dir, video_filename)
        _download_file_from_s3(s3, video_bucket, video_key, local_video_path)
        print(f"   ✅ Video downloaded")

        video_info = _get_video_info(local_video_path)
        if not video_info:
            return {"status": "error", "message": "Unable to read video information"}

        print(f"   📊 {video_info['width']}x{video_info['height']}, "
              f"{video_info['duration']:.2f}s, FPS={video_info['fps']:.2f}")

        output_video_path = os.path.join(temp_dir, f"processed_{video_filename}")

        processed_count = _process_video_with_mosaic(
            local_video_path, output_video_path, detections, video_info, mosaic_size
        )

        print(f"   ✅ Mosaic processing completed: {processed_count} regions")
        step3_time = time.time() - step3_start
        print(f"   ⏱️  Duration: {step3_time:.2f}s")

        # ========== Step 4: Upload Processed Video ==========
        step4_start = time.time()
        print(f"\n📤 Step 4: Uploading processed video...")

        # Use video basename as output folder name
        output_folder_prefix = f"{output_prefix}{video_basename}/"
        output_key = output_folder_prefix + video_filename

        _upload_file_to_s3(s3, output_video_path, output_bucket, output_key)

        # Return full file path instead of folder path
        full_output_path = f"s3://{output_bucket}/{output_key}"
        print(f"   ✅ Uploaded: {full_output_path}")
        step4_time = time.time() - step4_start
        print(f"   ⏱️  Duration: {step4_time:.2f}s")

        # ========== Step 5: Clean Up Temporary Files ==========
        step5_start = time.time()
        print(f"\n🧹 Step 5: Cleaning up temporary files...")
        _cleanup_temp_files(temp_dir)
        step5_time = time.time() - step5_start
        print(f"   ⏱️  Duration: {step5_time:.2f}s")

        # Calculate total time
        total_time = time.time() - total_start_time
        total_td = timedelta(seconds=int(total_time))

        print(f"\n🎉 Processing completed!")
        print(f"   📦 Output file: {full_output_path}")
        print(f"   🎥 Video name: {video_filename}")
        print(f"\n{'='*70}")
        print(f"⏱️  Total runtime: {total_td} ({total_time:.2f}s)")
        print(f"{'='*70}")
        print(f"   📥 Step 1 (Read JSON): {step1_time:.2f}s")
        print(f"   🎥 Step 2 (Verify Video): {step2_time:.2f}s")
        print(f"   🔲 Step 3 (Process Video): {step3_time:.2f}s ({step3_time/total_time*100:.1f}%)")
        print(f"   📤 Step 4 (Upload Video): {step4_time:.2f}s")
        print(f"   🧹 Step 5 (Clean Files): {step5_time:.2f}s")
        print(f"{'='*70}")

        return {
            "status": "success",
            "output_path": full_output_path,
        }

    except Exception as e:
        import traceback
        total_time = time.time() - total_start_time
        print(f"\n❌ Processing error: {e}")
        print(f"⏱️  Runtime (until error): {total_time:.2f}s")
        traceback.print_exc()

        if temp_dir:
            _cleanup_temp_files(temp_dir)

        return {
            "status": "error",
            "message": str(e),
            "total_time_seconds": round(total_time, 2)
        }


# ========== Core Processing Functions ==========

def _process_video_with_mosaic(input_path, output_path, detections, video_info, mosaic_size):
    """
    Process video and add mosaics - FIXED coordinate extraction
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"   ❌ Cannot open video: {input_path}")
        return 0

    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    time_based_detections = _build_time_based_index(detections)

    frame_idx = 0
    processed_count = 0
    last_log_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        regions = _get_regions_for_time(time_based_detections, current_time)

        # Process each region - FIXED coordinate extraction
        for detection in regions:
            pixel_coords = detection.get('pixel_coordinates')
            normalized_coords = detection.get('normalized_coordinates')
            coords = detection.get('coordinates')

            # Try pixel coordinates first
            if pixel_coords and isinstance(pixel_coords, dict):
                if 'x_min' in pixel_coords:
                    x = int(pixel_coords['x_min'])
                    y = int(pixel_coords['y_min'])
                    w = int(pixel_coords['x_max']) - x
                    h = int(pixel_coords['y_max']) - y
                else:
                    x = int(pixel_coords.get('x', 0))
                    y = int(pixel_coords.get('y', 0))
                    w = int(pixel_coords.get('width', 0))
                    h = int(pixel_coords.get('height', 0))

            # Try normalized coordinates
            elif normalized_coords and isinstance(normalized_coords, dict):
                if 'x_min' in normalized_coords:
                    x = int(normalized_coords['x_min'] * width)
                    y = int(normalized_coords['y_min'] * height)
                    w = int((normalized_coords['x_max'] - normalized_coords['x_min']) * width)
                    h = int((normalized_coords['y_max'] - normalized_coords['y_min']) * height)
                else:
                    x = int(normalized_coords.get('x', 0) * width)
                    y = int(normalized_coords.get('y', 0) * height)
                    w = int(normalized_coords.get('width', 0) * width)
                    h = int(normalized_coords.get('height', 0) * height)

            # Try generic coordinates
            elif coords and isinstance(coords, dict):
                if 'x_min' in coords:
                    x = int(coords['x_min'])
                    y = int(coords['y_min'])
                    w = int(coords['x_max']) - x
                    h = int(coords['y_max']) - y
                else:
                    x = int(coords.get('Left', 0) * width)
                    y = int(coords.get('Top', 0) * height)
                    w = int(coords.get('Width', 0) * width)
                    h = int(coords.get('Height', 0) * height)
            else:
                continue

            # Clamp coordinates
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(0, min(w, width - x))
            h = max(0, min(h, height - y))

            if w > 0 and h > 0:
                frame = _apply_mosaic_to_region(frame, x, y, w, h, mosaic_size)
                processed_count += 1

        out.write(frame)
        frame_idx += 1

        if time.time() - last_log_time >= 3.0:
            progress = (frame_idx / video_info['total_frames']) * 100
            print(f"      Progress: {progress:.1f}% ({frame_idx}/{video_info['total_frames']} frames)")
            last_log_time = time.time()

    cap.release()
    out.release()

    return processed_count


def _apply_mosaic_to_region(frame, x, y, w, h, mosaic_size):
    """Apply mosaic effect to specified region"""
    if w <= 0 or h <= 0:
        return frame

    # Extract region
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return frame

    # Downsample
    small = cv2.resize(roi, (max(1, w // mosaic_size), max(1, h // mosaic_size)),
                       interpolation=cv2.INTER_LINEAR)

    # Upsample
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Replace region
    frame[y:y+h, x:x+w] = mosaic

    return frame


def _build_time_based_index(detections):
    """
    Build time-based index for fast lookup of regions to mosaic at any time point

    Returns:
        Dictionary where key is (start_time, end_time), value is list of detections
    """
    time_based = {}

    for detection in detections:
        # Extract time information
        if 'start_time' in detection and 'end_time' in detection:
            start_time = detection['start_time']
            end_time = detection['end_time']
        elif 'timestamp' in detection:
            start_time = detection['timestamp']
            end_time = detection['timestamp']
        elif 'frame_range' in detection:
            frame_range = detection['frame_range']
            if isinstance(frame_range, dict):
                start_time = frame_range.get('start_time', 0)
                end_time = frame_range.get('end_time', float('inf'))
            else:
                start_time = 0
                end_time = float('inf')
        else:
            # If no time information, assume entire video
            if 'start' in detection and 'end' in detection:
                start_time = detection['start']
                end_time = detection['end']
            else:
                start_time = 0
                end_time = float('inf')

        time_key = (float(start_time), float(end_time))
        if time_key not in time_based:
            time_based[time_key] = []
        time_based[time_key].append(detection)

    return time_based


def _get_regions_for_time(time_based_detections, current_time):
    """Get regions that need mosaic at current time"""
    regions = []
    for (start_time, end_time), detections in time_based_detections.items():
        if start_time <= current_time <= end_time:
            regions.extend(detections)
    return regions


def _extract_detections_from_json(json_data):
    """
    Extract detection information from JSON
    Fully fixed version - correctly handles frames structure in face_detections
    """
    detections = []

    # ========== Process PII Detections ==========
    if isinstance(json_data, list):
        detections = json_data
    elif 'pii_detections' in json_data:
        pii_dets = json_data['pii_detections']
        if isinstance(pii_dets, list):
            detections.extend(pii_dets)

    pii_count = len(detections)
    print(f"   📝 PII Detections: {pii_count} items")

    # ========== Process Face Detections ==========
    face_count = 0
    if 'face_detections' in json_data:
        face_data = json_data['face_detections']

        # Case 1: face_detections is a dictionary containing frames array
        if isinstance(face_data, dict) and 'frames' in face_data:
            print(f"   👤 Processing face_detections.frames array...")
            frames = face_data['frames']

            for frame_detection in frames:
                # frame_detection itself contains face information and boxes
                if 'boxes' in frame_detection and isinstance(frame_detection['boxes'], list):
                    # Create independent detection object for each box
                    for box in frame_detection['boxes']:
                        face_det = {
                            'type': frame_detection.get('type', 'face'),
                            'label': frame_detection.get('label', 'face'),
                            'is_pii': frame_detection.get('is_pii', 1),
                            'confidence': frame_detection.get('confidence', 0.99),
                            'start_time': frame_detection.get('start_time'),
                            'end_time': frame_detection.get('end_time'),
                            'frame_range': frame_detection.get('frame_range')
                        }

                        # Add coordinate information
                        if 'normalized_coordinates' in box:
                            face_det['normalized_coordinates'] = box['normalized_coordinates']
                        if 'pixel_coordinates' in box:
                            face_det['pixel_coordinates'] = box['pixel_coordinates']
                        if 'coordinates' in box:
                            face_det['coordinates'] = box['coordinates']

                        detections.append(face_det)
                        face_count += 1

                        # Print detailed info of first face detection
                        if face_count == 1:
                            print(f"      Example: {face_det.get('start_time'):.2f}s-{face_det.get('end_time'):.2f}s, "
                                  f"coords={'pixel' if 'pixel_coordinates' in face_det else 'normalized' if 'normalized_coordinates' in face_det else 'none'}")
                else:
                    # If no boxes, add frame_detection directly
                    if frame_detection.get('is_pii', 0) == 1:
                        detections.append(frame_detection)
                        face_count += 1

        # Case 2: face_detections is directly a list
        elif isinstance(face_data, list):
            print(f"   👤 Processing face_detections array...")
            for face_detection in face_data:
                if 'boxes' in face_detection and isinstance(face_detection['boxes'], list):
                    for box in face_detection['boxes']:
                        face_det = {
                            'type': face_detection.get('type', 'face'),
                            'label': face_detection.get('label', 'face'),
                            'is_pii': face_detection.get('is_pii', 1),
                            'confidence': face_detection.get('confidence', 0.99),
                            'start_time': face_detection.get('start_time'),
                            'end_time': face_detection.get('end_time'),
                            'frame_range': face_detection.get('frame_range')
                        }

                        if 'normalized_coordinates' in box:
                            face_det['normalized_coordinates'] = box['normalized_coordinates']
                        if 'pixel_coordinates' in box:
                            face_det['pixel_coordinates'] = box['pixel_coordinates']
                        if 'coordinates' in box:
                            face_det['coordinates'] = box['coordinates']

                        detections.append(face_det)
                        face_count += 1
                else:
                    if face_detection.get('is_pii', 0) == 1:
                        detections.append(face_detection)
                        face_count += 1

    print(f"   👤 Face Detections: {face_count} items")

    # Process other detections
    if 'detections' in json_data:
        other_dets = [d for d in json_data['detections'] if d.get('is_pii', 0) == 1]
        detections.extend(other_dets)
        print(f"   📌 Other Detections: {len(other_dets)} items")

    # Keep only sensitive information
    sensitive_detections = [d for d in detections if d.get('is_pii', 1) == 1]

    print(f"   ✅ Total Sensitive Information: {len(sensitive_detections)} items")

    return sensitive_detections


# ========== Helper Functions ==========

def _parse_s3_path(s3_path):
    """
    Parse S3 path and return bucket and key

    Args:
        s3_path: S3 path, supports the following formats:
            - "bucket/path/to/file.ext"
            - "s3://bucket/path/to/file.ext"

    Returns:
        (bucket, key) tuple
    """
    # Remove s3:// prefix
    clean_path = s3_path.replace('s3://', '').strip('/')

    # Split bucket and key
    if '/' in clean_path:
        parts = clean_path.split('/', 1)
        bucket = parts[0]
        key = parts[1]
        return bucket, key
    else:
        # If only bucket name, return empty key
        return clean_path, ''


def _download_and_read_json(s3, bucket, key):
    """Download and read JSON file from S3"""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        print(f"   ❌ Failed to read JSON: {e}")
        return None


def _download_file_from_s3(s3, bucket, key, local_path):
    """Download file from S3"""
    s3.download_file(bucket, key, local_path)


def _upload_file_to_s3(s3, local_path, bucket, key):
    """Upload file to S3"""
    s3.upload_file(local_path, bucket, key)


def _get_video_info(video_path):
    """Get video information"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0

        cap.release()
        return info
    except Exception as e:
        print(f"   ❌ Failed to get video info: {e}")
        return None


def _cleanup_temp_files(temp_dir):
    """Clean up temporary files"""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            if '/dev/shm' in temp_dir:
                print(f"   ✅ Memory files cleaned up")
            else:
                print(f"   ✅ Temporary files cleaned up")
    except Exception as e:
        print(f"   ⚠️  Failed to clean files: {e}")


# ========== Main Entry (for testing) ==========
if __name__ == "__main__":
    # Test case - use specific file paths
    result = video_mosaic_processor(
        json_s3_path="s3://aws-uni-output1/text_framesandjson_output/1min_test/1min_test_.json",  # JSON file path
        video_s3_path="aws-uni-input1/1min_test.mp4",                             # Video file path
        region="ap-southeast-2"
    )

    print(f"\n{'='*70}")
    print(f"📋 Return Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))