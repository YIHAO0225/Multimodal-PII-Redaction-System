# backend/Function/video_stage1.py
import os
import tempfile
import boto3
import time
from urllib.parse import urlparse
from backend.Function.video_core import process_video


S3_BUCKET = "aws-uni-input1"  # Fixed S3 bucket name


def download_from_s3(s3_uri, local_path, region="ap-southeast-2"):
    
    parsed = urlparse(s3_uri)
    
    # Extract key from different formats
    if parsed.scheme == 's3':
        # Standard S3 URI: s3://bucket/key
        key = parsed.path.lstrip('/')
    else:
        # Direct key path
        key = s3_uri
    
    # Remove bucket name from key if it was included
    if key.startswith(f"{S3_BUCKET}/"):
        key = key[len(S3_BUCKET)+1:]
    
    if not key:
        raise ValueError(f"Cannot extract S3 key from: {s3_uri}")
    
    print(f"[DOWNLOAD] Downloading from bucket '{S3_BUCKET}', key: {key}")
    
    s3 = boto3.client('s3', region_name=region)
    s3.download_file(S3_BUCKET, key, local_path)
    print(f"[DOWNLOAD] Downloaded to: {local_path}")
    return local_path


def stage1(s3_input: str, region: str = "ap-southeast-2"):
    
    print("=" * 80)
    print(f"[STAGE1] Starting video processing")
    print(f"  Input: {s3_input}")
    print(f"  Region: {region}")
    print("=" * 80)
    
    stage1_start = time.time()

    # Parse S3 path to extract key
    parsed = urlparse(s3_input)
    
    if parsed.scheme == 's3':
        # Standard S3 URI: s3://bucket/key
        key = parsed.path.lstrip('/')
    else:
        # Direct key path
        key = s3_input
    
    # Remove bucket name from key if it was included
    if key.startswith(f"{S3_BUCKET}/"):
        key = key[len(S3_BUCKET)+1:]
    
    if not key:
        raise ValueError(f"Cannot extract S3 key from: {s3_input}")
    
    # Generate output prefix based on input filename
    video_name = os.path.splitext(os.path.basename(key))[0]
    s3_output_prefix = video_name  # Directly use video name as folder
    
    print(f"[STAGE1] S3 Bucket: {S3_BUCKET}")
    print(f"[STAGE1] S3 Key: {key}")
    print(f"[STAGE1] Output Prefix: {s3_output_prefix}")

    # Create temporary working directory
    workdir = tempfile.mkdtemp(prefix="stage1_")
    local_video_path = os.path.join(workdir, os.path.basename(key))

    try:
        # Download video from S3 to local
        print(f"[STAGE1] Downloading video...")
        download_from_s3(s3_input, local_video_path, region)
        
        # Call core processing function
        print(f"[STAGE1] Starting video processing pipeline...")
        output_dir = process_video(
            video_path=local_video_path,
            s3_output_prefix=s3_output_prefix,
            region=region
        )
        
        stage1_time = time.time() - stage1_start
        
        print("=" * 80)
        print(f"[TIMING] STAGE1 TOTAL TIME: {stage1_time:.3f}s")
        print("=" * 80)
        print("[STAGE1] Processing completed!")
        print(f"  Output directory: {output_dir}")
        print("=" * 80)

        return output_dir
    
    except Exception as e:
        print("=" * 80)
        print(f"[STAGE1] Processing failed: {e}")
        print("=" * 80)
        raise
    
    finally:
        # Cleanup local video file
        if os.path.exists(local_video_path):
            os.remove(local_video_path)
            print(f"[STAGE1] Deleted local video: {local_video_path}")