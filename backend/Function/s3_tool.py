# backend/Function/s3_simple.py
# backend/Function/s3_tool.py
import os
import boto3
from typing import Optional
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

def _client(region: str = "ap-southeast-2"):
    session = boto3.Session(region_name=region)
    return session.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}))

def push_AWS_S3(s3_path: str, local_path: Optional[str] = None, region: str = "ap-southeast-2") -> str:

    if not s3_path or "/" not in s3_path:
        raise ValueError("S3 Path must be 'bucket/key/...' format")

    bucket, key = s3_path.split("/", 1)

    if local_path is None:
        local_path = os.path.basename(key) 
    local_path = os.path.abspath(local_path)

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local file not exists: {local_path}")

    client = _client(region)
    cfg = TransferConfig(multipart_threshold=8 * 1024 * 1024, max_concurrency=8)
    client.upload_file(local_path, bucket, key, Config=cfg)
    return f"s3://{bucket}/{key}"


def pull_AWS_S3(s3_path: str, region: str = "ap-southeast-2") -> str:
 
    if not s3_path or "/" not in s3_path:
        raise ValueError("S3 Path must be 'bucket/key/...' format")

    bucket, key = s3_path.split("/", 1)
    filename = os.path.basename(key)
    local_path = os.path.abspath(filename)

    client = _client(region)
    cfg = TransferConfig(multipart_threshold=8 * 1024 * 1024, max_concurrency=8)
    client.download_file(bucket, key, local_path, Config=cfg)

    return local_path
