# backend/Function/upload.py
from fastapi import UploadFile
import shutil, os
from .s3_tool import push_AWS_S3

async def handle(file: UploadFile, workdir: str) -> dict:

    local_path = os.path.join(workdir, file.filename)
  
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    s3_path = f"aws-uni-input1/{file.filename}"
    s3_url = push_AWS_S3(s3_path, local_path)

    return {
        "filename": file.filename,
        "local_path": local_path, 
        "s3_path": s3_path,
        "s3_url": s3_url
    }
