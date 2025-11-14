# backend/controllers/main_controller.py
import tempfile
from fastapi import UploadFile
from backend.Function.upload import handle
from backend.Function.audio_detector import audio_transcript



async def run_pipeline(file: UploadFile):
    workdir = tempfile.mkdtemp(prefix="work_")

    up_res = await handle(file, workdir)

    s3_input = up_res["s3_path"]

    audio_PII_path = audio_detector(s3_input, region="ap-southeast-2")

    frame_processing = stage1(s3_input, region="ap-southeast-2")
    
    stageA(frame_processing, region="ap-southeast-2")

    result = pii_text_detector(frame_processing, region="ap-southeast-2")


    return {
        "status": "success",
        "inputs": {"s3_input": s3_input},
        "outputs": {
        "audio_output_dir": audio_PII_path,
        "video_output_dir": result,
        "original_video_S3_path" : s3_input
        }
    }

async def redaction(text_s3_path: str, audio_s3_path: str, original_video_S3_path: str):

    video_with_redact = video_mosaic_processor(text_s3_path, original_video_S3_path,region="ap-southeast-2")

    audio_with_redact = process_dedaction(audio_s3_path,original_video_S3_path,region="ap-southeast-2")

    final_result_path = mux_text_audio(video_with_redact,audio_with_redact)


    return {
        "status": "success",
        "outputs": {
        "audio_output_dir": final_result_path
    }
}
