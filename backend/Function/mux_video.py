import os
import subprocess
import tempfile
import boto3
from urllib.parse import urlparse


def _parse_s3(s: str):
    if s.startswith("s3://"):
        u = urlparse(s)
        return u.netloc, u.path.lstrip("/")
    parts = s.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 path: {s}")
    return parts[0], parts[1]


def _find_ffmpeg():
    for p in ["/opt/bin/ffmpeg", "/opt/ffmpeg/ffmpeg", "/opt/ffmpeg/bin/ffmpeg", "ffmpeg"]:
        try:
            subprocess.run([p, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return p
        except Exception:
            continue
    raise FileNotFoundError("ffmpeg not found")


def mux_text_audio(video_s3_path, audio_bundle: dict, region: str = "ap-southeast-2") -> str:

    if isinstance(video_s3_path, dict):
        for k in ("output_path", "video_output_dir", "output", "path", "s3_path", "video_path"):
            v = video_s3_path.get(k)
            if isinstance(v, str) and v:
                video_s3_path = v
                break
        else:
            raise TypeError(f"Invalid video_s3_path payload (dict without path): {video_s3_path}")
    elif not isinstance(video_s3_path, str):
        raise TypeError(f"video_s3_path must be str or dict, got {type(video_s3_path)}")

    v_bucket, v_key = _parse_s3(video_s3_path)
    a_bucket, a_key = _parse_s3(audio_bundle["aDedaction_path"])
    s_bucket, s_key = _parse_s3(audio_bundle["sDedaction_path"])

    video_base = os.path.splitext(os.path.basename(v_key))[0]
    out_bucket, out_key = _parse_s3(f"aws-uni-output2/{video_base}_redaction.mp4")

    s3 = boto3.client("s3", region_name=region)
    ffmpeg = _find_ffmpeg()

    with tempfile.TemporaryDirectory() as td:
        local_video = os.path.join(td, "input_video.mp4")
        local_audio = os.path.join(td, "input_audio.mp3")
        local_srt   = os.path.join(td, "input_subs.srt")
        local_out   = os.path.join(td, "output_final.mp4")

        s3.download_file(v_bucket, v_key, local_video)
        s3.download_file(a_bucket, a_key, local_audio)
        s3.download_file(s_bucket, s_key, local_srt)

        cmd = [
            ffmpeg, "-y",
            "-i", local_video,
            "-i", local_audio,
            "-i", local_srt,
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level:v", "4.1",
            "-c:a", "aac", "-b:a", "128k",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
            "-movflags", "+faststart",
            "-shortest",
            local_out
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)

        s3.upload_file(local_out, out_bucket, out_key, ExtraArgs={"ContentType": "video/mp4"})

    out_bucket = "aws-uni-output2"
    out_video_key = f"{video_base}_redaction.mp4"
    presigned_video_url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": out_bucket,
            "Key": out_video_key,
            "ResponseContentType": "video/mp4"
        },
        ExpiresIn=3600  # valid for 1 hour
    )
    # presigned_subtitle_url = s3.generate_presigned_url(
    #     "get_object",
    #     Params={
    #         "Bucket": s_bucket,
    #         "Key": s_key,
    #         "ResponseContentType": "text/srt"
    #     },
    #     ExpiresIn=3600  # valid for 1 hour
    # )
    # print(f"Presigned URL: {presigned_video_url}, {presigned_subtitle_url}")
    return presigned_video_url


if __name__ == "__main__":
    video_with_redact = {
        "status": "success",
        "output_path": "s3://aws-uni-input2/text_processed_videos/1min_test/1min_test.mp4"
    }
    audio_with_redact = {
        "status": "success",
        "aDedaction_path": "aws-uni-input2/audio/1min_test.mp3",
        "sDedaction_path": "aws-uni-input2/audio/1min_test_transcript.SRT"
    }
    result_path = mux_text_audio(video_with_redact, audio_with_redact, region="ap-southeast-2")
    print("Final MP4:", result_path)
