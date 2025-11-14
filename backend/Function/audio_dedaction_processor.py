import os,io,json,tempfile,boto3
from urllib.parse import urlparse
from pydub import AudioSegment
from pydub.generators import Sine
from .s3_tool import push_AWS_S3 


def audio_dedaction(json_s3_path: str, video_s3_path: str, region: str = "ap-southeast-2") -> str:

    def _parse(s):
        if s.startswith("s3://"):
            u = urlparse(s)
            return u.netloc, u.path.lstrip("/")
        parts = s.split("/", 1)
        return parts[0], parts[1]

    json_bucket, json_key = _parse(json_s3_path)
    vid_bucket, vid_key = _parse(video_s3_path)

    s3 = boto3.client("s3", region_name=region)

    audio_obj = s3.get_object(Bucket=vid_bucket, Key=vid_key)
    audio_data = audio_obj["Body"].read()
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp4")

    seg_obj = s3.get_object(Bucket=json_bucket, Key=json_key)
    segments_json = json.loads(seg_obj["Body"].read().decode("utf-8"))

    def _merge_times(pii_times, gap_threshold=0.5):
        if not pii_times:
            return []
        pii_times = sorted(pii_times, key=lambda x: x["start_time"])
        merged = [pii_times[0]]
        for cur in pii_times[1:]:
            last = merged[-1]
            if cur["start_time"] - last["end_time"] <= gap_threshold:
                last["end_time"] = max(last["end_time"], cur["end_time"])
            else:
                merged.append(cur)
        return merged

    def _times_from_segments(segments_json):
        times = []
        for seg in segments_json.get("segments", []):
            for w in seg.get("words", []):
                if w.get("is_PII", 0) == 1:
                    st = w.get("start_time")
                    et = w.get("end_time")
                    if isinstance(st, (int, float)) and isinstance(et, (int, float)) and et > st:
                        times.append({"start_time": float(st), "end_time": float(et)})
        return _merge_times(times)

    pii_times = _times_from_segments(segments_json)

    beep_hz = 500
    beep_gain_db = -20

    for span in pii_times:
        start_ms = int(span["start_time"] * 1000)
        end_ms   = int(span["end_time"] * 1000)
        if end_ms <= start_ms:
            continue

        dur = end_ms - start_ms
        beep = Sine(beep_hz).to_audio_segment(duration=dur).apply_gain(beep_gain_db)
        silence = AudioSegment.silent(duration=dur)

        audio = audio[:start_ms] + silence + audio[end_ms:]
        audio = audio.overlay(beep, position=start_ms)

    base_name = os.path.splitext(os.path.basename(vid_key))[0]

    with tempfile.TemporaryDirectory() as td:
        local_mp3 = os.path.join(td, f"{base_name}.mp3")
        audio.export(local_mp3, format="mp3")

        target_s3 = f"aws-uni-input2/audio/{base_name}.mp3"
        s3_url = push_AWS_S3(target_s3, local_mp3)

    return s3_url


def subtitle_dedaction(json_s3_path: str, video_s3_path: str, region: str = "ap-southeast-2") -> str:
    def _parse(s):
        if s.startswith("s3://"):
            u = urlparse(s)
            return u.netloc, u.path.lstrip("/")
        parts = s.split("/", 1)
        return parts[0], parts[1]

    def format_timestamp(seconds: float) -> str:
        millis = int(seconds * 1000)
        hours = millis // 3600000
        minutes = (millis % 3600000) // 60000
        secs = (millis % 60000) // 1000
        ms = millis % 1000
        return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"

    json_bucket, json_key = _parse(json_s3_path)
    vid_bucket, vid_key = _parse(video_s3_path)

    video_base = os.path.splitext(os.path.basename(vid_key))[0]

    s3 = boto3.client("s3", region_name=region)

    seg_obj = s3.get_object(Bucket=json_bucket, Key=json_key)
    segments_all = json.loads(seg_obj["Body"].read().decode("utf-8"))
    segments = segments_all["segments"]

    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        text_chars = list(seg["text"])
        seg_start_char = seg["start_char"]

        for w in seg.get("words", []):
            if w.get("is_PII", 0) == 1:
                local_begin = int(w["start_char"]) - int(seg_start_char)
                local_end   = int(w["end_char"])   - int(seg_start_char)
                for j in range(local_begin, local_end):
                    if 0 <= j < len(text_chars):
                        text_chars[j] = "*"

        redacted_text = "".join(text_chars)
        start_ts = format_timestamp(seg["start_time"])
        end_ts = format_timestamp(seg["end_time"])
        srt_lines.append(f"{i}\n{start_ts} --> {end_ts}\n{redacted_text}\n")

    srt_content = "\n".join(srt_lines)

    target_s3 = f"aws-uni-input2/audio/{video_base}_transcript.SRT"

    with tempfile.TemporaryDirectory() as td:
        local_srt = os.path.join(td, f"{video_base}_transcript.SRT")
        with open(local_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)

        s3_url = push_AWS_S3(target_s3, local_srt)

    return s3_url


def process_dedaction(json_s3_path: str, video_s3_path: str, region: str = "ap-southeast-2") -> str:
    aDedaction = audio_dedaction(json_s3_path, video_s3_path, region)
    sDedaction = subtitle_dedaction(json_s3_path,video_s3_path,region)
    return {
            "status": "success",
            "aDedaction_path": aDedaction,
            "sDedaction_path": sDedaction,
        }
