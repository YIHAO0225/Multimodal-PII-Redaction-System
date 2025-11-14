import os, uuid, time, tempfile, json, requests, boto3
from botocore.config import Config
from .s3_tool import push_AWS_S3



def _get_clients(region: str):
    transcribe = boto3.client(
        "transcribe",
        region_name=region,
        config=Config(retries={"max_attempts": 3, "mode": "standard"})
    )
    comprehend = boto3.client(
        "comprehend",
        region_name=region,
        config=Config(retries={"max_attempts": 3, "mode": "standard"})
    )
    return transcribe, comprehend


def start_transcribe(transcribe, media_uri: str) -> str:
    job_name = f"transcribe-{uuid.uuid4().hex[:8]}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        LanguageCode="en-US",
    )
    return job_name


def wait_transcribe_done(transcribe, job_name: str) -> dict:
    while True:
        r = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = r["TranscriptionJob"]["TranscriptionJobStatus"]
        if status in ("COMPLETED", "FAILED"):
            if status == "FAILED":
                raise RuntimeError(f"Transcribe job failed: {job_name}")
            return r
        time.sleep(5)


def download_transcribe_json(result_obj: dict) -> dict:
    uri = result_obj["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    return requests.get(uri, timeout=60).json()


def temp_write_text(content: str) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(content)
        return f.name


def temp_write_json(obj: dict) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        return f.name


def extract_full_text(transcribe_json: dict) -> str:
    return transcribe_json.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")


def detect_entities(comprehend, full_text: str) -> list:
    if not full_text:
        return []
    return comprehend.detect_entities(Text=full_text, LanguageCode="en").get("Entities", [])


def _filter_low_confidence(entities: list, threshold: float = 0.80) -> list:
    out = []
    for e in entities:
        try:
            sc = float(e.get("Score", 0.0))
        except Exception:
            sc = 0.0
        if sc >= threshold:
            out.append(e)
    return out


def redact_with_entities(full_text: str, entities: list) -> str:
    redacted_text = full_text
    for ent in entities:
        t = ent.get("Text")
        if t:
            redacted_text = redacted_text.replace(t, "*" * len(t))
    return redacted_text


def build_spans(entities: list) -> list:
    spans = []
    for ent in entities:
        b = ent.get("BeginOffset")
        e = ent.get("EndOffset")
        sc = round(ent.get("Score", 1.0), 3)
        lbl = ent.get("Type")  # e.g., PERSON / EMAIL / PHONE_NUMBER ...
        spans.append({
            "begin": b,
            "end": e,
            "score": sc,
            "label": lbl
        })
    return spans


def build_segments_from_items(transcribe_json: dict) -> dict:
    segments = []
    items = transcribe_json.get("results", {}).get("items", [])
    char_cursor = 0
    seg_id = 0
    current_seg = {
        "id": seg_id,
        "start_time": None,
        "end_time": None,
        "start_char": char_cursor,
        "text": "",
        "words": []
    }

    for item in items:
        itype = item.get("type")
        alts = item.get("alternatives", [])
        if not alts:
            continue

        if itype == "pronunciation":
            word = alts[0].get("content", "")
            if not word:
                continue
            start_time = float(item.get("start_time", 0.0))
            end_time = float(item.get("end_time", 0.0))

            word_entry = {
                "word": word,
                "start_time": start_time,
                "end_time": end_time,
                "start_char": char_cursor,
                "end_char": char_cursor + len(word),
                "is_PII": 0
            }
            current_seg["words"].append(word_entry)
            current_seg["text"] += (" " if current_seg["text"] else "") + word
            char_cursor += len(word) + 1
            if current_seg["start_time"] is None:
                current_seg["start_time"] = start_time
            current_seg["end_time"] = end_time
            current_seg["end_char"] = char_cursor

        elif itype == "punctuation":
            punc = alts[0].get("content", "")
            if not punc:
                continue
            current_seg["text"] += punc
            char_cursor += len(punc)
            current_seg["end_char"] = char_cursor
            if punc in [".", "?", "!"]:
                segments.append(current_seg)
                seg_id += 1
                current_seg = {
                    "id": seg_id,
                    "start_time": None,
                    "end_time": None,
                    "start_char": char_cursor,
                    "text": "",
                    "words": []
                }

    if current_seg["text"]:
        segments.append(current_seg)

    return {"segments": segments}


def mark_pii_by_char_local(spans: list, segments: dict) -> dict:
    span_ranges = [
        (int(s["begin"]), int(s["end"]), float(s.get("score", 1.0)), s.get("label"))
        for s in spans
        if s.get("begin") is not None and s.get("end") is not None
    ]

    for seg in segments.get("segments", []):
        for word in seg.get("words", []):
            if "is_PII" not in word:
                continue
            w_begin, w_end = word.get("start_char"), word.get("end_char")
            if w_begin is None or w_end is None:
                continue

            for b, e, sc, lbl in span_ranges:
                if w_begin >= b and w_end <= e:
                    word["is_PII"] = 1
                    prev = word.get("confidence")
                    word["confidence"] = sc if prev is None else max(prev, sc)

                    if lbl:
                        if "labels" not in word or not isinstance(word["labels"], list):
                            word["labels"] = []
                        if lbl not in word["labels"]:
                            word["labels"].append(lbl)

    return segments

def audio_detector(s3_input: str, region: str = "ap-southeast-2") -> dict:

    in_bucket, in_key = s3_input.split("/", 1)
    media_uri = f"s3://{in_bucket}/{in_key}"

    transcribe, comprehend = _get_clients(region)

    job = start_transcribe(transcribe, media_uri)
    result_obj = wait_transcribe_done(transcribe, job)
    transcribe_json = download_transcribe_json(result_obj)

    original_text = extract_full_text(transcribe_json)
    path_txt = temp_write_text(original_text)
    push_AWS_S3(f"aws-uni-output1/audio/{in_key}/transcript.txt", path_txt)

    raw_entities = detect_entities(comprehend, original_text)
    entities = _filter_low_confidence(raw_entities, threshold=0.80)
    redacted_text = redact_with_entities(original_text, entities)
    path_red_text = temp_write_text(redacted_text)
    push_AWS_S3(f"aws-uni-output1/audio/{in_key}/transcript_redacted.txt", path_red_text)

    spans = build_spans(entities)
    path_sp = temp_write_json(spans)
    push_AWS_S3(f"aws-uni-output1/audio/{in_key}/transcript_spans.json", path_sp)

    segments = build_segments_from_items(transcribe_json)
    path_seg = temp_write_json(segments)
    push_AWS_S3(f"aws-uni-output1/audio/{in_key}/transcript_segments.json", path_seg)

    updated_segments = mark_pii_by_char_local(spans, segments)
    path_seg2 = temp_write_json(updated_segments)
    push_AWS_S3(f"aws-uni-output1/audio/{in_key}/transcript_segments.json", path_seg2)

    return {"status": "success", "path": f"aws-uni-output1/audio/{in_key}/transcript_segments.json"}
