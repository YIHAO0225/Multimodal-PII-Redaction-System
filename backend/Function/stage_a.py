# -*- coding: utf-8 -*-
from __future__ import annotations
import io, json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import boto3
from botocore.config import Config
from PIL import Image
import numpy as np
import cv2

# Valid image file extensions allowed for processing
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Default S3 root directory for output JSON and preview (if used)
OUTPUT_ROOT = "s3://aws-uni-output1/face_framesandjson_output/"

# ------------ helpers ------------
def _is_s3_uri(s: Optional[str]) -> bool:
    """Check whether the given string is an S3 URI."""
    return isinstance(s, str) and s.lower().startswith("s3://")

def _split_s3_uri(uri: str) -> Tuple[str, str]:
    """Split s3://bucket/key into (bucket, key)."""
    s = uri[5:]
    i = s.find("/")
    if i < 0: return s, ""
    return s[:i], s[i + 1:]

def _session(profile: Optional[str] = None):
    """Create a boto3 session (optionally using AWS profile)."""
    if profile: return boto3.session.Session(profile_name=profile)
    return boto3.session.Session()

def _s3_client(profile: Optional[str], region: Optional[str]):
    """Create S3 client with tuned connection pool."""
    sess = _session(profile)
    return sess.client("s3", region_name=region,
                       config=Config(signature_version="s3v4", max_pool_connections=64))

def _rekognition_client(profile: Optional[str], region: Optional[str]):
    """Create AWS Rekognition client."""
    sess = _session(profile)
    return sess.client("rekognition", region_name=region)

def _s3_list_images(s3, bucket: str, prefix: str) -> List[str]:
    """
    List image keys under bucket/prefix and return sorted list.
    Filters by VALID_EXTS to ensure only image files are included.
    """
    if prefix and not prefix.endswith("/"): prefix += "/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"): continue
            if any(key.lower().endswith(ext) for ext in VALID_EXTS):
                keys.append(key)
    return sorted(keys)

def _s3_get_bytes(s3, bucket: str, key: str) -> bytes:
    """Download an object from S3 and return its raw bytes."""
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def _s3_put_bytes(s3, bucket: str, key: str, data: bytes, content_type="application/json"):
    """Upload raw bytes object to S3."""
    s3.put_object(Bucket=bucket, Key=key.lstrip("/"), Body=data, ContentType=content_type)

def _load_timestamps_json_bytes(js_bytes: bytes) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Normalize timestamp JSON input into two lookup dictionaries:

      ts_idx:  frame_index(int) -> metadata(dict)
      ts_name: filename(str)    -> metadata(dict)

    Supports multiple timestamp formats from different upstream systems.
    Ensures timestamp fields map to meta["timestamp_seconds"] where possible.
    """
    obj = json.loads(js_bytes.decode("utf-8"))
    ts_idx: Dict[int, Dict[str, Any]] = {}
    ts_name: Dict[str, Dict[str, Any]] = {}

    # Type 1: Standard structured object with "by_index" and/or "by_name" fields
    if isinstance(obj, dict) and ("by_index" in obj or "by_name" in obj):
        by_index = obj.get("by_index", {}) or {}
        by_name  = obj.get("by_name", {}) or {}

        # Normalize keys of by_index into integer indices
        for k, v in by_index.items():
            try:
                i = int(k)
            except Exception:
                continue
            meta = dict(v)
            # If timestamp exists but timestamp_seconds is missing, convert it
            if "timestamp_seconds" not in meta and "timestamp" in meta:
                try:
                    meta["timestamp_seconds"] = float(meta["timestamp"])
                except Exception:
                    pass
            ts_idx[i] = meta

        # Normalize by_name table: store data indexed by filename only
        for fn, v in by_name.items():
            meta = dict(v)
            if "timestamp_seconds" not in meta and "timestamp" in meta:
                try:
                    meta["timestamp_seconds"] = float(meta["timestamp"])
                except Exception:
                    pass
            ts_name[Path(fn).name] = meta
        return ts_idx, ts_name

    # Type 2: List of records (common case in raw timestamp JSON files)
    if isinstance(obj, list):
        for rec in obj:
            if not isinstance(rec, dict):
                continue

            # Support multiple field names for index: frame/index/frame_idx etc.
            if "frame" in rec:
                try:
                    i = int(rec["frame"])
                except Exception:
                    continue
            elif "index" in rec:
                try:
                    i = int(rec["index"])
                except Exception:
                    continue
            elif "frame_idx" in rec:
                try:
                    i = int(rec["frame_idx"])
                except Exception:
                    continue
            else:
                continue

            meta = dict(rec)
            # Convert "timestamp" into timestamp_seconds if needed
            if "timestamp_seconds" not in meta and "timestamp" in meta:
                try:
                    meta["timestamp_seconds"] = float(meta["timestamp"])
                except Exception:
                    pass

            ts_idx[i] = meta

            # If filename is available, also fill ts_name lookup
            fn = meta.get("frame_filename")
            if isinstance(fn, str) and fn:
                ts_name[Path(fn).name] = meta

        return ts_idx, ts_name

    # Type 3: Dictionary mapping filename -> seconds directly
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                sec = float(v)
            except Exception:
                continue
            fn = Path(k).name
            ts_name[fn] = {"timestamp_seconds": sec, "frame_filename": fn}
        return ts_idx, ts_name

    # Unknown/unsupported structure → return empty lookups
    return ts_idx, ts_name


def _pil_to_jpeg_bytes(pil: Image.Image, quality=90) -> bytes:
    """Convert a PIL image to JPEG-encoded bytes."""
    bio = io.BytesIO()
    pil.save(bio, format="JPEG", quality=quality)
    return bio.getvalue()


def _resize_for_detect(pil: Image.Image, max_side: int):
    """
    Resize image so that its longer side <= detect_max_side.
    Returns resized PIL and scale factor.
    """
    W, H = pil.size
    s = max(W, H)
    if s <= max_side:
        return pil, 1.0
    scale = max_side / float(s)
    return pil.resize((int(W * scale), int(H * scale)), Image.BILINEAR), scale


def _clamp(v, lo, hi): return max(lo, min(hi, v))


def _iou(a, b):
    """Compute IoU (Intersection over Union) between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)


def _greedy_match(A, B, iou_thr):
    """
    Perform greedy one-to-one matching between boxes in A and B based on IoU.
    Returns (matches, unmatched_A_indices, unmatched_B_indices).
    """
    triples = []
    for i, a in enumerate(A):
        best_j, best_iou = -1, 0.0
        for j, b in enumerate(B):
            v = _iou(a, b)
            if v > best_iou: best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            triples.append((i, best_j, best_iou))
    triples.sort(key=lambda x: -x[2])  # highest IoU first

    used_i, used_j, matches = set(), set(), []
    for i, j, _ in triples:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matches.append((i, j))

    return matches, [i for i in range(len(A)) if i not in used_i], [j for j in range(len(B)) if j not in used_j]


def _frames_similar(lo_boxes, hi_boxes, iou_thr):
    """
    Check if two frames have the same number of faces and matching layout.
    Return None if not similar; otherwise return matched box pairs.
    """
    if len(lo_boxes) != len(hi_boxes): return None
    if len(lo_boxes) == 0: return []
    matches, um_lo, um_hi = _greedy_match(lo_boxes, hi_boxes, iou_thr)
    if um_lo or um_hi: return None
    return matches


def _union_boxes(A, B, matches):
    """Combine matched bounding boxes from two frames by taking outer union."""
    merged = []
    for i, j in matches:
        ax1, ay1, ax2, ay2 = A[i]
        bx1, by1, bx2, by2 = B[j]
        merged.append((min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)))
    return merged


def _detect_faces_on_bytes(rek, img_bytes, pad, detect_max_side, min_box_wh):
    """
    Run Rekognition face detection on raw image bytes.
    - Resizes the image (if needed) to limit the longer side to detect_max_side to save cost/time.
    - Calls Rekognition with JPEG-encoded bytes.
    - Converts Rekognition's normalized box to absolute pixel box on original image size.
    - Applies optional padding and min width/height filtering.
    Returns:
      ((W0, H0), boxes) where boxes is a list of (x1, y1, x2, y2) in original pixel space.
    """
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W0, H0 = pil.size
    pil_small, scale = _resize_for_detect(pil, detect_max_side)

    # Rekognition expects bytes (we provide resized JPEG for efficiency)
    resp = rek.detect_faces(Image={"Bytes": _pil_to_jpeg_bytes(pil_small)}, Attributes=["DEFAULT"])

    boxes = []
    for det in resp.get("FaceDetails", []):
        bb = det["BoundingBox"]  # Rekognition's normalized bbox in [0,1] over resized image
        x1, y1 = int(bb["Left"] * pil_small.width),  int(bb["Top"] * pil_small.height)
        x2, y2 = int((bb["Left"] + bb["Width"]) * pil_small.width), int((bb["Top"] + bb["Height"]) * pil_small.height)

        # Optional padding around the box (still on resized coords)
        if pad: x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad

        # Map back to original resolution if we resized
        if scale != 1.0: x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)

        # Clamp to image bounds
        x1, y1, x2, y2 = _clamp(x1, 0, W0), _clamp(y1, 0, H0), _clamp(x2, 0, W0), _clamp(y2, 0, H0)

        # Filter tiny boxes
        if (x2-x1) >= min_box_wh and (y2-y1) >= min_box_wh:
            boxes.append((x1, y1, x2, y2))
    return (W0, H0), boxes

# ------------ public API ------------

def stageA(input_dir_s3: str,
           timestamps_json_s3: Optional[str] = None,
           *,
           profile: Optional[str] = None,
           region: Optional[str] = None,
           pad: int = 6,
           detect_max_side: int = 1600,
           min_box_wh: int = 16,
           similar_iou: float = 0.55,
           assume_fixed_size: bool = True,
           emit_norm: str = "all",
           fps: Optional[float] = None) -> str:
    """
    Produce face segments by adaptive binary-split:
      - Detect faces only on segment endpoints.
      - If endpoints are "similar" (1-1 matches pass IoU threshold), union the boxes and create a segment.
      - Else split the interval and recurse.
    Output schema (each segment):
      {
        "start_index": int,
        "end_index": int,
        "start_time": float|null,
        "end_time": float|null,
        "label": "face",
        "status": "auto",
        "boxes": [
          {
            "pixel": {"x1":..., "y1":..., "x2":..., "y2":...},
            "normalized_coordinates": {"x_min":..., "y_min":..., "x_max":..., "y_max":...}
          }, ...
        ]
      }

    Dependencies used:
      _is_s3_uri, _s3_client, _rekognition_client, _split_s3_uri, _s3_get_bytes,
      _s3_put_bytes, _s3_list_images, _detect_faces_on_bytes, _frames_similar,
      _union_boxes, OUTPUT_ROOT

    New arg:
      fps (Optional[float]): if timestamps lack seconds, derive seconds via index/fps as fallback.
    """
    assert _is_s3_uri(input_dir_s3)
    if timestamps_json_s3 is not None:
        assert _is_s3_uri(timestamps_json_s3)

    # Initialize AWS clients
    s3 = _s3_client(profile, region)
    rek = _rekognition_client(profile, region)
    bkt, pfx = _split_s3_uri(input_dir_s3)

    # Load / auto-discover timestamps JSON near frames
    if timestamps_json_s3:
        tb, tk = _split_s3_uri(timestamps_json_s3)
        ts_bytes = _s3_get_bytes(s3, tb, tk)
    else:
        ts_bytes = None
        # Try several common filenames if explicit path not provided
        for cand in ("frame_timestamps.json", "timestamps.json", "time.json"):
            test = (pfx.rstrip("/") + "/" + cand) if pfx else cand
            try:
                ts_bytes = _s3_get_bytes(s3, bkt, test)
                break
            except Exception:
                continue
        if not ts_bytes:
            raise RuntimeError("No timestamps JSON found (tried frame_timestamps.json / timestamps.json / time.json).")

    # Parse timestamps into index/name lookup maps
    ts_idx, ts_name = _load_timestamps_json_bytes(ts_bytes)

    # Enumerate frame keys under input prefix
    keys = _s3_list_images(s3, bkt, pfx)
    if not keys:
        raise RuntimeError(f"No image frames under {input_dir_s3}")

    # --- Robust parsing of time (seconds) from meta, with fps fallback ---
    def _parse_time_seconds(row: Optional[Dict[str, Any]], fallback_idx: int) -> Optional[float]:
        """
        Attempt to derive seconds from multiple possible fields:
          - numeric: timestamp_seconds / seconds / sec / t / s / timestamp
          - milliseconds fields: timestamp_milliseconds / milliseconds / ms
          - human-readable strings: timestamp_formatted / timestamp_simple / time
        If none present and fps is given, use index/fps fallback (prefer frame_number in meta).
        """
        if not isinstance(row, dict):
            row = {}

        # direct seconds from various aliases (including "timestamp")
        for key in ("timestamp_seconds", "seconds", "sec", "t", "s", "timestamp"):
            if key in row:
                try:
                    return float(row[key])
                except Exception:
                    pass

        # milliseconds → seconds
        for key in ("timestamp_milliseconds", "milliseconds", "ms"):
            if key in row:
                try:
                    return float(row[key]) / 1000.0
                except Exception:
                    pass

        # parse formatted time string "HH:MM:SS(.sss)" or "MM:SS(.sss)"
        for key in ("timestamp_formatted", "timestamp_simple", "time"):
            if key in row and isinstance(row[key], str):
                s = row[key].strip()
                try:
                    parts = s.split(":")
                    if len(parts) == 3:
                        h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
                        return h*3600 + m*60 + sec
                    elif len(parts) == 2:
                        m = float(parts[0]); sec = float(parts[1])
                        return m*60 + sec
                except Exception:
                    pass

        # fallback via fps (if provided)
        if fps and fps > 0:
            for key in ("frame_number", "frame", "frame_idx", "index", "i"):
                if key in row:
                    try:
                        return float(row[key]) / float(fps)
                    except Exception:
                        pass
            return float(fallback_idx) / float(fps)

        return None

    # Caches for image size and detection results to avoid repeated work
    size_cache: Dict[int, Tuple[int, int]] = {}
    det_cache: Dict[int, List[Tuple[int, int, int, int]]] = {}
    fixed_wh: Optional[Tuple[int, int]] = None  # single (W,H) if assume_fixed_size

    def ensure_detect(i: int):
        """
        Ensure frame i has been detected and cached:
          - downloads bytes from S3
          - calls Rekognition (with resize optimization)
          - caches original size and face boxes
        """
        nonlocal fixed_wh
        if i in det_cache:
            return
        img_bytes = _s3_get_bytes(s3, bkt, keys[i])
        (W, H), boxes = _detect_faces_on_bytes(rek, img_bytes, pad, detect_max_side, min_box_wh)
        size_cache[i] = (W, H)
        det_cache[i] = boxes
        if assume_fixed_size and fixed_wh is None:
            fixed_wh = (W, H)

    # Accumulated output segments
    segments: List[Dict[str, Any]] = []

    # Normalize one bbox into output schema (pixel + optional normalized_coordinates)
    def _to_out_box(x1: int, y1: int, x2: int, y2: int, W: Optional[int], H: Optional[int]) -> Dict[str, Any]:
        out = {
            "pixel": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }
        if emit_norm == "all":
            out["normalized_coordinates"] = {
                "x_min": (x1 / W) if W else 0.0,
                "y_min": (y1 / H) if H else 0.0,
                "x_max": (x2 / W) if W else 0.0,
                "y_max": (y2 / H) if H else 0.0,
            }
        return out

    # Core recursion:
    # If endpoints are similar -> create one segment using union boxes.
    # Else -> split [lo, hi] into [lo, mid] and [mid, hi].
    def recurse(lo: int, hi: int):
        ensure_detect(lo)
        ensure_detect(hi)
        matches = _frames_similar(det_cache[lo], det_cache[hi], similar_iou)

        if matches is not None:
            # Similar endpoints: union boxes to represent the whole segment
            merged = [] if matches == [] else _union_boxes(det_cache[lo], det_cache[hi], matches)

            # Resolve times for segment endpoints (prefer index lookup; fallback to filename lookup)
            meta_lo = ts_idx.get(lo) or ts_name.get(Path(keys[lo]).name)
            meta_hi = ts_idx.get(hi) or ts_name.get(Path(keys[hi]).name)
            t0 = _parse_time_seconds(meta_lo, lo)
            t1 = _parse_time_seconds(meta_hi, hi)

            # Use fixed (W,H) if assume_fixed_size, else the cached size from lo
            W, H = fixed_wh if (assume_fixed_size and fixed_wh) else size_cache.get(lo, (None, None))

            seg = {
                "start_index": lo,
                "end_index": hi,
                "start_time": t0,
                "end_time": t1,
                "label": "face",
                "status": "auto",
                "boxes": [_to_out_box(x1, y1, x2, y2, W, H) for (x1, y1, x2, y2) in merged]
            }
            segments.append(seg)
            return

        # Not similar and still splittable -> binary split
        mid = (lo + hi) // 2
        if mid == lo or mid == hi:
            # Can't split further → emit per-frame segments over [lo..hi]
            for k in range(lo, hi + 1):
                ensure_detect(k)
                meta_k = ts_idx.get(k) or ts_name.get(Path(keys[k]).name)
                tk = _parse_time_seconds(meta_k, k)
                Wk, Hk = fixed_wh if (assume_fixed_size and fixed_wh) else size_cache.get(k, (None, None))
                segments.append({
                    "start_index": k,
                    "end_index": k,
                    "start_time": tk,
                    "end_time": tk,
                    "label": "face",
                    "status": "auto",
                    "boxes": [_to_out_box(x1, y1, x2, y2, Wk, Hk) for (x1, y1, x2, y2) in det_cache[k]]
                })
            return

        recurse(lo, mid)
        recurse(mid, hi)

    # Single-frame special case
    if len(keys) == 1:
        ensure_detect(0)
        meta0 = ts_idx.get(0) or ts_name.get(Path(keys[0]).name)
        t0 = _parse_time_seconds(meta0, 0)
        W0, H0 = fixed_wh if (assume_fixed_size and fixed_wh) else size_cache[0]
        segments.append({
            "start_index": 0,
            "end_index": 0,
            "start_time": t0,
            "end_time": t0,
            "label": "face",
            "status": "auto",
            "boxes": [_to_out_box(x1, y1, x2, y2, W0, H0) for (x1, y1, x2, y2) in det_cache[0]]
        })
    else:
        # Multi-frame: start recursion from full range
        recurse(0, len(keys) - 1)

    # Final output object
    out_obj = {
        "segments": segments,
        "count": len(segments),
        "version": "adaptive-bsplit-segments-1"
    }
    json_bytes = json.dumps(out_obj, ensure_ascii=False, indent=2).encode("utf-8")

    # Persist JSON back to S3 under OUTPUT_ROOT, organized by last path token of input prefix
    out_b, out_p = _split_s3_uri(OUTPUT_ROOT)
    last = Path((pfx[:-1] if pfx.endswith("/") else pfx) or "").name or "output"
    # If you want to rename the final file (e.g., detect_segments.json), change only the filename below
    out_key = (out_p.rstrip("/") + f"/{last}/detect.json").lstrip("/")
    _s3_put_bytes(s3, out_b, out_key, json_bytes, content_type="application/json")

    out_full = f"s3://{out_b}/{out_key}"
    print(f"[StageA] JSON uploaded -> {out_full}")
    return out_full