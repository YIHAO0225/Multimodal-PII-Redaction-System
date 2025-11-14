"""
PII Text Detection Module - Unified Output Version + Keyword Enhancement + Coordinate Unification v1.3
- Collaborates with face detection module
- Input: S3 folder path
- Output: Unified PII + Face JSON (single file)
- Adapted to new index table and timestamp format
- Added: Australia-specific PII detection based on keywords
- Added: Keyframe coordinate unification (solves incomplete input box detection)
  * Solution 1: Use outermost boundary (not largest area)
  * Solution 2: Coordinate expansion of 5px (top, bottom, left, right)
- Fixed: Use word boundary matching to avoid keyword false matches (v1.3.1)
"""
import os
import time
import boto3
from PIL import Image
import io
import json
import re


# ========== Australian PII Keyword Dictionary ==========

AU_PII_KEYWORDS = {
    "MEDICAL_RECORD_NUMBER": {
        "keywords": [
            "medical record number", "medical record no", "hospital number",
            "hospital no", "unit record number", "urn", "mrn", "mr#", "mr #",
            "mr number", "chart number", "chart no", "chart #", "record number", "record no"
        ],
        "exclude_patterns": [
            "enter your medical record", "enter medical record",
            "provide medical record", "enter urn"
        ]
    },

    "PATIENT_ID": {
        "keywords": [
            "patient identifier", "patient id", "patient number", "patient no",
            "patient #", "nhi", "national health index", "patient reference", "patient ref"
        ],
        "exclude_patterns": [
            "enter patient id", "enter your patient", "provide patient"
        ]
    },

    "AU_MEDICARE": {
        "keywords": [
            "medicare number", "medicare card number", "medicare no", "medicare #",
            "medicare", "medicare card", "medicare reference", "medicare ref"
        ],
        "exclude_patterns": [
            "enter your medicare", "enter medicare", "provide medicare",
            "bring your medicare card"
        ]
    },

    "PRIVATE_HEALTH_INSURANCE": {
        "keywords": [
            "health insurance number", "health fund number", "insurance number",
            "insurance no", "private health insurance", "phi number",
            "membership number", "member number", "member no",
            "policy number", "policy no"
        ],
        "exclude_patterns": [
            "enter insurance", "enter your insurance", "do you have insurance"
        ]
    },

    "AU_DRIVERS_LICENSE": {
        "keywords": [
            "driver's licence", "driver licence", "drivers licence", "driving licence",
            "driver's license", "driver license", "drivers license",
            "dl number", "dl no", "dl #", "licence number", "license number",
            "licence no", "license no", "licence card number"
        ],
        "exclude_patterns": [
            "enter your driver", "enter driver", "provide driver", "bring your licence"
        ]
    },

    "AU_TAX_FILE_NUMBER": {
        "keywords": [
            "tax file number", "tfn", "tfn number", "tfn no",
            "australian tax file number"
        ],
        "exclude_patterns": [
            "enter tfn", "enter your tfn", "provide tfn"
        ]
    },

    "AU_BUSINESS_NUMBER": {
        "keywords": [
            "australian business number", "abn", "abn number", "abn no",
            "business number", "company abn"
        ],
        "exclude_patterns": [
            "enter abn", "enter your abn", "provide abn"
        ]
    },

    "AU_CENTRELINK": {
        "keywords": [
            "centrelink reference number", "centrelink customer reference number",
            "crn", "crn number", "centrelink number", "centrelink no",
            "customer reference number"
        ],
        "exclude_patterns": [
            "enter crn", "enter your crn", "contact centrelink"
        ]
    },

    "EMPLOYEE_ID": {
        "keywords": [
            "employee id", "employee number", "employee no", "emp id",
            "emp number", "emp no", "emp #", "staff id", "staff number", "staff no",
            "personnel id", "personnel number", "badge number", "badge no", "badge #",
            "worker id", "worker number"
        ],
        "exclude_patterns": [
            "enter employee", "enter your employee", "provide employee", "enter emp"
        ]
    },

    "ACCOUNT_NUMBER": {
        "keywords": [
            "account number", "account no", "account #", "acct number", "acct no", "acct #",
            "customer number", "customer no", "customer #", "reference number",
            "reference no", "ref no", "client number", "client no"
        ],
        "exclude_patterns": [
            "enter account", "enter your account", "provide account"
        ]
    }
}

# Common English words blacklist
COMMON_WORDS_BLACKLIST = {
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    # Articles
    "a", "an", "the",
    # Conjunctions
    "and", "or", "but", "if", "as",
    # Verbs
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "enter", "click", "type", "input", "fill",
    # Common nouns
    "box", "field", "form", "page", "below", "above",
    "here", "there", "this", "that",
    # Directional words
    "left", "right", "top", "bottom", "center", "centre"
}


def pii_text_detector(
    s3_input_path: str,
    region: str = "ap-southeast-2"
) -> dict:
    """
    PII Text Detection Main Function

    Features:
    1. Read video frame images from S3
    2. Detect PII in images (names, addresses, and other sensitive information)
       - Layer 1: AWS Comprehend automatic detection
       - Layer 2: Australia-specific PII detection based on keywords
       - Layer 3: Format inference
       - Layer 4: Substring matching
    3. Coordinate unification processing (new)
    4. Wait for face detection module to complete
    5. Merge and output unified detection result JSON

    Args:
        s3_input_path: S3 input path, for example:
            - "s3://aws-uni-input1/folder1/"
            - "aws-uni-input1/folder1/"
            - "folder1"
        region: AWS region, default ap-southeast-2

    Returns:
        {
            "status": "success",
            "output_path": "unified_detection_output/folder1/",
            "unified_json": "folder1_unified_detection.json"
        }
        or
        {
            "status": "error",
            "message": "Error message"
        }
    """

    print(f"\n{'='*60}")
    print(f"🚀 PII Text Detection Module Started (Unified Output + Keyword + Coordinate Unification)")
    print(f"{'='*60}")
    print(f"📍 Region: {region}")
    print(f"📂 Input Path: {s3_input_path}")

    try:
        # ========== Initialize AWS Clients ==========
        session = boto3.Session(region_name=region)
        textract = session.client("textract")    # OCR text recognition
        comprehend = session.client("comprehend")  # PII detection
        s3 = session.client("s3")                 # S3 storage

        # ========== Parse Input Path ==========
        clean_path = s3_input_path.replace('s3://', '').strip('/')
        path_parts = clean_path.split('/')

        if len(path_parts) >= 2:
            input_bucket = path_parts[0]
            folder_name = path_parts[-1]
            input_prefix = '/'.join(path_parts[1:]) + '/'
        else:
            input_bucket = os.environ.get('BUCKET_NAME', 'aws-uni-input1')
            folder_name = path_parts[0]
            input_prefix = f"{folder_name}/"

        output_bucket = os.environ.get('OUTPUT_BUCKET', 'aws-uni-output1')
        output_prefix = os.environ.get('OUTPUT_PREFIX', 'text_framesandjson_output/')
        face_output_bucket = os.environ.get('FACE_OUTPUT_BUCKET', 'aws-uni-output1')
        face_output_prefix = os.environ.get('FACE_OUTPUT_PREFIX', 'face_framesandjson_output/')

        print(f"📁 Parsed:")
        print(f"   Input Bucket: {input_bucket}")
        print(f"   Folder Name: {folder_name}")
        print(f"   Input Prefix: {input_prefix}")

        # ========== Step 1: Read Input Files ==========
        print(f"\n📥 Step 1: Reading input files...")

        all_files = _list_s3_files(s3, input_bucket, input_prefix,
                                    extensions=[".jpg", ".jpeg", ".png"])
        if not all_files:
            return {"status": "error", "message": f"Folder is empty: {input_prefix}"}
        print(f"   ✅ Found {len(all_files)} images")

        mapping_key = input_prefix + "mapping.json"
        mapping_data = _read_json_from_s3(s3, input_bucket, mapping_key)
        if not mapping_data:
            return {"status": "error", "message": f"mapping.json not found"}

        if not isinstance(mapping_data, list):
            return {"status": "error", "message": "mapping.json format incorrect, should be array"}

        print(f"   ✅ Read mapping.json: {len(mapping_data)} keyframe segments")

        timestamp_data = None
        timestamp_key = input_prefix + "frame_timestamps.json"
        try:
            s3.head_object(Bucket=input_bucket, Key=timestamp_key)
            timestamp_data = _read_json_from_s3(s3, input_bucket, timestamp_key)
            if timestamp_data:
                if isinstance(timestamp_data, list):
                    print(f"   ✅ Read timestamps: {len(timestamp_data)} frames")
                else:
                    print(f"   ⚠️  Timestamp format incorrect")
                    timestamp_data = None
        except:
            print(f"   ⚠️  Timestamp file not found")

        # ========== Step 2: PII Detection ==========
        print(f"\n🔍 Step 2: PII Detection (4-layer detection strategy)...")

        pii_json_list = []
        pii_index_counter = [0]

        first_name = os.path.splitext(all_files[0])[0]
        frame_start_index = 0
        if any(ch.isdigit() for ch in first_name):
            num_part = ''.join([c for c in first_name if c.isdigit()])
            frame_start_index = int(num_part)

        for idx, segment in enumerate(mapping_data):
            start_frame = int(segment.get("start_frame", 0))
            end_frame = int(segment.get("end_frame", 0))

            keyframe_file = _find_keyframe_file(all_files, start_frame, end_frame)

            if not keyframe_file:
                print(f"   ⚠️  [{idx + 1}/{len(mapping_data)}] Keyframe not found (frame range: {start_frame}-{end_frame})")
                continue

            start_idx = start_frame - frame_start_index
            end_idx = end_frame - frame_start_index

            if start_idx < 0:
                start_idx = 0
            if end_idx >= len(all_files):
                end_idx = len(all_files) - 1

            start_pii_idx = len(pii_json_list)

            print(f"   🔍 [{idx + 1}/{len(mapping_data)}] Processing keyframe: {keyframe_file} (frames {start_frame}-{end_frame})")

            try:
                key_img = _read_image_from_s3(s3, input_bucket, input_prefix + keyframe_file)

                # Detect PII (includes 4-layer detection)
                layer_stats = _detect_pii_in_image(
                    textract, comprehend, key_img, keyframe_file,
                    pii_json_list, pii_index_counter
                )

                # Print summary of detections
                total_detected = sum(layer_stats.values())
                if total_detected > 0:
                    print(f"      ✅ Detected {total_detected} PII items")

            except Exception as e:
                print(f"      ❌ Detection failed: {e}")
                continue

            actual_start_frame = start_frame
            actual_end_frame = end_frame
            frame_range = f"{actual_start_frame}-{actual_end_frame}"

            start_time = None
            end_time = None
            if timestamp_data:
                start_time, end_time = _get_time_range_from_frames(
                    actual_start_frame, actual_end_frame, timestamp_data
                )

            for i in range(start_pii_idx, len(pii_json_list)):
                pii_json_list[i]["frame_range"] = frame_range
                pii_json_list[i]["start_time"] = start_time
                pii_json_list[i]["end_time"] = end_time

        print(f"   ✅ PII detection complete: {len(pii_json_list)} PII items detected")

        # ========== Step 2.5: Coordinate Unification (New) ==========
        print(f"\n📐 Step 2.5: Coordinate unification processing...")
        pii_json_list = _unify_pii_coordinates(pii_json_list, tolerance=10)

        # ========== Step 3: Read and Parse Face Detection JSON ==========
        print(f"\n👤 Step 3: Reading face detection results...")

        face_json_data = None
        face_json_ready, face_json_filename = _wait_for_face_json(
            s3, face_output_bucket, face_output_prefix, folder_name
        )

        if face_json_ready and face_json_filename:
            face_json_key = f"{face_output_prefix}{folder_name}/{face_json_filename}"
            face_json_data = _read_json_from_s3(s3, face_output_bucket, face_json_key)

            if face_json_data:
                print(f"   ✅ Successfully read face JSON: {face_json_filename}")
            else:
                print(f"   ⚠️  Failed to read face JSON")
        else:
            print(f"   ⚠️  Face JSON not ready")

        # ========== Step 4: Merge PII and Face Detection Data ==========
        print(f"\n🔄 Step 4: Merging detection data...")

        unified_json = _merge_detections(
            pii_detections=pii_json_list,
            face_json_data=face_json_data,
            folder_name=folder_name
        )

        print(f"   ✅ Merge complete:")
        print(f"      - PII detections: {len(unified_json['pii_detections'])}")
        if face_json_data:
            print(f"      - Face detections: {len(unified_json['face_detections']['frames'])}")

        # ========== Step 5: Output Unified JSON to S3 ==========
        print(f"\n📤 Step 5: Uploading unified detection results...")

        output_folder_prefix = f"{output_prefix}{folder_name}/"
        unified_json_filename = f"{folder_name}.json"
        unified_json_key = output_folder_prefix + unified_json_filename

        _upload_json_to_s3(s3, unified_json, output_bucket, unified_json_key)
        print(f"   ✅ Unified JSON: {unified_json_key}")

        final_output_folder_prefix = f"{output_bucket}/{output_folder_prefix}"

        # ========== Step 6: Return Result Path ==========
        print(f"\n🎉 Step 6: Processing complete!")
        print(f"   📦 Output path: {final_output_folder_prefix}")
        print(f"   📋 Output file: {unified_json_filename}")

        return {
            "status": "success",
            "output_path": final_output_folder_prefix,
            "unified_json": unified_json_filename
        }

    except Exception as e:
        import traceback
        print(f"\n❌ Processing error: {e}")
        traceback.print_exc()

        return {
            "status": "error",
            "message": str(e)
        }


# ========== Core Detection Functions ==========

def _detect_pii_in_image(textract, comprehend, image, image_name, pii_json_list, pii_index_counter):
    """
    Detect PII in images - 4-layer detection strategy

    Layer 1: AWS Comprehend automatic detection
    Layer 2: Australia-specific PII detection based on keywords (new)
    Layer 3: Format inference
    Layer 4: Substring matching

    Returns:
        Dictionary with layer statistics
    """
    layer_stats = {"layer1": 0, "layer2": 0, "layer3": 0, "layer4": 0}

    # ========== Step 1: OCR Text Recognition ==========
    df_all = _run_ocr_textract(textract, image)

    if not df_all:
        return layer_stats

    img_width, img_height = image.size
    full_text = " ".join([row["text"] for row in df_all])
    drawn_boxes = set()
    known_sensitive_texts = []
    known_pii_texts = []

    # ========== Layer 1: Comprehend Detection ==========
    pii_entities = comprehend.detect_pii_entities(Text=full_text, LanguageCode="en")["Entities"]

    for ent in pii_entities:
        phrase = full_text[ent["BeginOffset"]:ent["EndOffset"]]
        norm_phrase = _normalize(phrase)
        matched = _match_pii(norm_phrase, df_all)

        if matched is not None:
            seen = set()
            matched_dedup = []
            for row in matched:
                key = (row["x_min"], row["y_min"], row["x_max"], row["y_max"])
                if key not in seen:
                    seen.add(key)
                    matched_dedup.append(row)

            for row in matched_dedup:
                coord = (row["x_min"], row["y_min"], row["x_max"], row["y_max"], row["text"])

                if coord not in drawn_boxes:
                    drawn_boxes.add(coord)

                    normalized = _normalize_coordinates(
                        row["x_min"], row["y_min"], row["x_max"], row["y_max"],
                        img_width, img_height
                    )

                    pii_json_list.append({
                        "index": pii_index_counter[0],
                        "source_image": image_name,
                        "text": row["text"],
                        "label": ent["Type"],
                        "confidence": float(ent["Score"]),
                        "is_pii": 1,
                        "detection_source": "comprehend",
                        "coordinates": {
                            "x_min": int(row["x_min"]),
                            "y_min": int(row["y_min"]),
                            "x_max": int(row["x_max"]),
                            "y_max": int(row["y_max"])
                        },
                        "normalized_coordinates": normalized,
                        "frame_range": None,
                        "start_time": None,
                        "end_time": None
                    })
                    pii_index_counter[0] += 1
                    layer_stats["layer1"] += 1

                    known_sensitive_texts.append(row["text"])
                    known_pii_texts.append(row["text"])

    # ========== Layer 2: Keyword-triggered Australia-specific PII Detection (New) ==========
    layer2_count = _detect_keyword_triggered_pii(
        df_all, drawn_boxes, image_name, pii_json_list,
        pii_index_counter, img_width, img_height,
        known_sensitive_texts, known_pii_texts
    )
    layer_stats["layer2"] = layer2_count

    # ========== Layer 3: Format Inference ==========
    layer3_start = len(pii_json_list)
    _infer_similar_format_words(df_all, known_sensitive_texts, drawn_boxes,
                                image_name, pii_json_list, pii_index_counter,
                                img_width, img_height)
    layer_stats["layer3"] = len(pii_json_list) - layer3_start

    # ========== Layer 4: Substring Matching ==========
    layer4_start = len(pii_json_list)
    _guess_pii_by_substring(df_all, known_pii_texts, drawn_boxes,
                            image_name, pii_json_list, pii_index_counter,
                            img_width, img_height)
    layer_stats["layer4"] = len(pii_json_list) - layer4_start

    return layer_stats


def _detect_keyword_triggered_pii(df_all, drawn_boxes, image_name, pii_json_list,
                                  pii_index_counter, img_width, img_height,
                                  known_sensitive_texts, known_pii_texts):
    """
    Keyword-triggered PII detection (Layer 2)

    Detection process:
    1. Iterate through OCR text lines
    2. Find keyword matches (using word boundary matching to avoid substring false matches)
    3. Determine if text is instructional (e.g., "Enter your...")
    4. Extract value after keyword (supports colon, space, newline separators)
    5. Validate value
    6. Label as corresponding PII type
    """
    detected_count = 0

    for idx, text_line in enumerate(df_all):
        text = text_line["text"]
        text_lower = text.lower()

        # Iterate through all PII rules
        for pii_type, rule in AU_PII_KEYWORDS.items():
            keywords = rule["keywords"]
            exclude_patterns = rule.get("exclude_patterns", [])

            # Find keywords
            for keyword in keywords:
                # ========== Modified: Use word boundary matching ==========
                # Build regex pattern: \b ensures only complete words are matched
                pattern = r'\b' + re.escape(keyword) + r'\b'

                # Use regex to match
                if re.search(pattern, text_lower):
                    # Keyword found! Check if it's instructional text
                    is_instructional = _is_instructional_text(text_lower, keyword, exclude_patterns)

                    if is_instructional:
                        continue  # Skip instructional text

                    # Extract value after keyword
                    candidate_values = _extract_value_after_keyword(
                        df_all, idx, text, keyword, max_lookahead=3
                    )

                    for candidate in candidate_values:
                        value_text = candidate["text"]
                        value_line = candidate["line"]

                        # Validate value
                        if not _is_valid_pii_value(value_text):
                            continue

                        # Check if already detected
                        coord_key = (
                            value_line["x_min"], value_line["y_min"],
                            value_line["x_max"], value_line["y_max"],
                            value_text
                        )

                        if coord_key in drawn_boxes:
                            continue

                        # Label as PII
                        drawn_boxes.add(coord_key)

                        normalized = _normalize_coordinates(
                            value_line["x_min"], value_line["y_min"],
                            value_line["x_max"], value_line["y_max"],
                            img_width, img_height
                        )

                        pii_json_list.append({
                            "index": pii_index_counter[0],
                            "source_image": image_name,
                            "text": value_text,
                            "label": pii_type,
                            "confidence": 0.85,  # Rule-based confidence
                            "is_pii": 1,
                            "detection_source": "keyword_triggered",
                            "trigger_keyword": keyword,
                            "coordinates": {
                                "x_min": int(value_line["x_min"]),
                                "y_min": int(value_line["y_min"]),
                                "x_max": int(value_line["x_max"]),
                                "y_max": int(value_line["y_max"])
                            },
                            "normalized_coordinates": normalized,
                            "frame_range": None,
                            "start_time": None,
                            "end_time": None
                        })
                        pii_index_counter[0] += 1
                        detected_count += 1

                        # Add to known lists
                        known_sensitive_texts.append(value_text)
                        known_pii_texts.append(value_text)

                    break  # One keyword match is enough

    return detected_count


def _is_instructional_text(text_lower, keyword, exclude_patterns):
    """
    Determine if text is instructional

    Instructional text characteristics:
    - Contains verbs like "enter", "provide", "type"
    - Matches exclude patterns
    """
    # Check exclude patterns
    for pattern in exclude_patterns:
        if pattern in text_lower:
            return True

    # Check common instructional verbs
    instructional_verbs = [
        "enter your", "enter the", "enter",
        "provide your", "provide the", "provide",
        "type your", "type the", "type",
        "fill in", "fill out",
        "please enter", "please provide",
        "input your", "input the"
    ]

    keyword_pos = text_lower.find(keyword)
    if keyword_pos > 0:
        prefix_text = text_lower[:keyword_pos].strip()
        for verb in instructional_verbs:
            if verb in prefix_text:
                return True

    return False


def _extract_value_after_keyword(df_all, current_idx, current_text, keyword, max_lookahead=3):
    """
    Extract value after keyword

    Supports multiple separators:
    - Colon: "Email Address: john@email.com"
    - Space: "Email Address  john@email.com"
    - Newline: Keyword on one line, value on next line
    """
    candidates = []
    text_lower = current_text.lower()
    keyword_pos = text_lower.find(keyword)

    if keyword_pos == -1:
        return candidates

    # Extract text after keyword
    text_after_keyword = current_text[keyword_pos + len(keyword):].strip()

    # Strategy 1: Same line, with clear separator
    separators = [':', '=', '-']
    for sep in separators:
        if text_after_keyword.startswith(sep):
            value_text = text_after_keyword[1:].strip()
            if value_text:
                # Take first token
                first_token = value_text.split()[0] if value_text.split() else value_text
                candidates.append({
                    "text": first_token,
                    "line": df_all[current_idx]
                })
                return candidates  # With clear separator, return directly

    # Strategy 2: Same line, space-separated
    if text_after_keyword:
        tokens = text_after_keyword.split()
        if tokens:
            first_token = tokens[0]
            candidates.append({
                "text": first_token,
                "line": df_all[current_idx]
            })

    # Strategy 3: Check subsequent lines
    for lookahead in range(1, min(max_lookahead + 1, len(df_all) - current_idx)):
        next_line = df_all[current_idx + lookahead]
        next_text = next_line["text"].strip()

        if next_text:
            # Take first token
            tokens = next_text.split()
            if tokens:
                first_token = tokens[0]
                candidates.append({
                    "text": first_token,
                    "line": next_line
                })

    return candidates


def _is_valid_pii_value(value_text):
    """
    Validate value

    Valid value characteristics:
    - Reasonable length (2-50 characters)
    - Contains digits OR all uppercase OR mixed case OR contains special characters
    - Not in common English words blacklist
    """
    # Length check
    if len(value_text) < 2 or len(value_text) > 50:
        return False

    # Blacklist check
    if value_text.lower() in COMMON_WORDS_BLACKLIST:
        return False

    # Feature check
    has_digit = any(c.isdigit() for c in value_text)
    has_alpha = any(c.isalpha() for c in value_text)
    has_upper = any(c.isupper() for c in value_text)
    has_lower = any(c.islower() for c in value_text)
    has_special = any(not c.isalnum() and not c.isspace() for c in value_text)

    # Valid if meets any of the following:
    # 1. Contains digits
    if has_digit:
        return True

    # 2. All uppercase letters (possibly code/ID)
    if has_alpha and has_upper and not has_lower:
        return True

    # 3. Mixed case (possibly username)
    if has_upper and has_lower:
        return True

    # 4. Contains special characters (e.g., @, _, -, etc.)
    if has_special:
        return True

    # Pure lowercase English word, length < 5, possibly common word
    if has_alpha and has_lower and not has_upper and len(value_text) < 5:
        return False

    return True


# ========== Coordinate Unification Functions (New) ==========

def _unify_pii_coordinates(pii_detections, tolerance=10):
    """
    Unify coordinates for PII detections at the same location

    Features:
    - Identify detections at same location (by proximity of any two corners)
    - Select largest coordinate range
    - Uniformly update all detections at same location

    How it works:
    1. Iterate through all detections
    2. Check if there's a "matching edge" (any edge has two close corners)
    3. If matching edge exists, consider as same location
    4. Select coordinates with largest area as unified coordinates
    5. Update all detections at same location

    Args:
        pii_detections: List of PII detection results
        tolerance: Tolerance for corner matching (pixels)

    Returns:
        List of unified PII detection results
    """
    if not pii_detections:
        return pii_detections

    print(f"      [Coordinate Unification] Starting coordinate unification...")
    print(f"      Tolerance: {tolerance}px")

    # Record maximum coordinates for each location
    # key: location group ID, value: {"indices": [index list], "max_coords": max coordinates, "corners": corner coordinates}
    position_groups = {}

    # Iterate through all detections
    for idx, detection in enumerate(pii_detections):
        coords = detection.get("coordinates", {})
        if not coords:
            continue

        x_min = coords.get("x_min", 0)
        y_min = coords.get("y_min", 0)
        x_max = coords.get("x_max", 0)
        y_max = coords.get("y_max", 0)

        # Four corners
        corners = {
            "top_left": (x_min, y_min),
            "top_right": (x_max, y_min),
            "bottom_left": (x_min, y_max),
            "bottom_right": (x_max, y_max)
        }

        # Find if a group with same location already exists
        found_group = None

        for group_id, group_data in position_groups.items():
            group_corners = group_data["corners"]

            # Check if two corners are close (4 edge cases)
            if _has_matching_edge(corners, group_corners, tolerance):
                found_group = group_id
                break

        # If found group with same location
        if found_group is not None:
            group_data = position_groups[found_group]
            group_data["indices"].append(idx)

            # Update maximum coordinates
            current_max = group_data["max_coords"]
            new_max = _get_larger_bbox(current_max, coords)

            # If coordinates updated, update corners too
            if new_max != current_max:
                group_data["max_coords"] = new_max
                group_data["corners"] = {
                    "top_left": (new_max["x_min"], new_max["y_min"]),
                    "top_right": (new_max["x_max"], new_max["y_min"]),
                    "bottom_left": (new_max["x_min"], new_max["y_max"]),
                    "bottom_right": (new_max["x_max"], new_max["y_max"])
                }
        else:
            # Create new group
            new_group_id = len(position_groups)
            position_groups[new_group_id] = {
                "indices": [idx],
                "max_coords": coords.copy(),
                "corners": corners.copy()
            }

    # Statistics
    unified_count = 0
    multi_detection_groups = 0
    for group_id, group_data in position_groups.items():
        if len(group_data["indices"]) > 1:
            multi_detection_groups += 1
            unified_count += len(group_data["indices"])

    print(f"      Found {len(position_groups)} different locations")
    print(f"      {multi_detection_groups} locations with multiple detections")
    print(f"      Unified {unified_count} detection results")

    # Apply unified coordinates
    for group_id, group_data in position_groups.items():
        indices = group_data["indices"]
        max_coords = group_data["max_coords"]

        if len(indices) > 1:
            # Multiple detections at same location, unify coordinates
            for idx in indices:
                detection = pii_detections[idx]
                old_coords = detection["coordinates"]

                # Solution 2: Expand coordinates (5px in each direction)
                expanded_coords = {
                    "x_min": max(0, int(max_coords["x_min"]) - 5),  # Expand left 5px, but not less than 0
                    "y_min": max(0, int(max_coords["y_min"]) - 5),  # Expand up 5px, but not less than 0
                    "x_max": int(max_coords["x_max"]) + 5,          # Expand right 5px
                    "y_max": int(max_coords["y_max"]) + 5           # Expand down 5px
                }

                # Update coordinates
                detection["coordinates"] = expanded_coords

                # Recalculate normalized coordinates
                # Infer image size from old coordinates and normalized coordinates
                norm_coords = detection.get("normalized_coordinates", {})
                if norm_coords and old_coords.get("x_min", 0) != 0 and norm_coords.get("x_min", 0) != 0:
                    try:
                        img_width = old_coords["x_min"] / norm_coords["x_min"]
                        img_height = old_coords["y_min"] / norm_coords["y_min"]

                        # Update normalized coordinates (using expanded coordinates)
                        detection["normalized_coordinates"] = _normalize_coordinates(
                            expanded_coords["x_min"], expanded_coords["y_min"],
                            expanded_coords["x_max"], expanded_coords["y_max"],
                            img_width, img_height
                        )
                    except (ZeroDivisionError, KeyError):
                        # If calculation fails, keep original normalized coordinates
                        pass

    print(f"      [Coordinate Unification] Complete! (5px expansion applied)")
    return pii_detections


def _has_matching_edge(corners1, corners2, tolerance):
    """
    Determine if two rectangles have a matching edge

    Check 4 cases:
    1. Left edge match: top-left + bottom-left close
    2. Right edge match: top-right + bottom-right close
    3. Top edge match: top-left + top-right close
    4. Bottom edge match: bottom-left + bottom-right close

    Args:
        corners1: Four corners of first rectangle {"top_left": (x,y), ...}
        corners2: Four corners of second rectangle
        tolerance: Tolerance (pixels)

    Returns:
        True if any edge matches
    """
    def points_close(p1, p2, tol):
        """Check if two points are close"""
        return abs(p1[0] - p2[0]) <= tol and abs(p1[1] - p2[1]) <= tol

    # Case 1: Left edge match (top-left + bottom-left)
    if (points_close(corners1["top_left"], corners2["top_left"], tolerance) and
        points_close(corners1["bottom_left"], corners2["bottom_left"], tolerance)):
        return True

    # Case 2: Right edge match (top-right + bottom-right)
    if (points_close(corners1["top_right"], corners2["top_right"], tolerance) and
        points_close(corners1["bottom_right"], corners2["bottom_right"], tolerance)):
        return True

    # Case 3: Top edge match (top-left + top-right)
    if (points_close(corners1["top_left"], corners2["top_left"], tolerance) and
        points_close(corners1["top_right"], corners2["top_right"], tolerance)):
        return True

    # Case 4: Bottom edge match (bottom-left + bottom-right)
    if (points_close(corners1["bottom_left"], corners2["bottom_left"], tolerance) and
        points_close(corners1["bottom_right"], corners2["bottom_right"], tolerance)):
        return True

    return False


def _get_larger_bbox(bbox1, bbox2):
    """
    Return outermost boundary of two bounding boxes (Solution 1)

    New logic: Not selecting larger area, but selecting:
    - x_min: Take minimum value (leftmost)
    - y_min: Take minimum value (topmost)
    - x_max: Take maximum value (rightmost)
    - y_max: Take maximum value (bottommost)

    This ensures complete coverage of both detection boxes

    Args:
        bbox1: First bounding box {"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}
        bbox2: Second bounding box

    Returns:
        Outermost bounding box
    """
    return {
        "x_min": min(bbox1["x_min"], bbox2["x_min"]),
        "y_min": min(bbox1["y_min"], bbox2["y_min"]),
        "x_max": max(bbox1["x_max"], bbox2["x_max"]),
        "y_max": max(bbox1["y_max"], bbox2["y_max"])
    }


# ========== Helper Functions ==========

def _merge_detections(pii_detections, face_json_data, folder_name):
    """Merge PII and face detection data into unified format"""
    # Process PII detection data
    for item in pii_detections:
        item["type"] = "pii"
        item["id"] = f"pii_{item['index']}"

        if item.get("confidence") == "inferred":
            item["confidence"] = None

    # Convert face detection data
    face_frames = []

    if face_json_data:
        if isinstance(face_json_data, dict) and "segments" in face_json_data:
            segments = face_json_data.get("segments", [])
        elif isinstance(face_json_data, list):
            segments = face_json_data
        else:
            segments = []

        for idx, segment in enumerate(segments):
            frame_item = {
                "id": f"face_frame_{idx}",
                "type": "face",
                "label": segment.get("label", "face"),
                "is_pii": 1,
                "confidence": 0.99,
                "start_time": segment.get("start_time"),
                "end_time": segment.get("end_time"),
                "frame_range": f"{segment.get('start_index', 0)}-{segment.get('end_index', 0)}",
                "boxes": []
            }

            boxes = segment.get("boxes", [])
            for box_idx, box in enumerate(boxes):
                norm_coords = box.get("normalized_coordinates", {})
                pixel_coords = box.get("pixel", {})

                box_item = {
                    "id": f"face_{idx}_box_{box_idx}",
                    "normalized_coordinates": {
                        "x_min": norm_coords.get("x_min", 0),
                        "y_min": norm_coords.get("y_min", 0),
                        "x_max": norm_coords.get("x_max", 0),
                        "y_max": norm_coords.get("y_max", 0)
                    }
                }

                if pixel_coords:
                    box_item["pixel_coordinates"] = {
                        "x_min": pixel_coords.get("x1", 0),
                        "y_min": pixel_coords.get("y1", 0),
                        "x_max": pixel_coords.get("x2", 0),
                        "y_max": pixel_coords.get("y2", 0)
                    }

                frame_item["boxes"].append(box_item)

            frame_item["extra_metadata"] = {
                "start_index": segment.get("start_index"),
                "end_index": segment.get("end_index"),
                "original_status": segment.get("status", "auto")
            }

            face_frames.append(frame_item)

    unified_json = {
        "metadata": {
            "video_id": folder_name,
            "version": "v1.3.1_word_boundary_matching",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_pii_detections": len(pii_detections),
            "total_face_detections": len(face_frames),
        },
        "pii_detections": pii_detections,
        "face_detections": {
            "detection_method": "rekognition",
            "frames": face_frames
        }
    }

    return unified_json


def _find_keyframe_file(all_files, start_frame, end_frame):
    """Infer keyframe filename from frame range"""
    for frame_num in [start_frame, (start_frame + end_frame) // 2, end_frame]:
        for file in all_files:
            name = os.path.splitext(file)[0]
            digits = ''.join([c for c in name if c.isdigit()])
            if digits and int(digits) == frame_num:
                return file

    for file in all_files:
        name = os.path.splitext(file)[0]
        digits = ''.join([c for c in name if c.isdigit()])
        if digits:
            file_frame = int(digits)
            if start_frame <= file_frame <= end_frame:
                return file

    return None


def _list_s3_files(s3, bucket, prefix, extensions=None):
    """List S3 files"""
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
            filename = os.path.basename(key)
            if extensions:
                if any(filename.lower().endswith(ext) for ext in extensions):
                    files.append(filename)
            else:
                files.append(filename)

    return sorted(files)


def _read_image_from_s3(s3, bucket, key):
    """Read image from S3"""
    response = s3.get_object(Bucket=bucket, Key=key)
    image_data = response['Body'].read()
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def _read_json_from_s3(s3, bucket, key):
    """Read JSON file"""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except:
        return None


def _upload_json_to_s3(s3, data, bucket, key):
    """Upload JSON to S3"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    s3.put_object(Bucket=bucket, Key=key, Body=json_str.encode('utf-8'),
                  ContentType='application/json')


def _get_time_range_from_frames(start_frame, end_frame, timestamp_data):
    """Get time range from frame range"""
    try:
        frame_map = {frame['frame']: frame for frame in timestamp_data}

        start_time_obj = frame_map.get(start_frame, {})
        end_time_obj = frame_map.get(end_frame, {})

        start_seconds = start_time_obj.get('timestamp', 0.0)
        end_seconds = end_time_obj.get('timestamp', 0.0)

        return start_seconds, end_seconds
    except:
        pass
    return None, None


def _wait_for_face_json(s3, face_bucket, face_prefix, folder_name,
                        max_wait_time=3600, check_interval=30):
    """Wait for face detection JSON"""
    face_folder_prefix = f"{face_prefix}{folder_name}/"
    start_time = time.time()

    print(f"   ⏳ Waiting for face detection JSON: {folder_name}")

    while True:
        try:
            json_files = _list_s3_files(s3, face_bucket, face_folder_prefix,
                                        extensions=[".json"])

            if json_files:
                face_json_filename = json_files[0]
                print(f"   ✅ Face JSON ready: {face_json_filename}")
                return True, face_json_filename

            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                print(f"   ⚠️  Wait timeout ({max_wait_time} seconds)")
                return False, None

            time.sleep(check_interval)
            print(f"      Waiting... {elapsed:.0f}/{max_wait_time} seconds")

        except Exception as e:
            print(f"   ❌ Check failed: {e}")
            time.sleep(check_interval)


def _normalize_coordinates(x_min, y_min, x_max, y_max, img_width, img_height):
    """Normalize coordinates"""
    return {
        "x_min": round(x_min / img_width, 6),
        "y_min": round(y_min / img_height, 6),
        "x_max": round(x_max / img_width, 6),
        "y_max": round(y_max / img_height, 6)
    }


def _normalize(text):
    """Normalize text"""
    return ''.join(e for e in text if e.isalnum())


def _generate_format_template(word):
    """Generate format template"""
    template = ''
    for c in word:
        if c.isdigit():
            template += 'd'
        elif c.isalpha():
            template += 'a'
        elif c.isspace():
            template += 's'
        else:
            template += c
    return template


def _match_pii(norm_phrase, df, max_window=6):
    """Match PII"""
    tokens = [_normalize(t["text"]) for t in df]
    raw_tokens = [t["text"] for t in df]

    for w in range(1, max_window + 1):
        for i in range(len(tokens) - w + 1):
            slice_norm = tokens[i:i + w]
            slice_raw = raw_tokens[i:i + w]
            concat_plain = ''.join(slice_norm)
            concat_hyphen = _normalize('-'.join(slice_raw))

            if norm_phrase in concat_plain or norm_phrase in concat_hyphen:
                return df[i:i + w]
    return None


def _run_ocr_textract(textract, img):
    """Run OCR"""
    img_byte_arr = io.BytesIO()
    img.convert("RGB").save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()

    response = textract.detect_document_text(Document={'Bytes': image_bytes})
    data = []
    for block in response.get("Blocks", []):
        if block["BlockType"] == "LINE":
            text = block["Text"]
            box = block["Geometry"]["BoundingBox"]
            data.append({
                "text": text,
                "x_min": int(box["Left"] * img.width),
                "y_min": int(box["Top"] * img.height),
                "x_max": int((box["Left"] + box["Width"]) * img.width),
                "y_max": int((box["Top"] + box["Height"]) * img.height)
            })

    return data


def _infer_similar_format_words(df_all, known_sensitive_texts, existing_labels,
                                image_name, pii_json_list, pii_index_counter,
                                img_width, img_height):
    """Infer words with similar format"""
    known_templates = set(_generate_format_template(t) for t in known_sensitive_texts)

    for row in df_all:
        template = _generate_format_template(row["text"])
        match_tuple = (row["x_min"], row["y_min"], row["x_max"], row["y_max"], row["text"])

        if template in known_templates and match_tuple not in existing_labels:
            if row["text"].strip().isdigit():
                existing_labels.add(match_tuple)

                normalized = _normalize_coordinates(
                    row["x_min"], row["y_min"], row["x_max"], row["y_max"],
                    img_width, img_height
                )

                pii_json_list.append({
                    "index": pii_index_counter[0],
                    "source_image": image_name,
                    "text": row["text"],
                    "label": "inferred",
                    "confidence": None,
                    "is_pii": 1,
                    "detection_source": "format_inference",
                    "coordinates": {
                        "x_min": int(row["x_min"]),
                        "y_min": int(row["y_min"]),
                        "x_max": int(row["x_max"]),
                        "y_max": int(row["y_max"])
                    },
                    "normalized_coordinates": normalized,
                    "frame_range": None,
                    "start_time": None,
                    "end_time": None
                })
                pii_index_counter[0] += 1


def _guess_pii_by_substring(df_all, known_pii_texts, existing_labels,
                            image_name, pii_json_list, pii_index_counter,
                            img_width, img_height):
    """Guess PII by substring"""
    for row in df_all:
        ocr_text = row["text"].strip()
        coord_tuple = (row["x_min"], row["y_min"], row["x_max"], row["y_max"], ocr_text)

        if coord_tuple in existing_labels or len(ocr_text) < 3:
            continue

        for pii_text in known_pii_texts:
            if ocr_text in pii_text and ocr_text != pii_text:
                existing_labels.add(coord_tuple)

                normalized = _normalize_coordinates(
                    row["x_min"], row["y_min"], row["x_max"], row["y_max"],
                    img_width, img_height
                )

                pii_json_list.append({
                    "index": pii_index_counter[0],
                    "source_image": image_name,
                    "text": ocr_text,
                    "label": "inferred",
                    "confidence": None,
                    "is_pii": 1,
                    "detection_source": "substring_match",
                    "coordinates": {
                        "x_min": int(row["x_min"]),
                        "y_min": int(row["y_min"]),
                        "x_max": int(row["x_max"]),
                        "y_max": int(row["y_max"])
                    },
                    "normalized_coordinates": normalized,
                    "frame_range": None,
                    "start_time": None,
                    "end_time": None
                })
                pii_index_counter[0] += 1
                break


# ========== Main Entry Point (for testing) ==========
if __name__ == "__main__":
    result = pii_text_detector(
        s3_input_path="aws-uni-input1/1min_test/",
        region="ap-southeast-2"
    )

    print(f"\n{'='*60}")
    print(f"📋 Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))