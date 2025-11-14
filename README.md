Multimodal PII Redaction System 

This project is a full-stack multimodal privacy-preservation system designed to automatically detect, classify, and redact sensitive personal information (PII) from user-testing videos.
It processes audio, screen text, and visual content, applies intelligent redaction (audio beep/silence, video mosaics), and provides a human-in-the-loop interface for reviewing results.

This repository contains both backend (FastAPI) and frontend (React + Vite) implementations.

🎯 Project Overview

Modern product teams often record user-testing sessions, which can unintentionally capture sensitive data such as names, emails, phone numbers, and ID information.
Our system automatically removes such information with minimal human intervention, while still supporting manual review to ensure accuracy and compliance.

End-to-end workflow:

Upload Video
→ Audio transcription & PII detection
→ Frame extraction & OCR-based scene text detection
→ Face detection & text box merging
→ Unified JSON for review
→ Human-in-the-loop corrections
→ Audio redaction (beep/silence) + Video mosaic
→ Final muxed redacted output
→ Download & history record


Final outputs include:

Redacted video (audio + video)

Structured JSON reports

Review logs

Visualized detection results

🧩 Tech Stack
Backend (Python, FastAPI)

FastAPI + Pydantic + Uvicorn

AWS S3 / Transcribe / Comprehend / Textract

OpenCV, PIL, pydub, ffmpeg

sentence-transformers, scikit-learn

Redis / Celery (optional for scaling)

Frontend (React + Vite)

React 18 (JSX + TypeScript mixed)

Material UI / DataGrid

Video preview + timeline visualization

Upload, review, finalize & download UI components

🧑‍💻 My Role & Responsibilities

I contributed to the project as a Backend Lead, Audio Module Developer, and Backend Test Engineer, including:

🏗 Backend Framework Lead

Designed and built the entire backend architecture (FastAPI routing structure, folder layout, API conventions).

Implemented core backend services and established the structure that the team used for development.

Managed backend integration with AWS and media pipelines.

Reviewed teammate pull requests and guided them on backend workflow.

🔊 Audio Module Owner

Implemented all audio-related components, including:

Audio extraction from uploaded videos

AWS Transcribe job orchestration

PII detection using AWS Comprehend

Word-level timestamp processing

Redaction logic:

beep insertion

silence attenuation

exact span replacement using pydub

Audio pipeline JSON schema

S3 upload utilities & error handling

Final mux pipeline (audio + video)

🧪 Backend Testing & Integration

Developed test scripts and frameworks for validating AWS responses, JSON schemas, and pipeline integrity

Conducted end-to-end backend tests (video → JSON → review → redacted output)

Debugged issues across multiple modules (audio, video, OCR, face detection)

Ensured stable runtime behaviour and maintained cross-team compatibility

🧭 Project Lead / Coordination

Acted as team coordinator for backend & AWS-related tasks

Helped align team workflows and resolve integration blockers

Conducted knowledge-sharing sessions with teammates on backend & AWS logic

📌 Key Features
🔍 Multimodal PII Detection

Audio: Transcribe + Comprehend (names, phone, ID, email, dates, etc.)

Visual: Textract OCR + heuristics

Face detection with AWS Rekognition

Custom box merging + coordinate normalization

🎛 Human-in-the-loop Review

Users can edit/delete detections

Manual annotation supported

Versioned reviewed JSON

Final redaction integrates all modifications

🔒 Privacy-preserving Redaction

Audio: beep / silence insertion

Video: mosaic over PII & faces

JSON logs for audit and downstream analysis

🚀 How to Run (Development)
Before run this code , you need to have AWS Certificates
Backend
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload


Backend: http://localhost:8000

API base path: /api

Frontend
cd frontend
npm install
npm run dev


Frontend: http://localhost:5173

📂 Repository Structure (Simplified)
backend/
  controllers/      # API routing
  Function/         # audio/video/OCR pipelines
  models/           # schemas
  main.py           # FastAPI entrypoint

frontend/
  src/
    App.jsx         # main UI
    client.ts       # API client
    main.tsx

📄 License

This project was developed as part of the University of Sydney 2025 Capstone Program.
