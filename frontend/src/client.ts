// src/client.ts

export const API_BASE = import.meta.env.VITE_API_BASE || '';

async function safeText(res: Response): Promise<string> {
  try { return await res.text() } catch { return '' }
}

export async function apiGet<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: 'GET', ...(init||{}) });
  if (!res.ok) throw new Error(`GET ${path} ${res.status} ${await safeText(res)}`);
  return res.json() as Promise<T>;
}

export async function apiPostJson<T>(path: string, body: unknown, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type':'application/json', ...(init?.headers||{}) },
    body: JSON.stringify(body),
    ...(init||{}),
  });
  if (!res.ok) throw new Error(`POST ${path} ${res.status} ${await safeText(res)}`);
  return res.json() as Promise<T>;
}

export async function apiUpload<T>(path: string, form: FormData, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method:'POST', body: form, ...(init||{}) });
  if (!res.ok) throw new Error(`UPLOAD ${path} ${res.status} ${await safeText(res)}`);
  return res.json() as Promise<T>;
}

export interface NormBox { x1:number; y1:number; x2:number; y2:number }

export interface DetectionItem {
  id: string;
  label: string;
  confidence: number;
  norm_box: NormBox;
  start: number;
  end: number;
  source: 'auto_detection' | 'manual_review';
  status: 'active' | 'deleted';
}

export interface CaptionWord {
  id: string|number;
  text: string;
  start: number;
  end: number;
  keep: boolean; 
}

export interface ProcessResp { 
  task_id: string; 
  status?: 'queued'|'processing'|'done'|'failed' 
}

export interface ResultPollResp { 
  status: 'queued'|'processing'|'done'|'failed'; 
  progress?: number; 
  message?: string;
}

export interface CommitResp { 
  ok: boolean;
  review_version: number;
}

export interface VideoListItem { 
  id?: string; 
  task_id?: string; 
  status?: string; 
  filename?: string; 
  created_at?: string; 
  redacted_video_url?: string;
}


export interface PayloadResponse {
  metadata: {
    task_id: string;
    video: {
      s3_key: string;
      width: number;
      height: number;
      duration: number;
      fps?: number;
    };
    generated_at: string;
    model_versions?: Record<string, string>;
  };
  pii_detections: Array<{
    id: string;
    label: string;
    text?: string;
    confidence?: number;
    start: number;
    end: number;
    norm_box: { x1: number; y1: number; x2: number; y2: number };
    source?: 'ocr' | 'asr' | 'ner' | 'face' | 'fusion';
  }>;
  face_detections: {
    frames: Array<{
      id?: string;
      t: number;
      end?: number;
      confidence?: number;
      frame_range?: string;
      extra_metadata?: Record<string, unknown>;
      norm_boxes: Array<{
        id?: string;
        x1: number;
        y1: number;
        x2: number;
        y2: number;
        confidence?: number;
      }>;
    }>;
  };
  captions: {
    segments: Array<{
      id: string;
      start: number;
      end: number;
      text?: string;
      words?: Array<{
        id: string;
        text: string;
        start: number;
        end: number;
        is_PII: number;
      }>;
    }>;
  };
}


export async function processVideo(file: File): Promise<ProcessResp> {
  const form = new FormData();
  form.append('file', file);
  return apiUpload<ProcessResp>('/api/process', form);
}

export async function getResult(taskId: string): Promise<ResultPollResp> {
  return apiGet<ResultPollResp>(`/api/result?task_id=${encodeURIComponent(taskId)}`);
}


export interface PayloadData {
  detections: DetectionItem[];
  captions: CaptionWord[];
  metadata: PayloadResponse['metadata'];
}

export async function getPayload(taskId: string, useVersion?: number): Promise<PayloadData> {
  if (!taskId) throw new Error('task_id required for /api/payload');
  
  let url = `/api/payload?task_id=${encodeURIComponent(taskId)}`;
  if (useVersion !== undefined) {
    url += `&use_version=${useVersion}`;
  }
  
  const payload = await apiGet<PayloadResponse>(url);

  const detections: DetectionItem[] = [];

  for (const d of payload.pii_detections || []) {
    detections.push({
      id: d.id,
      label: d.label,
      confidence: d.confidence ?? 0,
      norm_box: d.norm_box,
      start: d.start,
      end: d.end,
      source: 'auto_detection',
      status: 'active',
    });
  }

  for (const frame of payload.face_detections?.frames || []) {
    for (const box of frame.norm_boxes || []) {
      detections.push({
        id: box.id ?? cryptoRandomId('face_'),
        label: 'FACE',
        confidence: box.confidence ?? 0,
        norm_box: { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 },
        start: frame.t,
        end: frame.end ?? frame.t, 
        source: 'auto_detection',
        status: 'active',
      });
    }
  }

  const captions: CaptionWord[] = [];
  
  for (const seg of payload.captions?.segments || []) {
    for (const w of seg.words || []) {
      captions.push({
        id: w.id,
        text: w.text,
        start: w.start,
        end: w.end,
        keep: w.is_PII !== 1
      });
    }
  }

  return {
    detections,
    captions,
    metadata: payload.metadata,
  };
}

function cryptoRandomId(prefix = 'id_') {

  const a = crypto.getRandomValues(new Uint32Array(2));
  return `${prefix}${a[0].toString(36)}${a[1].toString(36)}`;
}


export interface PixelBox { x: number; y: number; w: number; h: number }

export async function getAnnotations(taskId: string) {
  const data = await apiGet<any>(`/api/annotations?task_id=${encodeURIComponent(taskId)}`);
  return Array.isArray(data?.annotations) ? data.annotations : (Array.isArray(data) ? data : []);
}

export async function postAnnotations(taskId: string, params: {
  target_space: 'processed' | 'original'; 
  ref_width: number; 
  ref_height: number;
  annotations: Array<{ 
    id: string; 
    label: string; 
    pixel_box: PixelBox; 
    start: number; 
    end: number; 
    source: 'manual_review' 
  }>
}) {
  return apiPostJson<{ ok: boolean }>(`/api/annotations`, { task_id: taskId, ...params });
}

export interface ReviewCommitRequest {
  task_id: string;
  deleted_ids?: string[];
  manual_boxes?: Array<{
    id: string;
    label: string;
    norm_box: { x1: number; y1: number; x2: number; y2: number };
    start: number;
    end: number;
    source?: 'manual_review';
  }>;
  captions_muted?: Array<{ start: number; end: number }>;
  message?: string;
}

export async function commitReview(payload: ReviewCommitRequest): Promise<CommitResp> {
  return apiPostJson<CommitResp>(`/api/review/commit`, payload);
}

export interface FinalizeRequest {
  task_id: string;
  use_version: number; 
}

export interface FinalizeResponse {
  ok: boolean;
  final_task_id: string;
  status: 'queued';
}

export async function finalizeRedaction(payload: FinalizeRequest): Promise<FinalizeResponse> {
  return apiPostJson<FinalizeResponse>(`/api/finalize`, payload);
}

export interface FinalResultResponse {
  status: 'queued' | 'processing' | 'done' | 'failed';
  progress?: number;
  final_video_url?: string;
  message?: string;
}

export async function pollFinalize(finalTaskId: string): Promise<FinalResultResponse> {
  return apiGet<FinalResultResponse>(`/api/final/result?final_task_id=${encodeURIComponent(finalTaskId)}`);
}

export interface CommitItem {
  review_version: number;
  created_at: string;
  message?: string;
  stats?: { 
    deleted?: number; 
    manual?: number; 
    mute_ranges?: number 
  };
}

export async function listCommits(taskId: string): Promise<CommitItem[]> {
  const data = await apiGet<{ items: CommitItem[] }>(`/api/commits?task_id=${encodeURIComponent(taskId)}`);
  return data.items || [];
}


export async function listVideos(): Promise<VideoListItem[]> {
  const data = await apiGet<any>(`/api/history`);
  return Array.isArray(data) ? data : (Array.isArray(data?.items) ? data.items : []);
}

export function buildDownloadUrl(idOrTaskId: string) {
  return `${API_BASE}/api/video/${encodeURIComponent(idOrTaskId)}/download`;
}

export function buildPlayUrl(idOrTaskId: string) {
  return `${API_BASE}/api/video/${encodeURIComponent(idOrTaskId)}/play`;
}
