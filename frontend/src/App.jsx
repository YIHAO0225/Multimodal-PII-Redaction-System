// src/App.jsx
// Based on the latest client.ts with a single endpoint "/api/payload", full P0 implementation:
// Upload → Results → Review → Final → History
// — Use normalized coordinates and seconds only; Save JSON only calls /api/review/commit; Finalize uses /api/finalize.

import { useEffect, useMemo, useRef, useState, useCallback } from 'react'

// MUI
import {
  Box, Button, Card, CardContent, Container, Stack, Typography, Alert, Grid,
  LinearProgress, Divider, Chip, Slider, Select, MenuItem,
  IconButton, Tooltip, TextField, Tabs, Tab, Snackbar,
  CircularProgress, Checkbox, FormControlLabel,
} from '@mui/material'
import { DataGrid } from '@mui/x-data-grid'

// Icons
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import PauseIcon from '@mui/icons-material/Pause'
import SkipNextIcon from '@mui/icons-material/SkipNext'
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious'
import SaveIcon from '@mui/icons-material/Save'
import UndoIcon from '@mui/icons-material/Undo'
import RedoIcon from '@mui/icons-material/Redo'
import SendIcon from '@mui/icons-material/Send'
import DeleteIcon from '@mui/icons-material/Delete'
import ClearIcon from '@mui/icons-material/Clear'

// API (aligned with your latest client.ts)
import {
  processVideo, getResult,
  getPayload, commitReview,
  finalizeRedaction, pollFinalize,
  listVideos, buildDownloadUrl, buildPlayUrl
} from './client'

// ========================= Utility functions =========================

// Format timecode (HH:MM:SS.mmm)
function fmtTimecode(sec = 0) {
  const s = Math.max(0, +sec || 0)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = Math.floor(s % 60)
  const ms = Math.floor((s - Math.floor(s)) * 1000)
  const pad = (n, l = 2) => String(n).padStart(l, '0')
  return `${pad(h)}:${pad(m)}:${pad(ss)}.${String(ms).padStart(3, '0')}`
}

// Time bucket index (step ≤ 0.1s), O(1) to query the current visible set
function buildTimeIndex(events, duration, step = 0.1) {
  const safeStep = Math.max(0.01, Math.min(0.1, step))
  const total = Math.max(0, duration || 0)
  const bucketsCount = Math.max(1, Math.ceil(total / safeStep) + 1)
  const tmp = Array.from({ length: bucketsCount }, () => [])
  events.forEach((e, i) => {
    const bs = Math.max(0, Math.floor((e.start ?? 0) / safeStep))
    const be = Math.min(bucketsCount - 1, Math.floor((e.end ?? 0) / safeStep))
    for (let b = bs; b <= be; b++) tmp[b].push(i)
  })
  return { step: safeStep, buckets: tmp.map(a => Int32Array.from(a)), events, duration: total }
}

function queryVisible(index, t, neighbors = 1) {
  if (!index || !index.buckets) return []
  const b = Math.floor(Math.max(0, t) / index.step)
  const out = []
  const seen = new Set()
  for (let off = -neighbors; off <= neighbors; off++) {
    const bi = b + off
    if (bi >= 0 && bi < index.buckets.length) {
      const arr = index.buckets[bi]
      for (let k = 0; k < arr.length; k++) {
        const idx = arr[k]
        if (!seen.has(idx)) {
          seen.add(idx)
          out.push(idx)
        }
      }
    }
  }
  return out
}

// Select/validate file + create blob preview
function validateAndMakeURL(file) {
  if (!file.type?.startsWith?.('video/')) throw new Error('Video files only (MP4 recommended).')
  const MAX = 1024 * 1024 * 1024 // 1GB
  if (file.size > MAX) throw new Error('File too large (>1GB). Choose a shorter clip.')
  return URL.createObjectURL(file)
}

// ========================= App main component =========================
export default function App() {
  // ===== Top tabs =====
  // 0=Upload 1=Results 2=Review 3=Final 4=History
  const [tab, setTab] = useState(0)

  // ===== Upload: task & video state =====
  const origInputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [originalURL, setOriginalURL] = useState('') // Original video blob preview (also reused on Final page)
  const [taskId, setTaskId] = useState('')

  const [procStatus, setProcStatus] = useState('idle') // idle/uploading/queued/processing/done/error
  const [procProgress, setProcProgress] = useState(0)
  const [procMessage, setProcMessage] = useState('') // Current processing message
  const [error, setError] = useState('')
  const [toast, setToast] = useState({ open:false, msg:'' })

  // Drag-and-drop upload
  const [isDragging, setIsDragging] = useState(false)
  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true) }
  const onDragLeave = (e) => { e.preventDefault(); if (!e.currentTarget.contains(e.relatedTarget)) setIsDragging(false) }
  const addFiles = (fileList) => {
    try {
      const f = Array.from(fileList || []).filter(Boolean)[0]
      if (!f) return
      const url = validateAndMakeURL(f)
      if (originalURL?.startsWith?.('blob:')) { try { URL.revokeObjectURL(originalURL) } catch {} }
      setSelectedFile(f); setOriginalURL(url); setError(''); setTab(0)
    } catch (e) { setError(e.message || 'Invalid file') }
  }
  const onDrop = (e) => { e.preventDefault(); setIsDragging(false); addFiles(e.dataTransfer.files) }
  const onPickOriginal = (e) => addFiles(e.target.files)
  useEffect(() => () => { if (originalURL?.startsWith?.('blob:')) { try { URL.revokeObjectURL(originalURL) } catch {} } }, [originalURL])

  // One-click reset: revoke local blob, reset state, clear sessionStorage, clear input and selections
  const clearAll = useCallback(() => {
    try {
      if (originalURL?.startsWith?.('blob:')) URL.revokeObjectURL(originalURL)
    } catch {}
    try {
      if (taskId) sessionStorage.removeItem(`deleted_${taskId}`)
    } catch {}
    setSelectedFile(null)
    setOriginalURL('')
    setTaskId('')
    setProcStatus('idle')
    setProcProgress(0)
    setProcMessage('')
    setDetections([])
    setDeletedIds(new Set())
    setCaptions([])
    setResultsLoaded(false)
    setMeta(null)
    setDuration(0)
    setPlayhead(0)
    setFps(30)
    setPlaybackRate(1.0)
    setReviewBoxes([])
    setReviewMuteRanges([])
    setReviewVersion(null)
    setFinalizing(false)
    setFinalProgress(0)
    setFinalMessage('')
    setFinalURL('')
    setHistoryItems([])
    setListError('')
    setIsDragging(false)
    setToast({ open:false, msg:'' })
    setTab(0)
    try { if (origInputRef.current) origInputRef.current.value = '' } catch {}
    try { setSelection?.([]) } catch {}
  }, [originalURL, taskId])

  // ===== Results: auto detections & captions =====
  const [detections, setDetections] = useState([])
  const [deletedIds, setDeletedIds] = useState(new Set())
  const [captions, setCaptions] = useState([])
  const [resultsLoaded, setResultsLoaded] = useState(false)
  const [meta, setMeta] = useState(null)

  // ===== Review: player/time/overlay =====
  const procVideoRef = useRef(null)
  const [duration, setDuration] = useState(0)
  const [playhead, setPlayhead] = useState(0)
  const [fps, setFps] = useState(30)
  const [playbackRate, setPlaybackRate] = useState(1.0)

  const [reviewBoxes, setReviewBoxes] = useState([])
  const [reviewMuteRanges, setReviewMuteRanges] = useState([])
  const [reviewVersion, setReviewVersion] = useState(null)
  const [refWidth, setRefWidth] = useState(1280)
  const [refHeight, setRefHeight] = useState(720)

  const [undoStack, setUndoStack] = useState([])
  const [redoStack, setRedoStack] = useState([])

  // Final
  const [finalizing, setFinalizing] = useState(false)
  const [finalProgress, setFinalProgress] = useState(0)
  const [finalMessage, setFinalMessage] = useState('')
  const [finalURL, setFinalURL] = useState('')

  const finalOrigRef = useRef(null)
  const finalRedRef = useRef(null)
  const [syncFinal, setSyncFinal] = useState(true)
  const [abEnabled, setAbEnabled] = useState(false)
  const [wipe, setWipe] = useState(50)

  // History
  const [historyItems, setHistoryItems] = useState([])
  const [listError, setListError] = useState('')

  const startDetection = async () => {
    try {
      if (!selectedFile) { setError('Please choose a video first'); return }
      setError(''); setProcStatus('uploading'); setProcProgress(0); setProcMessage('')
      const { task_id, status } = await processVideo(selectedFile)
      setTaskId(task_id); setProcStatus(status || 'queued')
      const started = Date.now()
      let pollCount = 0
      while (true) {
        pollCount++
        const r = await getResult(task_id)
        if (typeof r.progress === 'number') setProcProgress(Math.max(0, Math.min(100, r.progress)))
        if (r.status) setProcStatus(r.status)
        if (r.message) setProcMessage(r.message)
        if (r.status === 'done') break
        if (r.status === 'failed') throw new Error(r.message || 'Processing failed')
        const pollInterval = r.status === 'processing' ? 200 : 1000
        await new Promise(res => setTimeout(res, pollInterval))
        if (Date.now() - started > 100 * 60 * 1000) throw new Error('Timeout')
      }
      setTab(1)
      setToast({ open:true, msg:'Detection finished. Loading results…' })
      await loadPayload(task_id)
    } catch (e) {
      setError(e.message || 'Detection failed')
      setProcStatus('error')
    }
  }

  const loadPayload = async (tid) => {
    try {
      setError('')
      const id = tid || taskId
      if (!id) throw new Error('Missing task_id')
      const { detections: det, captions, metadata, pii_detections } = await getPayload(id)
      const detArr = Array.isArray(det) ? det : (Array.isArray(pii_detections) ? pii_detections : [])
      setDetections(detArr)
      setCaptions(Array.isArray(captions) ? captions : [])
      setMeta(metadata || null)
      setResultsLoaded(true)
      if (metadata?.video?.width && metadata?.video?.height) {
        setRefWidth(metadata.video.width); setRefHeight(metadata.video.height)
      }
      if (metadata?.video?.duration) setDuration(metadata.video.duration)
      if (metadata?.video?.fps) setFps(Math.min(120, Math.max(1, Math.round(metadata.video.fps))))
      const saved = JSON.parse(sessionStorage.getItem(`deleted_${id}`) || '[]')
      setDeletedIds(new Set(saved))
    } catch (e) {
      setError(e.message || 'Failed to load payload')
    }
  }

  const persistDeleted = (nextSet) => {
    if (taskId) sessionStorage.setItem(`deleted_${taskId}`, JSON.stringify(Array.from(nextSet)))
  }
  const lastDeletedRef = useRef([])
  const deleteOne = (id) => {
    lastDeletedRef.current = [id]
    setDeletedIds(prev => { const next = new Set(prev); next.add(id); persistDeleted(next); return next })
  }
  const deleteMany = (ids) => {
    lastDeletedRef.current = Array.isArray(ids) ? ids : []
    setDeletedIds(prev => { const next = new Set(prev); ids.forEach(id => next.add(id)); persistDeleted(next); return next })
  }
  const undoDeleteMany = (ids) => {
    setDeletedIds(prev => { const next = new Set(prev); ids.forEach(id => next.delete(id)); persistDeleted(next); return next })
  }

  useEffect(() => {
    let raf = 0
    const tick = () => {
      const v = procVideoRef.current
      if (v) setPlayhead(v.currentTime || 0)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  const timeIndex = useMemo(() => {
    const live = detections.filter(d => !deletedIds.has(d.id))
    return buildTimeIndex(live, Math.max(0.001, duration), 0.1)
  }, [detections, deletedIds, duration])

  const visibleAutoBoxes = useMemo(() => {
    const idxs = queryVisible(timeIndex, playhead, 1)
    const arr = []; for (let i = 0; i < idxs.length; i++) arr.push(timeIndex.events[idxs[i]])
    return arr.length > 500 ? arr.slice(0, 500) : arr
  }, [timeIndex, playhead])

  const recomputeMuteRanges = useCallback((wordsArg) => {
    const words = wordsArg || captions
    const off = words.filter(w => !w.keep).sort((a,b)=>a.start-b.start)
    const ranges = []
    const tol = 0.04
    for (const w of off) {
      const last = ranges[ranges.length-1]
      if (!last) ranges.push({ start:w.start, end:w.end })
      else if (w.start <= last.end + tol) last.end = Math.max(last.end, w.end)
      else ranges.push({ start:w.start, end:w.end })
    }
    setReviewMuteRanges(ranges.map(r => ({ start: Math.max(0, r.start - 0.02), end: r.end + 0.02 })))
  }, [captions])
  useEffect(() => { recomputeMuteRanges(captions) }, [captions, recomputeMuteRanges])

  const pushHistory = useCallback(() => {
    setUndoStack(s => [...s, { boxes: structuredClone(reviewBoxes), words: structuredClone(captions) }])
    setRedoStack([])
  }, [reviewBoxes, captions])

  const undo = () => {
    setUndoStack(s => {
      if (!s.length) return s
      const last = s[s.length-1]
      setRedoStack(r => [...r, { boxes: structuredClone(reviewBoxes), words: structuredClone(captions) }])
      setReviewBoxes(structuredClone(last.boxes))
      setCaptions(structuredClone(last.words))
      setTimeout(()=> recomputeMuteRanges(last.words), 0)
      return s.slice(0,-1)
    })
  }
  const redo = () => {
    setRedoStack(r => {
      if (!r.length) return r
      const last = r[r.length-1]
      setUndoStack(s => [...s, { boxes: structuredClone(reviewBoxes), words: structuredClone(captions) }])
      setReviewBoxes(structuredClone(last.boxes))
      setCaptions(structuredClone(last.words))
      setTimeout(()=> recomputeMuteRanges(last.words), 0)
      return r.slice(0,-1)
    })
  }

  const toggleWordKeep = (id) => {
    pushHistory()
    setCaptions(prev => {
      const next = prev.map(w => w.id === id ? { ...w, keep: !w.keep } : w)
      setTimeout(()=> recomputeMuteRanges(next), 0)
      return next
    })
  }

  const [selectedBoxId, setSelectedBoxId] = useState(null)
  const addManualBox = (partial) => {
    setReviewBoxes(prev => ([ ...prev, { ...partial, id: partial.id || 'box_'+Date.now(), source:'manual_review' } ]))
  }
  const deleteManualBox = (id) => {
    setReviewBoxes(prev => prev.filter(b => b.id !== id))
    if (selectedBoxId === id) setSelectedBoxId(null)
  }
  const patchManualBox = (id, patch) => {
    setReviewBoxes(prev => prev.map(b => b.id===id ? ({...b, ...patch}) : b))
  }

  const submitAndCommitReview = async () => {
    try {
      if (!taskId) { setToast({open:true, msg:'Missing taskId'}); return }
      const manual_boxes = reviewBoxes.map(b => ({
        id: b.id,
        label: b.label || 'manual',
        norm_box: b.norm_box,
        start: b.start,
        end: b.end,
        source: 'manual_review',
      }))
      const captions_muted = reviewMuteRanges.map(r => ({ start: r.start, end: r.end }))
      const { ok, review_version } = await commitReview({
        task_id: taskId,
        deleted_ids: Array.from(deletedIds),
        manual_boxes,
        captions_muted,
        message: 'review commit',
      })
      if (!ok) throw new Error('Commit not ok')
      setReviewVersion(review_version ?? null)
      setToast({ open:true, msg:`Committed v${review_version}` })
    } catch (e) {
      setToast({ open:true, msg: `Commit failed: ${e.message || e}` })
    }
  }

  const generateFinal = async () => {
    try {
      if (!taskId) { setToast({open:true, msg:'Missing taskId'}); return }
      if (!reviewVersion && reviewVersion !== 0) { setToast({open:true, msg:'Please Save JSON first to get review_version'}); return }
      setFinalizing(true); setFinalProgress(0); setFinalMessage('')
      const { final_task_id } = await finalizeRedaction({ task_id: taskId, use_version: reviewVersion })
      const pollId = final_task_id
      const started = Date.now()
      let pollCount = 0
      while (true) {
        pollCount++
        const r = await pollFinalize(pollId)
        if (typeof r.progress === 'number') setFinalProgress(Math.max(0, Math.min(100, r.progress)))
        if (r.message) setFinalMessage(r.message)
        if (r.status === 'done') {
          setFinalURL(r.final_video_url || '')
          setTab(3)
          setToast({ open:true, msg:'Final video ready.' })
          break
        }
        if (r.status === 'failed') throw new Error(r.message || 'Finalize failed')
        const pollInterval = r.status === 'processing' ? 200 : 1000
        await new Promise(res => setTimeout(res, pollInterval))
        if (Date.now() - started > 100*60*1000) throw new Error('Finalize timeout')
      }
    } catch (e) {
      setToast({ open:true, msg: `Finalize failed: ${e.message || e}` })
    } finally {
      setFinalizing(false)
    }
  }

  const fetchHistory = async () => {
    try {
      setListError(''); 
      const arr = await listVideos(); 
      setHistoryItems(arr)
    } catch (e) { 
      setListError(e.message || 'Failed to load list') 
    }
  }

  useEffect(() => {
    const o = finalOrigRef.current
    const r = finalRedRef.current
    if (!o || !r) return
    let syncing = false
    const makePlay = (src, dst) => () => {
      if (!syncFinal || !dst) return
      try { dst.play() } catch {}
    }
    const makePause = (src, dst) => () => {
      if (!syncFinal || !dst) return
      try { dst.pause() } catch {}
    }
    const makeSeek = (src, dst) => () => {
      if (!syncFinal || !dst || syncing) return
      const t = src.currentTime || 0
      if (Math.abs((dst.currentTime || 0) - t) > 0.1) {
        syncing = true
        dst.currentTime = t
        setTimeout(() => { syncing = false }, 0)
      }
    }
    const oPlay = makePlay(o, r)
    const oPause = makePause(o, r)
    const oSeek = makeSeek(o, r)
    const rPlay = makePlay(r, o)
    const rPause = makePause(r, o)
    const rSeek = makeSeek(r, o)
    o.addEventListener('play', oPlay)
    o.addEventListener('pause', oPause)
    o.addEventListener('seeked', oSeek)
    r.addEventListener('play', rPlay)
    r.addEventListener('pause', rPause)
    r.addEventListener('seeked', rSeek)
    return () => {
      o.removeEventListener('play', oPlay)
      o.removeEventListener('pause', oPause)
      o.removeEventListener('seeked', oSeek)
      r.removeEventListener('play', rPlay)
      r.removeEventListener('pause', rPause)
      r.removeEventListener('seeked', rSeek)
    }
  }, [syncFinal, originalURL, finalURL])

  const UploadTab = (
    <>
      <Card variant="outlined" sx={{ borderRadius: 3 }}>
        <CardContent>
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
            <input
              ref={origInputRef}
              type="file"
              accept="video/*"
              onChange={onPickOriginal}
              hidden
            />
            <Button
              variant="contained"
              onClick={() => origInputRef.current?.click()}
            >
              Select Video
            </Button>
            <Button
              variant="outlined"
              onClick={startDetection}
              disabled={
                !selectedFile ||
                procStatus === 'uploading' ||
                procStatus === 'queued' ||
                procStatus === 'processing'
              }
            >
              Start Detection
            </Button>
            <Button
              variant="text"
              color="secondary"
              startIcon={<ClearIcon />}
              onClick={clearAll}
            >
              Clear
            </Button>
            {!!procStatus && procStatus !== 'idle' && (
              <Typography variant="body2">Status: {procStatus}</Typography>
            )}
          </Stack>
          <Box
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            sx={{
              mt: 2, p: 3,
              border: '2px dashed',
              borderColor: isDragging ? 'primary.main' : 'divider',
              borderRadius: 2,
              textAlign: 'center',
              color: 'text.secondary',
              cursor: 'pointer',
              bgcolor: isDragging ? 'action.hover' : 'transparent'
            }}
            onClick={() => origInputRef.current?.click()}
          >
            Drag & drop a video here, or click to select
          </Box>
        </CardContent>
      </Card>
      {(procStatus === 'uploading' || procStatus === 'queued' || procStatus === 'processing') && (
        <Card variant="outlined" sx={{ borderRadius: 3, mt: 2 }}>
          <CardContent>
            <Stack spacing={1}>
              <Stack direction="row" spacing={1} alignItems="center">
                {(procStatus === 'queued' || procStatus === 'processing') && (
                  <CircularProgress size={16} thickness={5} />  
                )}
                <Typography variant="subtitle2">
                  {procMessage || (
                    procStatus === 'uploading' ? 'Uploading…' :
                    procStatus === 'queued'    ? 'Queued…'     :
                    'Processing…'
                  )}
                </Typography>
              </Stack>
              <LinearProgress
                variant={procStatus === 'uploading' ? 'determinate' : 'indeterminate'}
                value={procStatus === 'uploading' ? procProgress : undefined}
              />
              {procStatus === 'uploading' && (
                <Typography variant="caption" color="text.secondary">
                  {procProgress}%
                </Typography>
              )}
            </Stack>
          </CardContent>
        </Card>
      )}
      <Grid container spacing={2} sx={{ mt: 1 }}>
        <Grid item xs={12}>
          {originalURL ? (
            <>
              <Typography variant="subtitle2" color="text.secondary">
                Preview (Original)
                {meta?.video?.duration ? ` · ${fmtTimecode(meta.video.duration)}` : ''}
                {meta?.video?.fps ? ` · ${meta.video.fps}fps` : ''}
              </Typography>
              <video
                src={originalURL}
                controls
                style={{ width: '100%', background: '#000' }}
              />
            </>
          ) : (
            <Alert severity="info">
              Pick or drag a video. Only original is shown on Upload.
            </Alert>
          )}
        </Grid>
      </Grid>
    </>
  )

  const fmtNum2 = (value) => {
    const n = Number(value)
    return Number.isFinite(n) ? n.toFixed(2) : ''
  }

  const ResultsTab = (
    <Card variant="outlined" sx={{ borderRadius: 3 }}>
      <CardContent>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
          <Typography variant="h6">Detection Results</Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Button size="small" variant="contained" onClick={() => setTab(2)}>
            Go to Review
          </Button>
        </Stack>
        {!resultsLoaded ? (
          <Alert severity="info">Waiting for results…</Alert>
        ) : (
          <div style={{ width:'100%' }}>
            <DataGrid
              autoHeight
              rows={detections.map(d => ({
                id: d.id,
                label: d.label,
                confidence: Number.isFinite(Number(d.confidence)) ? Number(d.confidence) : null,
                start: Number.isFinite(Number(d.start)) ? Number(d.start) : null,
                end: Number.isFinite(Number(d.end)) ? Number(d.end) : null,
                status: deletedIds.has(d.id) ? 'deleted' : 'active',
              }))}
              columns={[
                { field:'label', headerName:'PII Type', flex:1 },
                { field:'confidence', headerName:'Confidence', flex:0.6, type:'number' },
                { field:'start', headerName:'Start Time', flex:0.6, type:'number' },
                { field:'end', headerName:'End Time', flex:0.6, type:'number' },
                {
                  field:'actions',
                  headerName:'Actions',
                  flex:0.8,
                  sortable:false,
                  renderCell:(p) => {
                    const isDeleted = p.row.status === 'deleted'
                    if (isDeleted) {
                      return (
                        <Tooltip title="Restore this row">
                          <span>
                            <IconButton
                              size="small"
                              onClick={() => {
                                setDeletedIds(prev => { const next = new Set(prev); next.delete(p.row.id); persistDeleted(next); return next })
                              }}
                            >
                              <UndoIcon fontSize="small" />
                            </IconButton>
                          </span>
                        </Tooltip>
                      )
                    }
                    return (
                      <Tooltip title="Delete (false positive)">
                        <span>
                          <IconButton
                            size="small"
                            onClick={() => deleteOne(p.row.id)}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                    )
                  }
                },
              ]}
              getRowClassName={(p) => p.row.status === 'deleted' ? 'row-deleted' : ''}
              sx={{ '& .row-deleted': { opacity: .45, textDecoration:'line-through' } }}
              initialState={{
                pagination: { paginationModel: { pageSize: 12 } },
                sorting: { sortModel: [{ field: 'start', sort: 'asc' }] },
              }}
            />
          </div>
        )}
      </CardContent>
    </Card>
  )

  function handlePlayPause() {
    const v = procVideoRef.current; if (!v) return
    if (v.paused) { v.playbackRate = playbackRate; v.play().catch(()=>{}); }
    else v.pause()
  }
  function stepFrame(dir) {
    const v = procVideoRef.current; if (!v) return
    const step = 1 / Math.max(1, fps)
    v.pause()
    v.currentTime = Math.max(0, Math.min((v.currentTime || 0) + dir * step, duration || 1e9))
    setPlayhead(v.currentTime)
  }
  function handleSpeed(val) {
    setPlaybackRate(val); const v = procVideoRef.current; if (v) v.playbackRate = val
  }

  const aspect = refWidth && refHeight ? `${refWidth}/${refHeight}` : '16/9'

  const ReviewTab = (
    <>
      <Card variant="outlined" sx={{ borderRadius: 3, mt: 1 }}>
        <CardContent>
          <Grid container spacing={2}
            sx={{ flexWrap: 'nowrap', alignItems: 'flex-start', overflowX: 'auto' }}
          >
            <Grid item sx={{ flex: '1 1 auto', minWidth: 640 }}>
              <Box sx={{ position: 'relative', width: '100%', maxWidth: '100%', aspectRatio: aspect, bgcolor: '#000', borderRadius: 2, overflow: 'hidden' }}>
                <video
                  ref={procVideoRef}
                  src={originalURL}
                  style={{ width: '100%', height: '100%', objectFit: 'fill', display: 'block', background: '#000' }}
                  controls={false}
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget
                    if (v?.videoWidth && v?.videoHeight) { setRefWidth(v.videoWidth); setRefHeight(v.videoHeight) }
                    if (!meta?.video?.duration) setDuration(v.duration || 0)
                  }}
                />
                <OverlayEditor
                  autoBoxes={visibleAutoBoxes}
                  manualBoxes={reviewBoxes}
                  currentTime={playhead}
                  onDeleteAuto={(id)=> setDeletedIds(prev => { const next=new Set(prev); next.add(id); persistDeleted(next); return next })}
                  onAdd={(partial)=> { pushHistory(); addManualBox({ ...partial, start: playhead, end: Math.min(playhead + 0.8, duration) }) }}
                  onSelect={(id)=> setSelectedBoxId(id)}
                  onChangeManual={(id, patch)=> { pushHistory(); patchManualBox(id, patch) }}
                  onDeleteManual={(id)=> { pushHistory(); deleteManualBox(id) }}
                />
              </Box>
              <BoxTimeControls
                selectedBoxId={selectedBoxId}
                duration={duration}
                boxes={reviewBoxes}
                onChange={(id, patch) => { pushHistory(); patchManualBox(id, patch) }}
                onSeek={(t) => { const v = procVideoRef.current; if (v) v.currentTime = t }}
              />
            </Grid>
            <Grid item sx={{
                flex: '0 0 360px',
                width: 360,
              }}
            >
              <Typography variant="subtitle1">Captions (click to toggle keep ✅/❌)</Typography>
              <CaptionEditor
                words={captions}
                currentTime={playhead}
                onToggle={(id)=> { pushHistory(); toggleWordKeep(id) }}
                onJump={(t)=> { const v = procVideoRef.current; if (v) v.currentTime = t }}
                height={480} 
              />
              <Divider sx={{ my: 1 }} />
              <Typography variant="body2" color="text.secondary">Batch mute (drag handles):</Typography>
              <BatchMuteControls
                duration={duration}
                onApply={(s,e,keep)=> {
                  pushHistory()
                  setCaptions(prev => {
                    const next = prev.map(w => (w.end >= s && w.start <= e) ? { ...w, keep } : w)
                    setTimeout(()=> recomputeMuteRanges(next), 0)
                    return next
                  })
                }}
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      <Card variant="outlined" sx={{ borderRadius: 3 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" spacing={2} flexWrap="wrap">
            <Stack direction="row" spacing={1} alignItems="center">
              <Tooltip title="Prev frame"><span><IconButton size="small" onClick={()=> stepFrame(-1)}><SkipPreviousIcon /></IconButton></span></Tooltip>
              <Tooltip title="Play/Pause"><span><IconButton size="small" onClick={handlePlayPause}>{(procVideoRef.current?.paused ?? true) ? <PlayArrowIcon/> : <PauseIcon/>}</IconButton></span></Tooltip>
              <Tooltip title="Next frame"><span><IconButton size="small" onClick={()=> stepFrame(1)}><SkipNextIcon /></IconButton></span></Tooltip>
              <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>Speed</Typography>
              <Select size="small" value={playbackRate} onChange={(e)=> handleSpeed(Number(e.target.value))}>
                {[0.25,0.5,1,1.5,2].map(r => <MenuItem key={r} value={r}>{r}x</MenuItem>)}
              </Select>
              <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>FPS</Typography>
              <TextField size="small" type="number" value={fps} onChange={(e)=> setFps(Math.max(1, Math.min(120, Number(e.target.value)||30)))} inputProps={{ style:{ width:60 } }}/>
              <Typography variant="body2" sx={{ ml: 2 }}>{fmtTimecode(playhead)} / {fmtTimecode(duration)}</Typography>
            </Stack>
            <Box sx={{ flexGrow: 1 }} />
            <Tooltip title="Undo"><span><IconButton onClick={undo} disabled={!undoStack.length}><UndoIcon/></IconButton></span></Tooltip>
            <Tooltip title="Redo"><span><IconButton onClick={redo} disabled={!redoStack.length}><RedoIcon/></IconButton></span></Tooltip>
            <Button variant="outlined" size="small" endIcon={<SaveIcon/>} onClick={submitAndCommitReview}>
              Save Changes
            </Button>
            <Button variant="contained" size="small" endIcon={<SendIcon/>} onClick={generateFinal} disabled={(reviewVersion==null) || finalizing}>
              {finalizing ? `Generate… ${finalProgress}%` : 'Generate Final'}
            </Button>
            {reviewVersion!=null && <Chip label={`v${reviewVersion}`} size="small" />}
          </Stack>
        </CardContent>
      </Card>
      {finalizing && (
        <Card variant="outlined" sx={{ borderRadius: 3, mt: 1 }}>
          <CardContent>
            <Stack spacing={1}>
              <Stack direction="row" spacing={1} alignItems="center">
                <CircularProgress size={16} thickness={5} />
                <Typography variant="subtitle2">
                  {finalMessage || 'Finalizing redacted video…'}
                </Typography>
              </Stack>
              <LinearProgress
                variant={finalProgress > 0 ? 'determinate' : 'indeterminate'}
                value={finalProgress > 0 ? finalProgress : undefined}
              />
              {finalProgress > 0 && (
                <Typography variant="caption" color="text.secondary">
                  {finalProgress}%
                </Typography>
              )}
            </Stack>
          </CardContent>
        </Card>
      )}
      <Card variant="outlined" sx={{ borderRadius: 3, mt: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>Timeline</Typography>
          <MiniTimeline
            duration={duration}
            playhead={playhead}
            autos={visibleAutoBoxes.map(b => ({ start:b.start, end:b.end }))}
            manual={reviewBoxes.map(b => ({ start:b.start || 0, end:b.end || 0 }))}
            mutes={reviewMuteRanges}
            onSeek={(t)=> { const v = procVideoRef.current; if (v) v.currentTime = t }}
          />
        </CardContent>
      </Card>
    </>
  )

  const FinalTab = (
    <Card variant="outlined" sx={{ borderRadius: 3 }}>
      <CardContent>
        <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
          <Typography variant="h6">Final</Typography>
          <Box sx={{ flexGrow: 1 }} />
          {originalURL && finalURL && (
            <>
              <FormControlLabel
                control={
                  <Checkbox
                    size="small"
                    checked={syncFinal}
                    onChange={(e) => setSyncFinal(e.target.checked)}
                  />
                }
                label="Sync playback"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    size="small"
                    checked={abEnabled}
                    onChange={(e) => setAbEnabled(e.target.checked)}
                  />
                }
                label="A/B wipe"
              />
            </>
          )}
        </Stack>
        {abEnabled && originalURL && finalURL ? (
          <>
            <Box
              sx={{
                position: 'relative',
                width: '100%',
                borderRadius: 2,
                overflow: 'hidden',
                boxShadow: 1,
                bgcolor: '#000',
              }}
            >
              <video
                key={originalURL}
                ref={finalOrigRef}
                src={originalURL}
                controls
                style={{ width: '100%', display: 'block', background: '#000' }}
              />
              <video
                key={finalURL}
                ref={finalRedRef}
                src={finalURL}
                controls
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  clipPath: `inset(0 ${100 - wipe}% 0 0)`,
                  background: '#000',
                }}
              />
              <Box
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: `${wipe}%`,
                  width: 2,
                  height: '100%',
                  bgcolor: 'primary.main',
                  transform: 'translateX(-1px)',
                  pointerEvents: 'none',
                }}
              />
            </Box>
            <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Wipe
              </Typography>
              <Slider
                value={wipe}
                onChange={(_, v) => {
                  if (typeof v === 'number') setWipe(v)
                }}
                min={0}
                max={100}
                sx={{ width: 260 }}
              />
              <Typography variant="body2" color="text.secondary">
                {wipe}%
              </Typography>
            </Stack>
          </>
        ) : (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="text.secondary">
                Original
              </Typography>
              {originalURL ? (
                <video
                  ref={finalOrigRef}
                  src={originalURL}
                  controls
                  style={{ width: '100%', background: '#000' }}
                />
              ) : (
                <Alert severity="info">No original video URL</Alert>
              )}
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="text.secondary">
                Redacted
              </Typography>
              {finalURL ? (
                <video
                  ref={finalRedRef}
                  src={finalURL}
                  controls
                  style={{ width: '100%', background: '#000' }}
                  onLoadStart={() =>
                    console.log('[FRONTEND] Video loading started:', finalURL)
                  }
                  onCanPlay={() =>
                    console.log('[FRONTEND] Video can play:', finalURL)
                  }
                  onError={(e) =>
                    console.log('[FRONTEND] Video load error:', e, 'URL:', finalURL)
                  }
                  onLoadedData={() =>
                    console.log('[FRONTEND] Video data loaded:', finalURL)
                  }
                />
              ) : (
                <Alert severity="info">Generating or missing final video…</Alert>
              )}
            </Grid>
          </Grid>
        )}
      </CardContent>
    </Card>
  )

  const HistoryTab = (
    <Card variant="outlined" sx={{ borderRadius: 3 }}>
      <CardContent>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 1 }}>
          <Typography variant="h6">History</Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Button variant="outlined" onClick={fetchHistory}>Refresh</Button>
        </Stack>
        {listError && <Alert severity="error" sx={{ mb: 1 }}>{listError}</Alert>}
        {historyItems.length === 0 ? (
          <Alert severity="info">No records.</Alert>
        ) : (
          <Stack spacing={1}>
            {historyItems.map((it) => (
              <Card key={it.id || it.task_id} variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={2} flexWrap="wrap">
                    <Typography variant="body2"><b>ID:</b> {it.id || it.task_id}</Typography>
                    {it.status && <Typography variant="body2" color="text.secondary"><b>Status:</b> {it.status}</Typography>}
                    {it.filename && <Typography variant="body2" color="text.secondary"><b>File:</b> {it.filename}</Typography>}
                    <Box sx={{ flexGrow: 1 }} />
                    <Stack direction="row" spacing={1}>
                      <Button size="small" variant="contained" component="a" href={buildPlayUrl(it.id || it.task_id)} target="_blank">Play</Button>
                      <Button size="small" variant="outlined" component="a" href={buildDownloadUrl(it.id || it.task_id)} target="_blank" rel="noreferrer">Download</Button>
                    </Stack>
                  </Stack>
                </CardContent>
              </Card>
            ))}
          </Stack>
        )}
      </CardContent>
    </Card>
  )

  return (
    <Container maxWidth="xl" sx={{ py: 3, pb: 4}}>
      <Box
      sx={{
        borderRadius: 4,
        p: 3,
        bgcolor: '#0f172a',
        background:
          'radial-gradient(circle at top, rgba(148,163,253,0.10), transparent 55%) #020817',
        minHeight: '100vh',
      }}
      >
      <Stack spacing={3}>
        <Typography variant="h4" fontWeight={700}>PII Detection & Redaction Model (CS14-1)</Typography>
        <Card variant="outlined" sx={{ borderRadius: 3 }}>
          <CardContent sx={{ pt: 1, pb: 0 }}>
            <Tabs value={tab} onChange={(_, v) => setTab(v)} variant="scrollable" allowScrollButtonsMobile>
              <Tab label="Upload" />
              <Tab label="Results" />
              <Tab label="Review" />
              <Tab label="Final" />
              <Tab label="History" />
            </Tabs>
          </CardContent>
        </Card>
        {error && <Alert severity="error">{error}</Alert>}
        {tab === 0 && UploadTab}
        {tab === 1 && ResultsTab}
        {tab === 2 && ReviewTab}
        {tab === 3 && FinalTab}
        {tab === 4 && HistoryTab}
      </Stack>
      </Box>
      <Snackbar
        open={toast.open}
        onClose={()=> setToast({ open:false, msg:'' })}
        message={toast.msg}
        autoHideDuration={2600}
      />
    </Container>
  )
}

function CaptionEditor({ words, currentTime, onToggle, onJump, height = 420  }) {
  return (
    <div
      style={{
        height,
        overflowY: 'auto',
        overflowX: 'hidden',
        padding: 8,
        border: '1px solid #eee',
        borderRadius: 8,
        whiteSpace: 'normal',
        wordBreak: 'break-word',
        overflowWrap: 'anywhere',
        lineHeight: 1.35,
      }}
    >
      {words.map((w) => {
        const active = currentTime >= w.start && currentTime <= w.end
        return (
          <span
            key={w.id}
            onClick={() => onToggle(w.id)}
            onDoubleClick={() => onJump(w.start)}
            title={`${Number(w.start).toFixed(2)}s ~ ${Number(w.end).toFixed(2)}s`}
            style={{
              marginRight:6, padding:'4px 6px', borderRadius:6, cursor:'pointer',
              border: active ? '1px solid #1976d2' : '1px solid transparent',
              background: w.keep ? '#f5f5f5' : '#ffebee',
              textDecoration: w.keep ? 'none' : 'line-through',
              display:'inline-block', marginBottom:6, maxWidth: '100%', whiteSpace: 'normal', wordBreak: 'break-word', overflowWrap: 'anywhere',
            }}
          >
            {w.text}
          </span>
        )
      })}
    </div>
  )
}

function BatchMuteControls({ duration, onApply }) {
  const [range, setRange] = useState([0, Math.max(1, duration)])
  useEffect(()=> { setRange([0, Math.max(1, duration)]) }, [duration])
  return (
    <>
      <Slider value={range} onChange={(_, v)=> Array.isArray(v) && setRange([Math.max(0, v[0]), Math.min(duration, v[1])])} min={0} max={Math.max(1, duration)} step={0.01} />
      <Stack direction="row" spacing={1}>
        <Button size="small" variant="contained" onClick={()=> onApply(range[0], range[1], false)}>Mute this time</Button>
        <Button size="small" variant="outlined" onClick={()=> onApply(range[0], range[1], true)}>Restore this time</Button>
      </Stack>
    </>
  )
}

function BoxTimeControls({ selectedBoxId, duration, boxes, onChange, onSeek }) {
  const cur = useMemo(() => boxes.find(b => b.id === selectedBoxId) || null, [boxes, selectedBoxId])
  if (!cur) return null
  const clamp = (t) => Math.max(0, Math.min(duration || 0, t))
  const handleChange = (_, v) => {
    if (!Array.isArray(v)) return
    const [s, e] = v
    onChange(cur.id, { start: clamp(Math.min(s, e)), end: clamp(Math.max(s, e)) })
  }
  return (
    <Box sx={{ mt: 1 }}>
      <Typography variant="body2">Selected Box Time</Typography>
      <Slider value={[cur.start || 0, cur.end || 0]} min={0} max={Math.max(1, duration)} step={0.01}
        onChange={handleChange} onChangeCommitted={(_, v)=> { if (Array.isArray(v)) onSeek(v[0]) }} />
      <Typography variant="caption" color="text.secondary">{fmtTimecode(cur.start)} ~ {fmtTimecode(cur.end)}</Typography>
    </Box>
  )
}

function MiniTimeline({ duration, playhead, autos, manual, mutes, onSeek }) {
  const ref = useRef(null)
  const down = (e) => {
    const r = ref.current.getBoundingClientRect()
    const ratio = (e.clientX - r.left) / r.width
    const t = Math.max(0, Math.min(duration, ratio * duration))
    onSeek && onSeek(t)
  }
  const seg = (arr, color, topPct) => (arr||[]).filter(s => s.end > s.start && duration > 0).map((s,i)=> {
    const left = `${(s.start / duration) * 100}%`
    const w = `${((s.end - s.start) / duration) * 100}%`
    return <div key={`${color}-${i}`} style={{ position:'absolute', left, width:w, top:topPct, height:'18%', background:color, opacity:.6, borderRadius:2 }}/>
  })
  const phLeft = duration>0 ? `${(playhead/duration)*100}%` : '0%'
  return (
    <Box ref={ref} onMouseDown={down}
      sx={{ position:'relative', width:'100%', height:60, bgcolor:'#f7f7f7', borderRadius:2, cursor:'crosshair', userSelect:'none' }}>
      {[...Array(11)].map((_,i)=> <div key={i} style={{ position:'absolute', left:`${i*10}%`, top:0, bottom:0, width:1, background:i===0?'#ddd':'#eee' }}/>)}
      {seg(manual, 'red', '12%')}
      {seg(autos,  '#FFC107', '36%')}
      {seg(mutes,  'orange', '60%')}
      <div style={{ position:'absolute', left:phLeft, top:0, bottom:0, width:2, background:'#1976d2' }}/>
    </Box>
  )
}

/**
 * OverlayEditor
 * Fix the "yellow box delete button can't be clicked":
 * - Elevate the button container zIndex
 * - Container has pointerEvents: 'none', button itself has pointerEvents: 'auto'
 * - Execute delete directly on onMouseDown
 */
function OverlayEditor({
  autoBoxes = [],
  manualBoxes = [],
  currentTime,
  onDeleteAuto,
  onAdd,
  onSelect,
  onChangeManual,
  onDeleteManual,
}) {
  const rootRef = useRef(null)
  const [drag, setDrag] = useState(null)
  const [activeId, setActiveId] = useState(null)
  const [mode, setMode] = useState(null)
  const [anchor, setAnchor] = useState(null)

  const clamp01 = (v) => Math.max(0, Math.min(1, v))

  const toNorm = useCallback((clientX, clientY) => {
    const rect = rootRef.current?.getBoundingClientRect()
    if (!rect || !rect.width || !rect.height) {
      return { x: 0, y: 0, inside: false }
    }
    const x = (clientX - rect.left) / rect.width
    const y = (clientY - rect.top) / rect.height
    return { x, y, inside: x >= 0 && x <= 1 && y >= 0 && y <= 1 }
  }, [])

  const manualVisible = manualBoxes.filter(
    (b) =>
      b.norm_box &&
      currentTime >= (b.start ?? 0) - 0.01 &&
      currentTime <= (b.end ?? 0) + 0.01
  )

  const hitTestManual = (nx, ny) => {
    const pad = 0.015
    for (const b of manualVisible) {
      const nb = b.norm_box
      if (!nb) continue
      const { x1, y1, x2, y2 } = nb
      const inside = nx >= x1 && nx <= x2 && ny >= y1 && ny <= y2
      if (!inside) continue
      const near = (ax, ay) =>
        Math.abs(ax - nx) <= pad && Math.abs(ay - ny) <= pad
      if (near(x1, y1)) return { id: b.id, handle: 'resize-nw' }
      if (near(x2, y1)) return { id: b.id, handle: 'resize-ne' }
      if (near(x1, y2)) return { id: b.id, handle: 'resize-sw' }
      if (near(x2, y2)) return { id: b.id, handle: 'resize-se' }
      return { id: b.id, handle: 'move' }
    }
    return { id: null, handle: null }
  }

  const onMouseDown = (e) => {
    if (e.target.closest('[data-box-btn="1"]')) {
      return
    }
    const { x, y, inside } = toNorm(e.clientX, e.clientY)
    if (!inside) return
    const n = { x: clamp01(x), y: clamp01(y) }
    const { id, handle } = hitTestManual(n.x, n.y)
    if (id) {
      setActiveId(id)
      onSelect?.(id)
      setMode(handle)
      setAnchor(n)
      e.stopPropagation()
      return
    }
    setActiveId(null)
    onSelect?.(null)
    setMode(null)
    setAnchor(n)
    setDrag({ x1: n.x, y1: n.y, x2: n.x, y2: n.y })
  }

  const onMouseMove = (e) => {
    const { x, y } = toNorm(e.clientX, e.clientY)
    const n = { x: clamp01(x), y: clamp01(y) }
    if (drag) {
      setDrag((d) => (d ? { ...d, x2: n.x, y2: n.y } : d))
      return
    }
    if (activeId && mode && anchor) {
      const b = manualBoxes.find((x) => x.id === activeId)
      if (!b || !b.norm_box) return
      const { x1, y1, x2, y2 } = b.norm_box
      const dx = n.x - anchor.x
      const dy = n.y - anchor.y
      let nb = { x1, y1, x2, y2 }
      if (mode === 'move') {
        const w = x2 - x1
        const h = y2 - y1
        nb.x1 = clamp01(x1 + dx)
        nb.y1 = clamp01(y1 + dy)
        nb.x2 = clamp01(nb.x1 + w)
        nb.y2 = clamp01(nb.y1 + h)
      } else {
        if (mode === 'resize-nw') {
          nb.x1 = clamp01(Math.min(nb.x2 - 0.005, x1 + dx))
          nb.y1 = clamp01(Math.min(nb.y2 - 0.005, y1 + dy))
        }
        if (mode === 'resize-ne') {
          nb.x2 = clamp01(Math.max(nb.x1 + 0.005, x2 + dx))
          nb.y1 = clamp01(Math.min(nb.y2 - 0.005, y1 + dy))
        }
        if (mode === 'resize-sw') {
          nb.x1 = clamp01(Math.min(nb.x2 - 0.005, x1 + dx))
          nb.y2 = clamp01(Math.max(nb.y1 + 0.005, y2 + dy))
        }
        if (mode === 'resize-se') {
          nb.x2 = clamp01(Math.max(nb.x1 + 0.005, x2 + dx))
          nb.y2 = clamp01(Math.max(nb.y1 + 0.005, y2 + dy))
        }
      }
      setAnchor(n)
      onChangeManual?.(activeId, { norm_box: nb })
    }
  }

  const finishDrag = () => {
    if (drag) {
      const x1 = Math.min(drag.x1, drag.x2)
      const y1 = Math.min(drag.y1, drag.y2)
      const x2 = Math.max(drag.x1, drag.x2)
      const y2 = Math.max(drag.y1, drag.y2)
      setDrag(null)
      if (x2 - x1 > 0.003 && y2 - y1 > 0.003) {
        onAdd?.({
          id: 'box_' + Date.now(),
          start: currentTime,
          end: currentTime + 0.8,
          label: 'manual',
          norm_box: { x1, y1, x2, y2 },
          source: 'manual_review',
        })
      }
    }
    setMode(null)
    setAnchor(null)
  }

  const onMouseUp = () => {
    finishDrag()
  }

  useEffect(() => {
    const end = () => finishDrag()
    window.addEventListener('mouseup', end)
    window.addEventListener('touchend', end)
    window.addEventListener('touchcancel', end)
    return () => {
      window.removeEventListener('mouseup', end)
      window.removeEventListener('touchend', end)
      window.removeEventListener('touchcancel', end)
    }
  }, [drag, activeId, mode, anchor])

  const rectPct = (nb) => {
    const x = (nb.x1 * 100).toFixed(4)
    const y = (nb.y1 * 100).toFixed(4)
    const w = ((nb.x2 - nb.x1) * 100).toFixed(4)
    const h = ((nb.y2 - nb.y1) * 100).toFixed(4)
    return { x, y, w, h }
  }

  return (
    <div
      ref={rootRef}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 2,
        userSelect: 'none',
        cursor: mode?.startsWith('resize')
          ? 'nwse-resize'
          : mode === 'move'
          ? 'move'
          : 'crosshair',
      }}
    >
      {/* Yellow box layer */}
      <svg
        width="100%"
        height="100%"
        style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
      >
        {autoBoxes.map((b) => {
          if (!b.norm_box) return null
          const { x, y, w, h } = rectPct(b.norm_box)
          return (
            <rect
              key={b.id}
              x={x + '%'}
              y={y + '%'}
              width={w + '%'}
              height={h + '%'}
              fill="none"
              stroke="#FFC107"
              strokeWidth={2}
            />
          )
        })}
      </svg>

      {/* Yellow box delete button: explicit zIndex & pointerEvents to fix click issue */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 10, pointerEvents: 'none' }}>
        {autoBoxes.map((b) => {
          if (!b.norm_box) return null
          const { x, y } = rectPct(b.norm_box)
          return (
            <button
              key={b.id}
              data-box-btn="1"
              onMouseDown={(e) => {
                e.stopPropagation()
                e.preventDefault()
                onDeleteAuto?.(b.id)
              }}
              title="Delete this detection"
              style={{
                position: 'absolute',
                left: `${x}%`,
                top: `${y}%`,
                transform: 'translate(-50%, -50%)',
                background: '#FFC107',
                color: '#000',
                border: 'none',
                borderRadius: 6,
                padding: '2px 8px',
                fontWeight: 700,
                cursor: 'pointer',
                pointerEvents: 'auto',      // Important
                boxShadow: '0 0 0 1px rgba(0,0,0,0.25)',
              }}
            >
              ×
            </button>
          )
        })}
      </div>

      {/* Red box layer */}
      <svg
        width="100%"
        height="100%"
        style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
      >
        {manualVisible.map((b) => {
          if (!b.norm_box) return null
          const { x, y, w, h } = rectPct(b.norm_box)
          const isActive = b.id === activeId
          return (
            <g key={b.id}>
              <rect
                x={x + '%'}
                y={y + '%'}
                width={w + '%'}
                height={h + '%'}
                fill="none"
                stroke={isActive ? '#00BCD4' : 'red'}
                strokeWidth={isActive ? 2.5 : 2}
              />
              {isActive &&
                ['nw', 'ne', 'sw', 'se'].map((pos) => {
                  const cx = pos.includes('e') ? b.norm_box.x2 : b.norm_box.x1
                  const cy = pos.includes('s') ? b.norm_box.y2 : b.norm_box.y1
                  return (
                    <rect
                      key={pos}
                      x={`${cx * 100 - 0.8}%`}
                      y={`${cy * 100 - 0.8}%`}
                      width="1.6%"
                      height="1.6%"
                      fill="#00BCD4"
                    />
                  )
                })}
            </g>
          )
        })}

        {drag && (
          <rect
            x={`${Math.min(drag.x1, drag.x2) * 100}%`}
            y={`${Math.min(drag.y1, drag.y2) * 100}%`}
            width={`${Math.abs(drag.x2 - drag.x1) * 100}%`}
            height={`${Math.abs(drag.y2 - drag.y1) * 100}%`
            }
            fill="rgba(0,188,212,0.15)"
            stroke="#00BCD4"
            strokeWidth={1.5}
          />
        )}
      </svg>

      {/* Red box delete button */}
      <div style={{ position: 'absolute', inset: 0, pointerEvents:'none' }}>
        {manualVisible.map((b) => {
          if (!b.norm_box) return null
          const { x, y, w } = rectPct(b.norm_box)
          return (
            <button
              key={b.id}
              data-box-btn="1"
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation()
                onDeleteManual?.(b.id)
              }}
              title="Delete this manual box"
              style={{
                position: 'absolute',
                left: `calc(${parseFloat(x) + parseFloat(w)}%)`,
                top: `${y}%`,
                transform: 'translate(-50%, -50%)',
                background: '#EF5350',
                color: '#fff',
                border: 'none',
                fontSize: 14,
                fontWeight: 700,
                borderRadius: 4,
                padding: '0 4px',
                cursor: 'pointer',
                boxShadow: '0 0 0 1px rgba(0,0,0,0.25)',
                pointerEvents:'auto'
              }}
            >
              ×
            </button>
          )
        })}
      </div>
    </div>
  )
}
