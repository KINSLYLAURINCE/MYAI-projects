import { useState, useEffect, useRef, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000'

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / 1048576).toFixed(1) + ' MB'
}

/* Circular gauge — pure CSS via SVG stroke-dashoffset */
function ConfidenceGauge({ pct, cls }) {
  const radius = 40
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - pct / 100)

  return (
    <div className="result-gauge">
      <svg viewBox="0 0 100 100">
        <circle className="gauge-track" cx="50" cy="50" r={radius} />
        <circle
          className={`gauge-fill ${cls}`}
          cx="50"
          cy="50"
          r={radius}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: offset,
          }}
        />
      </svg>
      <div className="gauge-text">
        <span className="gauge-pct">{Math.round(pct)}%</span>
        <span className="gauge-sub">confidence</span>
      </div>
    </div>
  )
}

export default function Classify() {
  const [file, setFile]             = useState(null)
  const [preview, setPreview]       = useState(null)
  const [imageDims, setImageDims]   = useState(null)
  const [dragging, setDragging]     = useState(false)
  const [loading, setLoading]       = useState(false)
  const [result, setResult]         = useState(null)
  const [error, setError]           = useState(null)
  const [apiStatus, setApiStatus]   = useState('checking')
  const inputRef = useRef(null)

  useEffect(() => {
    const check = async () => {
      try {
        const res  = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(60000) })
        const data = await res.json()
        setApiStatus(data.model_loaded ? 'online' : 'offline')
      } catch {
        setApiStatus('offline')
      }
    }
    check()
    const id = setInterval(check, 8000)
    return () => clearInterval(id)
  }, [])

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    setResult(null)
    setError(null)
    setImageDims(null)

    const url = URL.createObjectURL(f)
    setPreview(url)

    // Get image dimensions
    const img = new Image()
    img.onload = () => {
      setImageDims({ w: img.naturalWidth, h: img.naturalHeight })
    }
    img.src = url
  }, [])

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const removeFile = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setImageDims(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const predict = async () => {
    if (!file) return
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `Server error ${res.status}`)
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message || 'Could not reach the API. Make sure it is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  const isOnline    = apiStatus === 'online'
  const cls         = result?.prediction?.toLowerCase()
  const confidence  = result ? parseFloat(result.confidence) : 0
  const dogPct      = cls === 'dog' ? confidence : 100 - confidence
  const catPct      = cls === 'cat' ? confidence : 100 - confidence

  const statusLabel = {
    checking: 'Connecting to API...',
    online:   'API online — model ready',
    offline:  'API offline or model still loading',
  }[apiStatus]

  return (
    <div className="page">
      <div className="container">

        <div className="classify-header">
          <h1>Classify an Image</h1>
          <p>Upload a photo to get an instant cat or dog prediction with confidence score.</p>
        </div>

        <div className="status-pill">
          <div className={`status-dot ${apiStatus}`} />
          <span className={`status-text ${apiStatus}`}>{statusLabel}</span>
        </div>

        <div className="classify-layout">

          {/* Left: upload form */}
          <div>
            <div className="classify-card">
              <div className="classify-card-header">
                <svg viewBox="0 0 24 24">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <span>Upload Image</span>
              </div>
              <div className="classify-card-body">

                {!preview ? (
                  <div
                    className={`dropzone${dragging ? ' dragging' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={onDrop}
                    onClick={() => inputRef.current?.click()}
                  >
                    <input
                      ref={inputRef}
                      type="file"
                      accept="image/*"
                      style={{ display: 'none' }}
                      onChange={(e) => handleFile(e.target.files[0])}
                    />
                    <div className="dropzone-icon-wrap">
                      <svg viewBox="0 0 24 24">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                      </svg>
                    </div>
                    <div className="dropzone-title">Drop an image here</div>
                    <div className="dropzone-sub">
                      or click to browse<br />JPG, PNG, WEBP &mdash; up to 10 MB
                    </div>
                  </div>
                ) : (
                  <div className="preview-wrap">
                    <img src={preview} alt="Selected preview" />
                    <button className="preview-remove" onClick={removeFile} title="Remove image">
                      <svg viewBox="0 0 24 24">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                      </svg>
                    </button>
                    <div className="preview-meta">
                      <span className="preview-name">{file?.name}</span>
                      <span className="preview-info">
                        {file && formatBytes(file.size)}
                        {imageDims && ` · ${imageDims.w}×${imageDims.h}`}
                      </span>
                    </div>
                  </div>
                )}

                <button
                  className="btn-classify"
                  onClick={predict}
                  disabled={!file || loading || apiStatus === 'offline'}
                >
                  {loading
                    ? <><span className="spinner" />Analysing...</>
                    : !isOnline
                    ? 'Waiting for API...'
                    : 'Classify Image'}
                </button>

                {error && (
                  <div className="error-box">
                    <svg viewBox="0 0 24 24">
                      <circle cx="12" cy="12" r="10"/>
                      <line x1="12" y1="8" x2="12" y2="12"/>
                      <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    {error}
                  </div>
                )}

              </div>
            </div>
          </div>

          {/* Right: result panel */}
          <div>
            <div className="result-panel">
              <div className="result-panel-header">
                <svg viewBox="0 0 24 24">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                <span>Prediction Result</span>
              </div>

              {!result ? (
                <div className="result-empty">
                  <div className="result-empty-icon">
                    <svg viewBox="0 0 24 24">
                      <rect x="3" y="3" width="18" height="18" rx="2"/>
                      <circle cx="8.5" cy="8.5" r="1.5"/>
                      <polyline points="21 15 16 10 5 21"/>
                    </svg>
                  </div>
                  <h3>No result yet</h3>
                  <p>Upload an image and click Classify Image to see the prediction here.</p>
                </div>
              ) : (
                <div className="result-content">
                  <div className="result-prediction-area">
                    <ConfidenceGauge pct={confidence} cls={cls} />
                    <div className={`result-label-large ${cls}`}>
                      {result.prediction}
                    </div>
                    <div className="result-sublabel">Classification result</div>
                  </div>

                  <div className="result-breakdown">
                    <div className="breakdown-row">
                      <div className="breakdown-labels">
                        <span className="breakdown-name">Cat</span>
                        <span className="breakdown-pct">{catPct.toFixed(1)}%</span>
                      </div>
                      <div className="bar-wrap">
                        <div className="bar-fill cat" style={{ width: `${catPct}%` }} />
                      </div>
                    </div>
                    <div className="breakdown-row">
                      <div className="breakdown-labels">
                        <span className="breakdown-name">Dog</span>
                        <span className="breakdown-pct">{dogPct.toFixed(1)}%</span>
                      </div>
                      <div className="bar-wrap">
                        <div className="bar-fill dog" style={{ width: `${dogPct}%` }} />
                      </div>
                    </div>
                  </div>

                  <div className="result-raw">
                    Raw score: <span>{result.raw_score}</span>
                    &nbsp;&middot;&nbsp; score &gt; 0.5 = Dog, &lt; 0.5 = Cat
                  </div>

                  <button className="btn-try-another" onClick={removeFile}>
                    <svg viewBox="0 0 24 24">
                      <polyline points="1 4 1 10 7 10"/>
                      <path d="M3.51 15a9 9 0 102.25-8.36L1 10"/>
                    </svg>
                    Try another image
                  </button>
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
