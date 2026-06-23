import { Link } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'

/* Animated counter hook using IntersectionObserver */
function useCountUp(target, duration = 1400) {
  const [value, setValue] = useState(0)
  const ref = useRef(null)
  const started = useRef(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true
          const start = performance.now()
          const tick = (now) => {
            const elapsed = now - start
            const progress = Math.min(elapsed / duration, 1)
            // ease-out
            const eased = 1 - Math.pow(1 - progress, 3)
            setValue(Math.round(eased * target))
            if (progress < 1) requestAnimationFrame(tick)
          }
          requestAnimationFrame(tick)
        }
      },
      { threshold: 0.3 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [target, duration])

  return [value, ref]
}

function StatCard({ iconPath, value, suffix, label, accent }) {
  const [count, ref] = useCountUp(value)
  return (
    <div className="stat-card" ref={ref}>
      <div className="stat-icon">
        <svg viewBox="0 0 24 24">
          {iconPath}
        </svg>
      </div>
      <div className="stat-value">
        {accent ? <span className="accent">{count}{suffix}</span> : <>{count}{suffix}</>}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

export default function Home() {
  return (
    <div className="page">
      <div className="container">

        {/* Hero */}
        <div className="hero">
          <div className="hero-bg" aria-hidden="true" />
          <div className="hero-blob hero-blob-1" aria-hidden="true" />
          <div className="hero-blob hero-blob-2" aria-hidden="true" />

          <div className="hero-content">
            <div className="hero-tag">
              <div className="hero-tag-dot" />
              Deep Learning Classifier
            </div>

            <h1>
              Identify <span>Cats</span> and<br />
              Dogs Instantly
            </h1>

            <p className="hero-desc">
              Upload any image and our EfficientNetB4 model will classify it in
              under a second with 97.92% accuracy, powered by 3-phase
              transfer learning on ImageNet.
            </p>

            <div className="hero-actions">
              <Link to="/classify" className="btn-primary">
                <svg viewBox="0 0 24 24">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                Try the Classifier
              </Link>
              <Link to="/about" className="btn-secondary">
                <svg viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                How It Works
              </Link>
            </div>

            <p className="hero-hint">No account needed &mdash; free to use</p>
          </div>
        </div>

        {/* Tech stack row */}
        <div className="stack-row">
          <span className="stack-label">Built with</span>
          <div className="stack-items">
            <div className="stack-item">
              <svg viewBox="0 0 24 24">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
              </svg>
              EfficientNetB4
            </div>
            <div className="stack-item">
              <svg viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
              </svg>
              TensorFlow
            </div>
            <div className="stack-item">
              <svg viewBox="0 0 24 24">
                <polyline points="16 18 22 12 16 6"/>
                <polyline points="8 6 2 12 8 18"/>
              </svg>
              FastAPI
            </div>
            <div className="stack-item">
              <svg viewBox="0 0 24 24">
                <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2z"/>
                <path d="M8 12h8M12 8v8"/>
              </svg>
              React + Vite
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="stats-row">
          <StatCard
            iconPath={<><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></>}
            value={97.92}
            suffix="%+"
            label="Final Accuracy"
            accent
          />
          <StatCard
            iconPath={<><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></>}
            value={8}
            suffix="K+"
            label="Training Images"
          />
          <StatCard
            iconPath={<><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></>}
            value={19}
            suffix="M"
            label="Parameters"
          />
        </div>

        {/* Features */}
        <div>
          <div className="section-header">
            <div className="section-tag">Capabilities</div>
            <div className="section-title">Everything you need</div>
            <p className="section-sub">
              A complete pipeline from raw image upload to confident prediction,
              backed by state-of-the-art transfer learning.
            </p>
          </div>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10"/>
                  <polyline points="12 6 12 12 16 14"/>
                </svg>
              </div>
              <h3>Fast Inference</h3>
              <p>
                Predictions returned in under one second using optimized
                TensorFlow serving with TTA averaging.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
              </div>
              <h3>High Accuracy</h3>
              <p>
                97.92% accuracy achieved through 3-phase fine-tuning of
                EfficientNetB4 pretrained on ImageNet&apos;s 1.4M images.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <polyline points="16 18 22 12 16 6"/>
                  <polyline points="8 6 2 12 8 18"/>
                </svg>
              </div>
              <h3>REST API</h3>
              <p>
                Clean FastAPI endpoints with JSON responses. Integrate into
                any app with a single HTTP POST request.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                  <path d="M2 17l10 5 10-5"/>
                  <path d="M2 12l10 5 10-5"/>
                </svg>
              </div>
              <h3>Transfer Learning</h3>
              <p>
                Leverages 19M pretrained parameters from EfficientNetB4,
                progressively unfrozen across three training phases.
              </p>
            </div>
          </div>
        </div>

        {/* How it works */}
        <div>
          <div className="section-header">
            <div className="section-tag">Process</div>
            <div className="section-title">How it works</div>
            <p className="section-sub">Three steps from image to confident prediction</p>
          </div>
          <div className="steps-grid">
            <div className="step-card">
              <div className="step-num">1</div>
              <h3>Upload an Image</h3>
              <p>
                Drag and drop or click to select any JPG, PNG, or WEBP image.
                The dropzone accepts files up to 10 MB.
              </p>
            </div>
            <div className="step-card">
              <div className="step-num">2</div>
              <h3>AI Processing</h3>
              <p>
                The image is resized to 224x224, normalized, and passed through
                EfficientNetB4 five times with slight augmentation (TTA x5).
              </p>
            </div>
            <div className="step-card">
              <div className="step-num">3</div>
              <h3>Get Results</h3>
              <p>
                The averaged prediction is returned with a confidence score
                and a raw probability value, displayed instantly.
              </p>
            </div>
          </div>
        </div>

        {/* CTA Banner */}
        <div className="cta-banner">
          <div className="cta-banner-inner">
            <h2>Ready to classify your first image?</h2>
            <p>
              Upload any photo of a cat or dog and get an instant prediction
              powered by state-of-the-art deep learning.
            </p>
            <Link to="/classify" className="btn-white">
              <svg viewBox="0 0 24 24">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              Start Classifying
            </Link>
          </div>
        </div>

      </div>
    </div>
  )
}
