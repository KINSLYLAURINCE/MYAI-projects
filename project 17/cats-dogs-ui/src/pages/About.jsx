import { useEffect, useRef } from 'react'

/* Animated progress bar using IntersectionObserver */
function AnimatedBar({ pct, cls }) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.style.width = pct + '%'
          observer.disconnect()
        }
      },
      { threshold: 0.3 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [pct])

  return (
    <div className="tl-progress">
      <div
        ref={ref}
        className={`tl-progress-fill${cls ? ' ' + cls : ''}`}
        style={{ width: '0%', transition: 'width 0.9s cubic-bezier(0.4,0,0.2,1)' }}
      />
    </div>
  )
}

export default function About() {
  return (
    <div className="page">
      <div className="container">

        <div className="about-header">
          <h1>About the Model</h1>
          <p>Architecture, training strategy, performance metrics, and model card.</p>
        </div>

        <div className="about-grid">
          {/* Performance metrics */}
          <div className="about-card">
            <h2>
              <svg viewBox="0 0 24 24">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              Performance Metrics
            </h2>
            <div className="metric-list">
              <div className="metric-row">
                <span className="metric-key">Final Accuracy</span>
                <span className="metric-val accent">99%+</span>
              </div>
              <div className="metric-row">
                <span className="metric-key">Starting Accuracy</span>
                <span className="metric-val">77.7%</span>
              </div>
              <div className="metric-row">
                <span className="metric-key">Base Model</span>
                <span className="metric-val">EfficientNetB4</span>
              </div>
              <div className="metric-row">
                <span className="metric-key">Parameters</span>
                <span className="metric-val">19M</span>
              </div>
              <div className="metric-row">
                <span className="metric-key">Input Size</span>
                <span className="metric-val">224 x 224</span>
              </div>
              <div className="metric-row">
                <span className="metric-key">Training Images</span>
                <span className="metric-val">8,000+</span>
              </div>
              <div className="metric-row">
                <span className="metric-key">Pretrained On</span>
                <span className="metric-val">ImageNet (1.4M)</span>
              </div>
            </div>
          </div>

          {/* Accuracy journey */}
          <div className="about-card">
            <h2>
              <svg viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
              Accuracy Journey
            </h2>
            <div className="timeline">

              <div className="tl-item">
                <div className="tl-left">
                  <div className="tl-dot">V1</div>
                  <div className="tl-connector" />
                </div>
                <div className="tl-content">
                  <div className="tl-title">Plain CNN from scratch</div>
                  <div className="tl-desc">
                    3 Conv layers, 64x64 input, 10 epochs, no callbacks or augmentation.
                  </div>
                  <div className="tl-acc-row">
                    <span className="tl-acc">77.7%</span>
                    <AnimatedBar pct={77.7} />
                  </div>
                </div>
              </div>

              <div className="tl-item">
                <div className="tl-left">
                  <div className="tl-dot">V2</div>
                  <div className="tl-connector" />
                </div>
                <div className="tl-content">
                  <div className="tl-title">MobileNetV2 transfer learning</div>
                  <div className="tl-desc">
                    160x160 input, 2-phase training, EarlyStopping, ReduceLROnPlateau.
                  </div>
                  <div className="tl-acc-row">
                    <span className="tl-acc">97.92%</span>
                    <AnimatedBar pct={97.92} />
                  </div>
                </div>
              </div>

              <div className="tl-item">
                <div className="tl-left">
                  <div className="tl-dot">V3</div>
                  <div className="tl-connector" />
                </div>
                <div className="tl-content">
                  <div className="tl-title">EfficientNetB4 + full fine-tune</div>
                  <div className="tl-desc">
                    224x224 input, 3-phase progressive training, TTA x5, deeper head.
                  </div>
                  <div className="tl-acc-row">
                    <span className="tl-acc">99%+</span>
                    <AnimatedBar pct={99} />
                  </div>
                </div>
              </div>

            </div>
          </div>
        </div>

        {/* Training Architecture */}
        <div className="about-card-full">
          <h2>
            <svg viewBox="0 0 24 24">
              <rect x="2" y="3" width="20" height="14" rx="2"/>
              <line x1="8" y1="21" x2="16" y2="21"/>
              <line x1="12" y1="17" x2="12" y2="21"/>
            </svg>
            Training Architecture
          </h2>
          <div className="arch-grid">
            <div className="arch-item">
              <div className="arch-phase">Phase 1</div>
              <div className="arch-name">Head Only</div>
              <div className="arch-detail">
                Base model frozen. Train custom Dense head: Dense(512) &rarr; Dense(256) &rarr; Dense(128) &rarr; Sigmoid.
              </div>
              <div className="arch-hover-detail">
                LR 1e-3, up to 10 epochs. Adam optimizer. BatchNormalization after GlobalAveragePooling. Dropout(0.4, 0.3, 0.2).
              </div>
            </div>
            <div className="arch-item">
              <div className="arch-phase">Phase 2</div>
              <div className="arch-name">Top 80 Layers</div>
              <div className="arch-detail">
                Unfreeze top 80 EfficientNetB4 layers. LR 1e-5, up to 20 epochs.
              </div>
              <div className="arch-hover-detail">
                Careful fine-tuning with ReduceLROnPlateau and EarlyStopping(patience=5). Best weights restored automatically.
              </div>
            </div>
            <div className="arch-item">
              <div className="arch-phase">Phase 3</div>
              <div className="arch-name">Full Fine-Tune</div>
              <div className="arch-detail">
                All layers trainable. LR 1e-6 (ultra-slow). Micro-polishing to squeeze out final errors.
              </div>
              <div className="arch-hover-detail">
                Very low learning rate prevents catastrophic forgetting. EarlyStopping(patience=3). This phase typically adds 1-2% accuracy.
              </div>
            </div>
          </div>
        </div>

        {/* Key Techniques */}
        <div className="about-card-full">
          <h2>
            <svg viewBox="0 0 24 24">
              <path d="M12 20h9"/>
              <path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/>
            </svg>
            Key Techniques Used
          </h2>
          <div className="arch-grid">
            <div className="arch-item">
              <div className="arch-phase">Regularization</div>
              <div className="arch-name">Dropout + BatchNorm</div>
              <div className="arch-detail">
                Dropout(0.4, 0.3, 0.2) at each Dense layer. BatchNorm after pooling to stabilize training.
              </div>
              <div className="arch-hover-detail">
                Prevents overfitting dramatically. Without these, V3 overfit to ~95% train / ~88% val.
              </div>
            </div>
            <div className="arch-item">
              <div className="arch-phase">Augmentation</div>
              <div className="arch-name">8 Techniques</div>
              <div className="arch-detail">
                Rotation, shift, shear, zoom, flip, brightness, channel shift. Prevents overfitting on 8K images.
              </div>
              <div className="arch-hover-detail">
                Applied on-the-fly during training using Keras ImageDataGenerator. Effectively multiplies dataset size.
              </div>
            </div>
            <div className="arch-item">
              <div className="arch-phase">Inference</div>
              <div className="arch-name">TTA x5</div>
              <div className="arch-detail">
                Test-Time Augmentation: each image predicted 5 times with slight augmentation, results averaged.
              </div>
              <div className="arch-hover-detail">
                Reduces variance and boosts accuracy by ~0.5-1% on borderline cases. Mean of 5 sigmoid outputs.
              </div>
            </div>
          </div>
        </div>

        {/* Model Card */}
        <div className="about-card-full">
          <h2>
            <svg viewBox="0 0 24 24">
              <rect x="2" y="3" width="20" height="18" rx="2"/>
              <path d="M8 7h8M8 11h8M8 15h4"/>
            </svg>
            Model Card
          </h2>
          <div className="model-card-grid">
            <div className="model-card-item">
              <div className="model-card-label">Task</div>
              <div className="model-card-value">Binary Image Classification</div>
            </div>
            <div className="model-card-item">
              <div className="model-card-label">Model Type</div>
              <div className="model-card-value">Convolutional Neural Network</div>
            </div>
            <div className="model-card-item">
              <div className="model-card-label">Base Architecture</div>
              <div className="model-card-value accent">EfficientNetB4</div>
            </div>
            <div className="model-card-item">
              <div className="model-card-label">Framework</div>
              <div className="model-card-value">TensorFlow 2 / Keras</div>
            </div>
            <div className="model-card-item">
              <div className="model-card-label">Format</div>
              <div className="model-card-value">.keras (SavedModel)</div>
            </div>
            <div className="model-card-item">
              <div className="model-card-label">Input Shape</div>
              <div className="model-card-value">224 x 224 x 3 (RGB)</div>
            </div>
            <div className="model-card-item">
              <div className="model-card-label">Output</div>
              <div className="model-card-value">Sigmoid (0 = Cat, 1 = Dog)</div>
            </div>
            <div className="model-card-item">
              <div className="model-card-label">License</div>
              <div className="model-card-value">MIT (Research Use)</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
