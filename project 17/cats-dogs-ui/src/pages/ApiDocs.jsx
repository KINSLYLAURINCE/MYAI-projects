import { useState, useRef, useEffect } from 'react'

function C({ t, children }) {
  return <span className={`c-${t}`}>{children}</span>
}

function CopyButton({ getText }) {
  const [label, setLabel] = useState('Copy')

  const handleCopy = () => {
    const text = getText()
    navigator.clipboard.writeText(text).then(() => {
      setLabel('Copied!')
      setTimeout(() => setLabel('Copy'), 2000)
    }).catch(() => {
      setLabel('Failed')
      setTimeout(() => setLabel('Copy'), 2000)
    })
  }

  return (
    <button
      className={`copy-btn${label === 'Copied!' ? ' copied' : ''}`}
      onClick={handleCopy}
    >
      {label}
    </button>
  )
}

function CodeBlock({ children, text }) {
  return (
    <div className="code-block-wrap">
      <div className="code-block">
        <pre>{children}</pre>
      </div>
      <CopyButton getText={() => text || ''} />
    </div>
  )
}

function RequestResponseBlock({ requestChildren, requestText, responseChildren }) {
  const [tab, setTab] = useState('request')
  return (
    <div>
      <div className="endpoint-tabs">
        <button
          className={`endpoint-tab${tab === 'request' ? ' active' : ''}`}
          onClick={() => setTab('request')}
        >
          Request
        </button>
        <button
          className={`endpoint-tab${tab === 'response' ? ' active' : ''}`}
          onClick={() => setTab('response')}
        >
          Response
        </button>
      </div>
      {tab === 'request' && (
        <CodeBlock text={requestText}>{requestChildren}</CodeBlock>
      )}
      {tab === 'response' && responseChildren}
    </div>
  )
}

const BASE_URL = 'https://cats-dogs-api-vkit.onrender.com'

/* Scroll spy: track which endpoint is active */
function useScrollSpy(ids) {
  const [activeId, setActiveId] = useState(ids[0])
  useEffect(() => {
    const handler = () => {
      let found = ids[0]
      for (const id of ids) {
        const el = document.getElementById(id)
        if (el) {
          const rect = el.getBoundingClientRect()
          if (rect.top <= 100) found = id
        }
      }
      setActiveId(found)
    }
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [ids])
  return activeId
}

const ENDPOINTS = ['health', 'predict', 'reload']

export default function ApiDocs() {
  const activeId = useScrollSpy(ENDPOINTS)

  const scrollTo = (id) => {
    const el = document.getElementById(id)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const [baseUrlCopied, setBaseUrlCopied] = useState(false)
  const copyBaseUrl = () => {
    navigator.clipboard.writeText(BASE_URL).then(() => {
      setBaseUrlCopied(true)
      setTimeout(() => setBaseUrlCopied(false), 2000)
    })
  }

  return (
    <div className="page">
      <div className="container">
        <div className="api-page-layout">

          {/* Sidebar */}
          <aside className="api-sidebar">
            <div className="sidebar-label">Endpoints</div>
            {[
              { id: 'health',  method: 'get',  label: '/health'  },
              { id: 'predict', method: 'post', label: '/predict' },
              { id: 'reload',  method: 'post', label: '/reload'  },
            ].map(({ id, method, label }) => (
              <div
                key={id}
                className={`sidebar-link${activeId === id ? ' active' : ''}`}
                onClick={() => scrollTo(id)}
              >
                <span className={`sidebar-method ${method}`}>{method.toUpperCase()}</span>
                {label}
              </div>
            ))}
          </aside>

          {/* Main content */}
          <main className="api-main">
            <div className="api-header">
              <h1>API Reference</h1>
              <p>The FastAPI backend exposes three endpoints for model inference and management.</p>
            </div>

            {/* Base URL banner */}
            <div className="base-url-banner">
              <span className="base-url-badge">Base URL</span>
              <span className="base-url-value">{BASE_URL}</span>
              <button
                className="base-url-copy"
                onClick={copyBaseUrl}
              >
                {baseUrlCopied ? 'Copied!' : 'Copy'}
              </button>
            </div>

            {/* GET /health */}
            <div id="health" className="endpoint-card">
              <div className="endpoint-head">
                <span className="method get">GET</span>
                <span className="endpoint-path">/health</span>
                <span className="endpoint-desc">Check model status</span>
              </div>
              <div className="endpoint-body">
                <RequestResponseBlock
                  requestText={`curl https://cats-dogs-api-vkit.onrender.com/health`}
                  requestChildren={
                    <>
                      <C t="muted"># Request{'\n'}</C>
                      <C t="green">curl</C>{' https://cats-dogs-api-vkit.onrender.com/health'}
                    </>
                  }
                  responseChildren={
                    <div className="response-example">
                      <div className="response-label">
                        <span className="response-status ok">200</span>
                        OK
                      </div>
                      <div className="code-block" style={{ borderRadius: '0 0 8px 8px' }}>
                        <pre>
                          {'{\n  '}
                          <C t="yellow">"model_loaded"</C>{': '}
                          <C t="blue">true</C>{',\n  '}
                          <C t="yellow">"status"</C>{':      '}
                          <C t="green">"Model loaded and ready."</C>{',\n  '}
                          <C t="yellow">"model_path"</C>{': '}
                          <C t="green">"H:/AI/project 17/cats_dogs_model.keras"</C>{',\n  '}
                          <C t="yellow">"file_exists"</C>{': '}
                          <C t="blue">true</C>
                          {'\n}'}
                        </pre>
                      </div>
                    </div>
                  }
                />
              </div>
            </div>

            {/* POST /predict */}
            <div id="predict" className="endpoint-card">
              <div className="endpoint-head">
                <span className="method post">POST</span>
                <span className="endpoint-path">/predict</span>
                <span className="endpoint-desc">Classify an image</span>
              </div>
              <div className="endpoint-body">
                <RequestResponseBlock
                  requestText={`curl -X POST https://cats-dogs-api-vkit.onrender.com/predict \\\n     -F "file=@my_dog.jpg"`}
                  requestChildren={
                    <>
                      <C t="muted"># cURL{'\n'}</C>
                      <C t="green">curl</C>{' -X POST https://cats-dogs-api-vkit.onrender.com/predict \\\n     -F '}
                      <C t="yellow">"file=@my_dog.jpg"</C>
                      {'\n\n'}
                      <C t="muted"># Python requests{'\n'}</C>
                      <C t="purple">import</C>{' requests\n\n'}
                      {'res = requests.'}
                      <C t="green">post</C>
                      {'(\n    '}
                      <C t="yellow">"https://cats-dogs-api-vkit.onrender.com/predict"</C>
                      {',\n    files={'}
                      <C t="yellow">"file"</C>
                      {': '}
                      <C t="purple">open</C>
                      {'('}
                      <C t="yellow">"dog.jpg"</C>
                      {', '}
                      <C t="yellow">"rb"</C>
                      {')}\n)\n'}
                      <C t="purple">print</C>
                      {'(res.'}
                      <C t="green">json</C>
                      {'())'}
                    </>
                  }
                  responseChildren={
                    <>
                      <div className="response-example">
                        <div className="response-label">
                          <span className="response-status ok">200</span>
                          OK
                        </div>
                        <div className="code-block" style={{ borderRadius: '0 0 8px 8px' }}>
                          <pre>
                            {'{\n  '}
                            <C t="yellow">"prediction"</C>{':     '}
                            <C t="green">"Dog"</C>{',\n  '}
                            <C t="yellow">"confidence"</C>{':     '}
                            <C t="green">"98.74%"</C>{',\n  '}
                            <C t="yellow">"raw_score"</C>{':      '}
                            <C t="blue">0.987421</C>{',\n  '}
                            <C t="yellow">"interpretation"</C>{': '}
                            <C t="green">"score &gt; 0.5 = Dog, score &lt; 0.5 = Cat"</C>
                            {'\n}'}
                          </pre>
                        </div>
                      </div>
                      <div className="response-example">
                        <div className="response-label">
                          <span className="response-status err">503</span>
                          Model not ready
                        </div>
                        <div className="code-block" style={{ borderRadius: '0 0 8px 8px' }}>
                          <pre>
                            {'{ '}
                            <C t="yellow">"detail"</C>
                            {': '}
                            <C t="red">"Model is not ready yet. Check /health for status."</C>
                            {' }'}
                          </pre>
                        </div>
                      </div>
                    </>
                  }
                />
              </div>
            </div>

            {/* POST /reload */}
            <div id="reload" className="endpoint-card">
              <div className="endpoint-head">
                <span className="method post">POST</span>
                <span className="endpoint-path">/reload</span>
                <span className="endpoint-desc">Force reload model from disk</span>
              </div>
              <div className="endpoint-body">
                <RequestResponseBlock
                  requestText={`curl -X POST https://cats-dogs-api-vkit.onrender.com/reload`}
                  requestChildren={
                    <>
                      <C t="muted"># cURL{'\n'}</C>
                      <C t="green">curl</C>{' -X POST https://cats-dogs-api-vkit.onrender.com/reload'}
                    </>
                  }
                  responseChildren={
                    <div className="response-example">
                      <div className="response-label">
                        <span className="response-status ok">200</span>
                        OK
                      </div>
                      <div className="code-block" style={{ borderRadius: '0 0 8px 8px' }}>
                        <pre>
                          {'{ '}
                          <C t="yellow">"status"</C>
                          {': '}
                          <C t="green">"Model loaded and ready."</C>
                          {', '}
                          <C t="yellow">"loaded"</C>
                          {': '}
                          <C t="blue">true</C>
                          {' }'}
                        </pre>
                      </div>
                    </div>
                  }
                />
              </div>
            </div>

          </main>
        </div>
      </div>
    </div>
  )
}
