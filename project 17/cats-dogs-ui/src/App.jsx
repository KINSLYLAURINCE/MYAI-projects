import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { useEffect } from 'react'
import './App.css'

const API_BASE = 'https://cats-dogs-api-vkit.onrender.com'
const PING_INTERVAL = 10 * 60 * 1000  // 10 minutes

function KeepAlive() {
  useEffect(() => {
    const ping = () => fetch(`${API_BASE}/health`).catch(() => {})
    ping()
    const id = setInterval(ping, PING_INTERVAL)
    return () => clearInterval(id)
  }, [])
  return null
}
import Navbar    from './components/Navbar'
import MobileBar from './components/MobileBar'
import Home      from './pages/Home'
import Classify  from './pages/Classify'
import About     from './pages/About'
import ApiDocs   from './pages/ApiDocs'

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-main">
        {/* Column 1: Brand + description */}
        <div>
          <div className="footer-brand">
            <div className="footer-logo">
              <svg viewBox="0 0 24 24">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
              </svg>
            </div>
            <span className="footer-brand-name">CatsVsDogs AI</span>
          </div>
          <p className="footer-desc">
            A high-accuracy deep learning classifier built with EfficientNetB4
            and 3-phase transfer learning. Achieves 97.92% accuracy on the
            cats vs dogs classification task.
          </p>
          <div className="footer-credit">
            Developed by <strong>Kenfoguim Laurince Kinsly</strong>
          </div>
        </div>

        {/* Column 2: Navigation */}
        <div>
          <div className="footer-col-title">Navigation</div>
          <nav className="footer-links">
            <NavLink to="/" className="footer-link" end>
              <svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
              Home
            </NavLink>
            <NavLink to="/classify" className="footer-link">
              <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
              Classify Image
            </NavLink>
            <NavLink to="/about" className="footer-link">
              <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              About the Model
            </NavLink>
            <NavLink to="/api" className="footer-link">
              <svg viewBox="0 0 24 24"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
              API Docs
            </NavLink>
          </nav>
        </div>

        {/* Column 3: Tech stack */}
        <div>
          <div className="footer-col-title">Tech Stack</div>
          <div className="footer-tech">
            <span className="chip">EfficientNetB4</span>
            <span className="chip">TensorFlow</span>
            <span className="chip">FastAPI</span>
            <span className="chip">React</span>
            <span className="chip">Transfer Learning</span>
            <span className="chip">3-Phase Fine-Tuning</span>
            <span className="chip">TTA x5</span>
            <span className="chip">97.92% Accuracy</span>
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="footer-bottom">
        <span className="footer-copyright">
          &copy; {new Date().getFullYear()} &mdash; Developed by <strong>Kenfoguim Laurince Kinsly</strong>. All rights reserved.
        </span>
      </div>
    </footer>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <KeepAlive />
      <Navbar />
      <Routes>
        <Route path="/"         element={<Home />} />
        <Route path="/classify" element={<Classify />} />
        <Route path="/about"    element={<About />} />
        <Route path="/api"      element={<ApiDocs />} />
      </Routes>
      <Footer />
      <MobileBar />
    </BrowserRouter>
  )
}
