import { NavLink } from 'react-router-dom'
import { useEffect, useState } from 'react'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <nav className={`navbar${scrolled ? ' scrolled' : ''}`}>
      <div className="navbar-inner">
        {/* Brand */}
        <NavLink to="/" className="navbar-brand">
          <div className="navbar-logo">
            <svg viewBox="0 0 24 24">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div>
            <div className="navbar-title">CatsVsDogs AI</div>
            <div className="navbar-subtitle">by Kenfoguim Laurince Kinsly</div>
          </div>
        </NavLink>

        {/* Right side: links + CTA */}
        <div className="navbar-right">
          <div className="navbar-links">
            <NavLink
              to="/"
              end
              className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
            >
              <svg viewBox="0 0 24 24">
                <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>
                <polyline points="9 22 9 12 15 12 15 22"/>
              </svg>
              Home
            </NavLink>
            <NavLink
              to="/about"
              className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
            >
              <svg viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              About
            </NavLink>
            <NavLink
              to="/api"
              className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
            >
              <svg viewBox="0 0 24 24">
                <polyline points="16 18 22 12 16 6"/>
                <polyline points="8 6 2 12 8 18"/>
              </svg>
              API Docs
            </NavLink>
          </div>

          <div className="nav-divider" />

          <NavLink to="/classify" className="btn-nav-cta">
            <svg viewBox="0 0 24 24">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <path d="M3 9h18M9 21V9"/>
            </svg>
            Try Now
          </NavLink>
        </div>
      </div>
    </nav>
  )
}
