import { NavLink } from 'react-router-dom'

export default function MobileBar() {
  return (
    <div className="mobile-bar">
      <NavLink to="/" className={({ isActive }) => 'mob-link' + (isActive ? ' active' : '')} end>
        <svg viewBox="0 0 24 24">
          <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>
          <polyline points="9 22 9 12 15 12 15 22"/>
        </svg>
        Home
      </NavLink>

      <NavLink to="/classify" className={({ isActive }) => 'mob-link' + (isActive ? ' active' : '')}>
        <svg viewBox="0 0 24 24">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <path d="M3 9h18M9 21V9"/>
        </svg>
        Classify
      </NavLink>

      <NavLink to="/about" className={({ isActive }) => 'mob-link' + (isActive ? ' active' : '')}>
        <svg viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        About
      </NavLink>

      <NavLink to="/api" className={({ isActive }) => 'mob-link' + (isActive ? ' active' : '')}>
        <svg viewBox="0 0 24 24">
          <polyline points="16 18 22 12 16 6"/>
          <polyline points="8 6 2 12 8 18"/>
        </svg>
        API
      </NavLink>
    </div>
  )
}
