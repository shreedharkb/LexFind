import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Scale, Menu, X, LogOut, MessageSquare, ChevronDown, User } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Navigation() {
  const location  = useLocation();
  const navigate  = useNavigate();
  const { isAuthenticated, user, logout } = useAuth();
  const [isMobileMenuOpen, setMobileOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen]   = useState(false);

  const navLinks = [
    { path: '/search',    label: 'Search'    },
    { path: '/assistant', label: 'Assistant' },
  ];

  // Only show nav links when authenticated; unauthenticated users just see Sign in / Get started
  const links = isAuthenticated ? navLinks : [];

  const handleLogout = () => {
    logout();
    setUserMenuOpen(false);
    navigate('/');
  };

  const isActive = (path) =>
    location.pathname === path || location.pathname.startsWith(path + '/');

  return (
    <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200/60 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex items-center justify-between h-14">

          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <div className="bg-gray-900 p-1.5 rounded-lg">
              <Scale className="h-4 w-4 text-white" />
            </div>
            <span className="text-base font-semibold text-gray-900 tracking-tight">LexFind</span>
          </Link>

          {/* Desktop links */}
          <div className="hidden md:flex items-center gap-1">
            {links.map((link) => (
              <Link key={link.path} to={link.path}
                className={`px-3.5 py-1.5 text-sm rounded-lg transition-colors ${
                  isActive(link.path)
                    ? 'text-gray-900 font-medium bg-gray-100'
                    : 'text-gray-500 hover:text-gray-900 hover:bg-gray-50'
                }`}>
                {link.label}
              </Link>
            ))}
          </div>

          {/* Right side */}
          <div className="hidden md:flex items-center gap-2">
            {isAuthenticated ? (
              /* User avatar + dropdown */
              <div className="relative">
                <button onClick={() => setUserMenuOpen(!userMenuOpen)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-xl
                    hover:bg-gray-100 transition-colors text-sm text-gray-700">
                  <div className="w-6 h-6 bg-gray-900 rounded-full flex items-center justify-center">
                    <User className="h-3 w-3 text-white" />
                  </div>
                  <span className="max-w-[120px] truncate text-gray-700 font-medium">
                    {user?.email?.split('@')[0] ?? 'Account'}
                  </span>
                  <ChevronDown className={`h-3.5 w-3.5 text-gray-400 transition-transform ${userMenuOpen ? 'rotate-180' : ''}`} />
                </button>

                {userMenuOpen && (
                  <>
                    {/* Backdrop */}
                    <div className="fixed inset-0 z-10" onClick={() => setUserMenuOpen(false)} />
                    {/* Dropdown */}
                    <div className="absolute right-0 mt-1.5 w-52 bg-white border border-gray-200
                      rounded-xl shadow-lg z-20 overflow-hidden py-1">
                      <div className="px-4 py-2.5 border-b border-gray-100">
                        <p className="text-xs text-gray-400">Signed in as</p>
                        <p className="text-sm font-medium text-gray-900 truncate">{user?.email}</p>
                      </div>
                      <Link to="/assistant" onClick={() => setUserMenuOpen(false)}
                        className="flex items-center gap-2.5 px-4 py-2 text-sm text-gray-700
                          hover:bg-gray-50 transition-colors">
                        <MessageSquare className="h-4 w-4 text-gray-400" />
                        My Assistant
                      </Link>
                      <button onClick={handleLogout}
                        className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-red-600
                          hover:bg-red-50 transition-colors">
                        <LogOut className="h-4 w-4" />
                        Sign out
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              /* Unauthenticated: Login + Get started */
              <>
                <Link to="/login"
                  className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1.5 rounded-lg
                    hover:bg-gray-50 transition-colors font-medium">
                  Sign in
                </Link>
              </>
            )}
          </div>

          {/* Mobile menu button */}
          <button className="md:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
            onClick={() => setMobileOpen(!isMobileMenuOpen)}>
            {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>

        {/* Mobile menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden py-3 border-t border-gray-100 space-y-1">
            {links.map((link) => (
              <Link key={link.path} to={link.path}
                className={`block px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive(link.path)
                    ? 'text-gray-900 font-medium bg-gray-100'
                    : 'text-gray-500 hover:text-gray-900 hover:bg-gray-50'
                }`}
                onClick={() => setMobileOpen(false)}>
                {link.label}
              </Link>
            ))}
            <div className="pt-2 border-t border-gray-100 space-y-1">
              {isAuthenticated ? (
                <button onClick={() => { handleLogout(); setMobileOpen(false); }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600
                    hover:bg-red-50 rounded-lg transition-colors">
                  <LogOut className="h-4 w-4" />
                  Sign out ({user?.email?.split('@')[0]})
                </button>
              ) : (
                <>
                  <Link to="/login" onClick={() => setMobileOpen(false)}
                    className="block px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                    Sign in
                  </Link>
                  <Link to="/login" onClick={() => setMobileOpen(false)}
                    className="block px-3 py-2 text-sm bg-gray-900 text-white rounded-full text-center font-medium">
                    Get started
                  </Link>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
