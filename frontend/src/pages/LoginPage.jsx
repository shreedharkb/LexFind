import React, { useState, useEffect } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import {
  Scale, Mail, Lock, Eye, EyeOff, ArrowRight,
  AlertCircle, CheckCircle, Search, MessageSquare, Upload,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { authApi } from '../config/apiClient';

/* ── tiny animated stat card shown in the right panel ── */
function StatCard({ icon: Icon, label, value, delay = 0 }) {
  return (
    <div
      className="flex items-center gap-3 bg-white/10 backdrop-blur-sm border border-white/10 rounded-2xl px-4 py-3"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="w-8 h-8 rounded-xl bg-amber-400/20 flex items-center justify-center flex-shrink-0">
        <Icon className="h-4 w-4 text-amber-300" />
      </div>
      <div>
        <p className="text-white font-semibold text-sm leading-none">{value}</p>
        <p className="text-white/50 text-xs mt-0.5">{label}</p>
      </div>
    </div>
  );
}

export default function LoginPage() {
  const navigate   = useNavigate();
  const location   = useLocation();
  const { login, isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      const from = location.state?.from?.pathname || '/';
      navigate(from, { replace: true });
    }
  }, [isAuthenticated]);

  const [mode, setMode]         = useState('login'); // 'login' | 'register'
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm]   = useState('');
  const [showPw, setShowPw]     = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');
  const [success, setSuccess]   = useState('');

  const switchMode = (m) => {
    setMode(m);
    setError('');
    setSuccess('');
    setPassword('');
    setConfirm('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (mode === 'register') {
      if (password.length < 8) { setError('Password must be at least 8 characters.'); return; }
      if (password !== confirm) { setError('Passwords do not match.'); return; }
    }

    setLoading(true);
    try {
      if (mode === 'register') {
        await authApi.register(email, password);
        setSuccess('Account created! Signing you in…');
        const data = await authApi.login(email, password);
        login(data.access_token, { email });
        navigate('/');
      } else {
        const data = await authApi.login(email, password);
        login(data.access_token, { email });
        const from = location.state?.from?.pathname || '/';
        navigate(from, { replace: true });
      }
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex overflow-hidden" style={{ fontFamily: 'Inter, sans-serif' }}>

      {/* ══════════════════════════════════════════
          LEFT  —  Form panel  (light beige)
      ══════════════════════════════════════════ */}
      <div
        className="flex flex-col w-full lg:w-[46%] px-8 sm:px-12 xl:px-16 py-10 overflow-y-auto"
        style={{ backgroundColor: '#EAEAE4', minHeight: '100vh' }}
      >
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 mb-12">
          <div className="bg-gray-900 p-2 rounded-xl">
            <Scale className="h-5 w-5 text-white" />
          </div>
          <span className="font-semibold text-gray-900 text-lg tracking-tight">LexFind</span>
        </Link>

        {/* Heading */}
        <div className="flex-1 flex flex-col justify-center max-w-sm w-full mx-auto lg:mx-0">
          <h1 className="font-serif-display text-3xl text-gray-900 mb-1" style={{ letterSpacing: '-0.02em' }}>
            {mode === 'login' ? 'Sign in' : 'Create account'}
          </h1>
          <p className="text-sm text-gray-500 mb-2">
            {mode === 'login' ? (
              <>Don't have an account?{' '}
                <button onClick={() => switchMode('register')}
                  className="text-gray-900 font-medium hover:underline underline-offset-2">
                  Create now
                </button>
              </>
            ) : (
              <>Already have an account?{' '}
                <button onClick={() => switchMode('login')}
                  className="text-gray-900 font-medium hover:underline underline-offset-2">
                  Sign in
                </button>
              </>
            )}
          </p>

          {/* Alerts */}
          {error && (
            <div className="flex items-start gap-2.5 bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 mt-4 text-sm">
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
              {error}
            </div>
          )}
          {success && (
            <div className="flex items-start gap-2.5 bg-green-50 border border-green-200 text-green-700 rounded-xl px-4 py-3 mt-4 text-sm">
              <CheckCircle className="h-4 w-4 mt-0.5 shrink-0" />
              {success}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4 mt-6">
            {/* Email */}
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1.5">E-mail</label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
                <input
                  id="login-email"
                  type="email" required value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="example@gmail.com"
                  className="w-full pl-10 pr-4 py-3 text-sm border border-gray-300/70 rounded-xl bg-white/70
                    focus:outline-none focus:ring-2 focus:ring-gray-900/10 focus:border-gray-400
                    placeholder:text-gray-300 transition-all"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-gray-600">Password</label>
                {mode === 'login' && (
                  <button type="button" className="text-xs text-gray-500 hover:text-gray-900 transition-colors">
                    Forgot Password?
                  </button>
                )}
              </div>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
                <input
                  id="login-password"
                  type={showPw ? 'text' : 'password'} required value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={mode === 'register' ? 'Min. 8 characters' : '••••••••'}
                  className="w-full pl-10 pr-10 py-3 text-sm border border-gray-300/70 rounded-xl bg-white/70
                    focus:outline-none focus:ring-2 focus:ring-gray-900/10 focus:border-gray-400
                    placeholder:text-gray-300 transition-all"
                />
                <button type="button" onClick={() => setShowPw(!showPw)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors">
                  {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* Confirm password (register only) */}
            {mode === 'register' && (
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1.5">Confirm password</label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
                  <input
                    id="login-confirm"
                    type={showPw ? 'text' : 'password'} required value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                    placeholder="Re-enter password"
                    className="w-full pl-10 pr-4 py-3 text-sm border border-gray-300/70 rounded-xl bg-white/70
                      focus:outline-none focus:ring-2 focus:ring-gray-900/10 focus:border-gray-400
                      placeholder:text-gray-300 transition-all"
                  />
                </div>
              </div>
            )}

            {/* Submit */}
            <button
              id="login-submit"
              type="submit" disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-gray-900 text-white
                py-3 rounded-xl text-sm font-medium hover:bg-gray-700 active:scale-[0.99]
                transition-all disabled:opacity-50 disabled:cursor-not-allowed mt-1"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity=".25"/>
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
                  </svg>
                  {mode === 'login' ? 'Signing in…' : 'Creating account…'}
                </span>
              ) : (
                <>
                  {mode === 'login' ? 'Sign in' : 'Create account'}
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>

          <p className="text-xs text-gray-400 text-center mt-8">
            By continuing, you agree to our{' '}
            <a href="#terms" className="underline hover:text-gray-600 transition-colors">Terms</a>
            {' '}and{' '}
            <a href="#privacy" className="underline hover:text-gray-600 transition-colors">Privacy Policy</a>.
          </p>
        </div>
      </div>

      {/* ══════════════════════════════════════════
          RIGHT  —  Branded panel  (dark)
      ══════════════════════════════════════════ */}
      <div
        className="hidden lg:flex lg:w-[54%] flex-col relative overflow-hidden"
        style={{ background: 'linear-gradient(145deg, #161616 0%, #0d0d0d 60%, #1a1200 100%)' }}
      >
        {/* Amber glow orb */}
        <div
          className="absolute top-[-80px] right-[-80px] w-[420px] h-[420px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at center, rgba(251,191,36,0.18) 0%, rgba(245,158,11,0.08) 50%, transparent 70%)',
            filter: 'blur(40px)',
          }}
        />
        {/* Bottom glow */}
        <div
          className="absolute bottom-[-60px] left-1/2 -translate-x-1/2 w-[500px] h-[300px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at center, rgba(251,191,36,0.08) 0%, transparent 70%)',
            filter: 'blur(50px)',
          }}
        />

        {/* Content */}
        <div className="relative z-10 flex flex-col h-full px-14 py-12">

          {/* Top label */}
          <div className="flex items-center gap-2 mb-auto">
            <span className="text-white/30 text-xs font-medium tracking-widest uppercase">Legal Intelligence</span>
          </div>

          {/* Main card — the "feature showcase" */}
          <div className="my-auto">
            {/* Big card */}
            <div
              className="rounded-3xl p-7 mb-6 border border-white/8"
              style={{
                background: 'linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%)',
                backdropFilter: 'blur(20px)',
              }}
            >
              {/* Case preview mockup */}
              <div className="space-y-2.5 mb-6">
                {[
                  { title: 'Kesavananda Bharati v. State of Kerala', tag: 'Constitutional · 1973' },
                  { title: 'Maneka Gandhi v. Union of India',         tag: 'Fundamental Rights · 1978' },
                  { title: 'Vishaka v. State of Rajasthan',           tag: 'Labour Law · 1997' },
                ].map((c, i) => (
                  <div
                    key={c.title}
                    className="flex items-center gap-3 bg-white/5 rounded-xl px-4 py-3 border border-white/5"
                    style={{ opacity: 1 - i * 0.15 }}
                  >
                    <div className="w-7 h-7 rounded-lg bg-amber-400/15 flex items-center justify-center flex-shrink-0">
                      <Scale className="h-3.5 w-3.5 text-amber-400" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-white text-xs font-medium truncate">{c.title}</p>
                      <p className="text-white/40 text-[10px] mt-0.5">{c.tag}</p>
                    </div>
                    <span className="text-white/20 text-xs ml-auto flex-shrink-0">→</span>
                  </div>
                ))}
              </div>

              <h2
                className="font-serif-display text-white text-3xl leading-snug mb-3"
                style={{ letterSpacing: '-0.02em' }}
              >
                Research smarter,<br />decide faster.
              </h2>
              <p className="text-white/40 text-sm leading-relaxed">
                Search 46,000+ cases, analyze private documents with RAG AI, and keep a persistent legal research trail.
              </p>
            </div>

            {/* Stat cards row */}
            <div className="grid grid-cols-3 gap-3">
              <StatCard icon={Search}       value="46k+"  label="Cases indexed"    delay={0}   />
              <StatCard icon={MessageSquare} value="RAG"   label="AI-powered Q&A"  delay={100} />
              <StatCard icon={Upload}        value="100%"  label="Private & secure" delay={200} />
            </div>
          </div>

          {/* Bottom quote */}
          <div className="mt-auto pt-8">
            <p className="text-white/20 text-xs italic leading-relaxed border-t border-white/8 pt-6">
              "Justice delayed is justice denied." — William Ewart Gladstone
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
