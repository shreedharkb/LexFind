import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, MessageSquare, Upload, FileText, Scale, Files } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

function LandingPage() {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  return (
    <div
      className="min-h-screen"
      style={{ backgroundColor: '#EAEAE4' }}
    >
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Warm radial gradient blob in the background */}
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/3 w-[700px] h-[400px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at center, rgba(255,210,170,0.45) 0%, rgba(255,200,140,0.2) 45%, transparent 70%)',
            filter: 'blur(40px)',
          }}
        />

        <div className="relative max-w-5xl mx-auto px-4 sm:px-6 pt-14 sm:pt-20 pb-10 text-center">
          {/* Headline */}
          <h1
            className="font-serif-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl text-gray-900 leading-tight mb-5"
            style={{ letterSpacing: '-0.02em' }}
          >
            Intelligent legal research<br className="hidden sm:block" />powered by AI
          </h1>

          {/* Subtitle */}
          <p className="text-base md:text-lg text-gray-500 max-w-lg mx-auto mb-6 leading-relaxed">
            Search thousands of legal cases, get AI-powered insights,
            and analyze confidential documents — all in one place.
          </p>



          {/* CTA Button */}
          <button
            onClick={() => navigate(isAuthenticated ? '/assistant' : '/login')}
            className="inline-flex items-center gap-2 bg-gray-900 text-white px-7 py-3 rounded-full text-sm font-medium hover:bg-gray-700 transition-colors shadow-sm"
          >
            {isAuthenticated ? 'Open Assistant' : 'Start for free'}
          </button>
        </div>

        {/* Product Preview Mockup */}
        <div className="max-w-5xl mx-auto px-4 sm:px-6 pb-16 sm:pb-24">
          <div
            className="rounded-2xl overflow-hidden shadow-2xl border border-gray-200/50"
            style={{ backgroundColor: '#ffffff' }}
          >
            {/* Mockup Top Bar */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100 bg-white">
              <div className="flex items-center gap-2">
                <div className="bg-gray-900 p-1 rounded-md">
                  <Scale className="h-3.5 w-3.5 text-white" />
                </div>
                <span className="text-sm font-semibold text-gray-800">LexFind</span>
                <span className="text-gray-300 mx-1">/</span>
                <span className="text-sm text-gray-500">Legal Search</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-gray-200" />
                <div className="w-6 h-6 rounded-full bg-gradient-to-br from-amber-400 to-orange-500" />
              </div>
            </div>

            {/* Mockup Sidebar + Content */}
            <div className="flex" style={{ minHeight: '280px' }}>
              {/* Sidebar - hidden on mobile */}
              <div className="hidden sm:flex w-40 border-r border-gray-100 p-4 bg-gray-50/50 flex-col gap-1">
                {[
                  { icon: Search, label: 'Legal Search', active: true },
                  { icon: MessageSquare, label: 'AI Assistant', active: false },
                  { icon: Upload, label: 'Analysis', active: false },
                ].map(({ icon: Icon, label, active }) => (
                  <div
                    key={label}
                    className={`flex items-center gap-2 px-2.5 py-2 rounded-lg text-xs font-medium cursor-default ${
                      active ? 'bg-white shadow-sm text-gray-900' : 'text-gray-400'
                    }`}
                  >
                    <Icon className={`h-3.5 w-3.5 ${active ? 'text-gray-700' : 'text-gray-300'}`} />
                    {label}
                  </div>
                ))}
              </div>

              {/* Main content */}
              <div className="flex-1 p-4 sm:p-6 min-w-0">
                <div className="flex items-center justify-between mb-4">
                  <div className="min-w-0">
                    <h2 className="text-sm sm:text-base font-semibold text-gray-900">Case Search</h2>
                    <p className="text-xs text-gray-400 mt-0.5 hidden sm:block">Search through 46,000+ legal precedents</p>
                  </div>
                  <button className="text-xs bg-gray-900 text-white px-3 py-1.5 rounded-full font-medium flex-shrink-0 ml-2">
                    New Search
                  </button>
                </div>

                {/* Search Bar Mockup */}
                <div className="flex items-center gap-2 border border-gray-200 rounded-xl px-3 py-2 mb-4 bg-gray-50">
                  <Search className="h-3.5 w-3.5 text-gray-400 flex-shrink-0" />
                  <span className="text-xs text-gray-400 truncate">Search legal cases, acts, precedents...</span>
                </div>

                {/* Result Cards */}
                <div className="space-y-3">
                  {[
                    { title: 'Kesavananda Bharati v. State of Kerala', year: '1973', type: 'Constitutional' },
                    { title: 'Maneka Gandhi v. Union of India', year: '1978', type: 'Fundamental Rights' },
                    { title: 'Vishaka v. State of Rajasthan', year: '1997', type: 'Labour Law' },
                  ].map((c) => (
                    <div key={c.title} className="flex items-center justify-between p-3 rounded-xl border border-gray-100 bg-white cursor-default">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-amber-50 border border-amber-100 flex items-center justify-center flex-shrink-0">
                          <FileText className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-amber-600" />
                        </div>
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-gray-800 line-clamp-1">{c.title}</p>
                          <p className="text-xs text-gray-400 mt-0.5">{c.type} · {c.year}</p>
                        </div>
                      </div>
                      <span className="text-xs text-gray-300 font-medium ml-2 flex-shrink-0 hidden sm:inline">View →</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Strip */}
      <div className="border-t border-gray-200/60" style={{ backgroundColor: '#EAEAE4' }}>
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-14 sm:py-20">
          <div className="text-center mb-12">
            <h2
              className="font-serif-display text-3xl md:text-4xl text-gray-900 mb-4"
              style={{ letterSpacing: '-0.01em' }}
            >
              Everything a legal professional needs
            </h2>
            <p className="text-sm text-gray-500 max-w-md mx-auto">
              From quick case lookups to deep document analysis — built for speed, accuracy, and privacy.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {[
              {
                icon: Search,
                title: 'Case Search',
                desc: 'Instantly find relevant precedents from our comprehensive legal database of 46,000+ cases.',
                path: '/search',
              },
              {
                icon: MessageSquare,
                title: 'AI Legal Assistant',
                desc: 'Ask complex legal questions or upload confidential files for deep AI-powered analysis with citations.',
                path: '/assistant',
              },
            ].map((f) => (
              <div
                key={f.title}
                onClick={() =>
                  isAuthenticated
                    ? navigate(f.path)
                    : navigate('/login', { state: { from: { pathname: f.path } } })
                }
                className="group p-6 rounded-2xl border border-gray-200/60 bg-white/60 hover:bg-white hover:shadow-md transition-all duration-200 cursor-pointer"
              >
                <div className="w-9 h-9 rounded-xl bg-gray-900 flex items-center justify-center mb-4">
                  <f.icon className="h-4 w-4 text-white" />
                </div>
                <h3 className="text-sm font-semibold text-gray-900 mb-1.5">{f.title}</h3>
                <p className="text-sm text-gray-500 leading-relaxed">{f.desc}</p>
                <div className="mt-4 text-xs font-medium text-gray-900 group-hover:gap-2 flex items-center gap-1 transition-all">
                  Get started <span className="group-hover:translate-x-0.5 transition-transform inline-block">→</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom CTA */}
      <div className="border-t border-gray-200/60" style={{ backgroundColor: '#E0E0DA' }}>
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-14 sm:py-20 text-center">
          <h2
            className="font-serif-display text-3xl md:text-4xl text-gray-900 mb-4"
            style={{ letterSpacing: '-0.01em' }}
          >
            Ready to research smarter?
          </h2>
          <p className="text-sm text-gray-500 mb-8 max-w-sm mx-auto">
            Join legal professionals who trust LexFind for fast, reliable, AI-powered research.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            {isAuthenticated ? (
              <button
                onClick={() => navigate('/assistant')}
                className="bg-gray-900 text-white px-7 py-3 rounded-full text-sm font-medium hover:bg-gray-700 transition-colors"
              >
                Go to Assistant →
              </button>
            ) : (
              <>
                <button
                  onClick={() => navigate('/login')}
                  className="bg-gray-900 text-white px-7 py-3 rounded-full text-sm font-medium hover:bg-gray-700 transition-colors"
                >
                  Get started — it's free
                </button>
                <button
                  onClick={() => navigate('/login')}
                  className="bg-white text-gray-700 px-7 py-3 rounded-full text-sm font-medium border border-gray-200 hover:border-gray-300 hover:bg-gray-50 transition-colors"
                >
                  Sign in
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;
