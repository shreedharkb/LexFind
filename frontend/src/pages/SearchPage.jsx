import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search, FileText, Clock, Brain, Eye, Download, Loader2,
  DownloadCloud, AlertCircle, X, SlidersHorizontal, ChevronDown,
  Users, Layers, ChevronUp, ArrowRight
} from 'lucide-react';
import { getPdfUrl, casesApi } from '../config/apiClient';
import { useAuth } from '../context/AuthContext';

// ── PDF Viewer Modal ────────────────────────────────────────────────────────
function PdfViewerModal({ url, title, onClose }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const [fetchError, setFetchError] = useState(false);

  useEffect(() => {
    if (!url) return;
    let objectUrl = null;
    fetch(url)
      .then((res) => { if (!res.ok) throw new Error('fetch failed'); return res.blob(); })
      .then((blob) => { objectUrl = URL.createObjectURL(blob); setBlobUrl(objectUrl); })
      .catch(() => setFetchError(true));
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [url]);

  if (!url) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-white w-full max-w-5xl h-[90vh] rounded-2xl overflow-hidden flex flex-col shadow-2xl">
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between bg-white shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-8 h-8 bg-amber-50 rounded-lg flex items-center justify-center shrink-0">
              <FileText className="h-4 w-4 text-amber-600" />
            </div>
            <h3 className="text-sm font-semibold text-gray-900 truncate">{title}</h3>
          </div>
          <div className="flex items-center gap-2">
            <a href={url} download={`${title}.pdf`} className="p-2 text-gray-400 hover:text-gray-700 hover:bg-gray-50 rounded-lg transition-all" title="Download PDF">
              <Download className="h-4 w-4" />
            </a>
            <button onClick={onClose} className="p-2 text-gray-400 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-all">
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>
        <div className="flex-1 bg-gray-100 overflow-hidden relative flex items-center justify-center">
          {fetchError ? (
            <div className="text-center text-gray-500 text-sm space-y-3">
              <AlertCircle className="h-8 w-8 mx-auto text-gray-300" />
              <p>Could not load PDF preview.</p>
              <a href={url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 px-4 py-2 bg-gray-900 text-white text-xs font-medium rounded-xl hover:bg-gray-700 transition-all">
                <Eye className="h-3.5 w-3.5" /> Open in new tab
              </a>
            </div>
          ) : blobUrl ? (
            <iframe src={`${blobUrl}#toolbar=1&view=FitH`} className="w-full h-full border-none" title={title} />
          ) : (
            <div className="flex flex-col items-center gap-3 text-gray-400">
              <Loader2 className="h-6 w-6 animate-spin" />
              <span className="text-xs">loading pdf...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Similar Cases Modal ──────────────────────────────────────────────────────
function SimilarCasesModal({ documentId, caseTitle, token, onClose, onAnalyze, onViewPdf }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!documentId) return;
    casesApi.getSimilarCases(documentId, token, 6)
      .then(data => setResults(data?.results || []))
      .catch(() => setError('Could not load similar cases.'))
      .finally(() => setLoading(false));
  }, [documentId, token]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-white w-full max-w-2xl max-h-[80vh] rounded-2xl overflow-hidden flex flex-col shadow-2xl">
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between shrink-0">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-gray-900">Similar Cases</h3>
            <p className="text-xs text-gray-400 truncate mt-0.5">{caseTitle}</p>
          </div>
          <button onClick={onClose} className="p-2 text-gray-400 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-all shrink-0">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {loading && (
            <div className="flex items-center justify-center py-12 gap-3 text-gray-400">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span className="text-sm">Finding similar cases...</span>
            </div>
          )}
          {error && <p className="text-center text-sm text-red-500 py-8">{error}</p>}
          {!loading && !error && results.length === 0 && (
            <p className="text-center text-sm text-gray-400 py-8">No similar cases found.</p>
          )}
          {results.map((r, i) => {
            const id = r.document_id || r.case_id;
            return (
              <div key={i} className="bg-gray-50 border border-gray-200/60 rounded-xl p-4 hover:border-gray-300 transition-all">
                <div className="flex items-start justify-between gap-2 mb-1">
                  <h4 className="text-sm font-medium text-gray-900 flex-1 leading-snug">{r.title}</h4>
                  <span className="shrink-0 bg-white text-gray-500 border border-gray-200 px-2 py-0.5 rounded-full text-xs font-medium">
                    {Math.round(r.similarity_percentage || r.score * 100 || 80)}%
                  </span>
                </div>
                {(r.court || r.year) && (
                  <p className="text-xs text-gray-400 mb-2">{[r.court, r.year].filter(Boolean).join(' · ')}</p>
                )}
                <p className="text-xs text-gray-500 line-clamp-2 mb-3">{r.top_chunk?.chunk_text || ''}</p>
                <div className="flex items-center gap-2">
                  <button onClick={() => onAnalyze(id)} className="flex items-center gap-1.5 bg-gray-900 hover:bg-gray-700 text-white px-3 py-1.5 rounded-full text-xs font-medium transition-colors">
                    <Brain className="h-3 w-3" /> Analyze
                  </button>
                  <button onClick={() => onViewPdf(id, r.title)} className="flex items-center gap-1.5 bg-white border border-gray-200 hover:border-gray-300 text-gray-700 px-3 py-1.5 rounded-full text-xs font-medium transition-colors">
                    <Eye className="h-3 w-3" /> View PDF
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── Skeleton Loader ──────────────────────────────────────────────────────────
function SkeletonCard() {
  return (
    <div className="bg-white border border-gray-200/60 rounded-2xl p-5 sm:p-6 animate-pulse">
      <div className="flex justify-between mb-3">
        <div className="h-4 bg-gray-200 rounded w-2/3"></div>
        <div className="h-5 bg-gray-100 rounded-full w-16"></div>
      </div>
      <div className="h-3 bg-gray-100 rounded w-1/3 mb-3"></div>
      <div className="space-y-2 mb-4">
        <div className="h-3 bg-gray-100 rounded w-full"></div>
        <div className="h-3 bg-gray-100 rounded w-5/6"></div>
        <div className="h-3 bg-gray-100 rounded w-4/6"></div>
      </div>
      <div className="flex gap-2">
        <div className="h-8 bg-gray-200 rounded-full w-36"></div>
        <div className="h-8 bg-gray-100 rounded-full w-24"></div>
        <div className="h-8 bg-gray-100 rounded-full w-24"></div>
      </div>
    </div>
  );
}

// ── Main SearchPage ──────────────────────────────────────────────────────────
const COURTS = [
  'Supreme Court of India',
  'High Court of Bombay',
  'High Court of Delhi',
  'High Court of Madras',
  'High Court of Calcutta',
  'High Court of Allahabad',
  'High Court of Karnataka',
  'High Court of Kerala',
  'High Court of Gujarat',
  'High Court of Rajasthan',
];

const CASE_TYPES = [
  'Criminal Appeal',
  'Civil Appeal',
  'Writ Petition',
  'Special Leave Petition',
  'Review Petition',
  'Transfer Petition',
  'Original Suit',
  'Reference',
];

const STATES = [
  'Maharashtra', 'Delhi', 'Tamil Nadu', 'West Bengal', 'Uttar Pradesh',
  'Karnataka', 'Kerala', 'Gujarat', 'Rajasthan', 'Andhra Pradesh',
  'Telangana', 'Madhya Pradesh', 'Bihar', 'Odisha', 'Punjab',
];

function SearchPage() {
  const navigate = useNavigate();
  const { token } = useAuth();

  const [query, setQuery] = useState(() => sessionStorage.getItem('lexSearchQuery') || '');
  const [results, setResults] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('lexSearchResults') || '[]'); } catch { return []; }
  });
  const [loading, setLoading] = useState(false);
  const [searchTime, setSearchTime] = useState(null);
  const [isKeywordMode, setIsKeywordMode] = useState(false);
  const [isNameMode, setIsNameMode] = useState(false);
  const [analyzingId, setAnalyzingId] = useState(null);
  const [viewerPdf, setViewerPdf] = useState(null);
  const [similarModal, setSimilarModal] = useState(null); // { documentId, caseTitle }

  // Filters
  const [showFilters, setShowFilters] = useState(false);
  const [filterCourt, setFilterCourt] = useState('');
  const [filterYearMin, setFilterYearMin] = useState('');
  const [filterYearMax, setFilterYearMax] = useState('');
  const [filterState, setFilterState] = useState('');
  const [filterCaseType, setFilterCaseType] = useState('');

  const activeFilterCount = [filterCourt, filterYearMin, filterYearMax, filterState, filterCaseType].filter(Boolean).length;

  useEffect(() => { sessionStorage.setItem('lexSearchQuery', query); }, [query]);
  useEffect(() => { sessionStorage.setItem('lexSearchResults', JSON.stringify(results)); }, [results]);

  const filenameToTitle = (filename) => {
    if (!filename) return 'Legal Case Document';
    return filename.replace(/\.pdf$/i, '').replace(/__+/g, ' — ').replace(/[_]+/g, ' ').replace(/\s+/g, ' ').trim();
  };

  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setResults([]);
    try {
      let data;
      if (isNameMode) {
        data = await casesApi.searchByName(query, token, 10);
      } else {
        const filters = {};
        if (filterCourt) filters.court = filterCourt;
        if (filterYearMin) filters.year_min = parseInt(filterYearMin);
        if (filterYearMax) filters.year_max = parseInt(filterYearMax);
        if (filterState) filters.state = filterState;
        if (filterCaseType) filters.case_type = filterCaseType;
        const search_mode = isKeywordMode ? 'keyword' : 'hybrid';
        data = await casesApi.search(query, token, 10, search_mode, filters);
      }
      setResults(data.results || []);
      setSearchTime(data.search_time_ms ? (data.search_time_ms / 1000).toFixed(2) : null);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async (caseId) => {
    if (!caseId) return;
    setAnalyzingId(caseId);
    try {
      const data = await casesApi.analyze(caseId, token);
      if (data.session_id) navigate(`/assistant/${data.session_id}`);
    } catch (err) {
      console.error('Analyze error:', err);
      alert('Failed to start analysis session.');
    } finally {
      setAnalyzingId(null);
    }
  };

  const handleReset = () => {
    setQuery('');
    setResults([]);
    setSearchTime(null);
    clearFilters();
    sessionStorage.removeItem('lexSearchQuery');
    sessionStorage.removeItem('lexSearchResults');
  };

  const clearFilters = () => {
    setFilterCourt(''); setFilterYearMin(''); setFilterYearMax('');
    setFilterState(''); setFilterCaseType('');
  };

  const suggestedQueries = [
    "Contract breach in employment law",
    "Property rights in joint ownership",
    "Criminal liability in corporate fraud",
    "Constitutional rights violation cases",
    "Intellectual property infringement",
  ];

  const selectStyle = "w-full text-sm border border-gray-200 rounded-xl px-3 py-2 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-900/20 focus:border-gray-400 transition-all";
  const inputStyle = "w-full text-sm border border-gray-200 rounded-xl px-3 py-2 bg-white text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-900/20 focus:border-gray-400 transition-all";

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#EAEAE4' }}>
      {/* Search Header */}
      <div className="bg-white/70 backdrop-blur-sm border-b border-gray-200/60">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-10">
          <div className="text-center mb-8">
            <h1 className="font-serif-display text-3xl sm:text-4xl text-gray-900 mb-3" style={{ letterSpacing: '-0.01em' }}>
              Legal Case Search
            </h1>
            <p className="text-sm text-gray-500 max-w-xl mx-auto">
              Search through 46,000+ legal cases and precedents instantly.
            </p>
          </div>

          {/* Search Form */}
          <form onSubmit={handleSearch} className="max-w-3xl mx-auto">
            <div className="relative bg-white rounded-2xl shadow-sm border border-gray-200/60 p-1.5 sm:p-2">
              <div className="flex items-center gap-1 sm:gap-2">
                <Search className="h-4 w-4 text-gray-400 ml-2 sm:ml-3 flex-shrink-0" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={isNameMode ? "Enter party name or case title..." : "Enter your legal query or case details..."}
                  className="flex-1 min-w-0 px-1 sm:px-2 py-2.5 text-sm text-gray-800 placeholder-gray-400 border-none outline-none bg-transparent"
                />
                {(query || results.length > 0) && (
                  <button
                    type="button"
                    onClick={handleReset}
                    title="Reset Search"
                    className="p-1.5 text-gray-400 hover:text-gray-900 rounded-full hover:bg-gray-100 transition-colors flex-shrink-0"
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="bg-gray-900 hover:bg-gray-700 disabled:bg-gray-300 text-white px-3 sm:px-5 py-2.5 rounded-xl text-sm font-medium transition-colors flex-shrink-0 flex items-center gap-2"
                >
                  {loading ? (
                    <div className="flex items-center gap-1.5">
                      <Loader2 className="animate-spin h-4 w-4" />
                      <span className="hidden sm:inline">Searching</span>
                    </div>
                  ) : (
                    <>
                      <Search className="h-4 w-4 sm:hidden" />
                      <span className="hidden sm:inline">Search Cases</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>

          {/* Toggles row */}
          <div className="max-w-3xl mx-auto mt-3 px-1 flex flex-wrap items-center gap-x-5 gap-y-2">
            {/* Keyword mode toggle */}
            <label className="inline-flex items-center gap-2 cursor-pointer select-none group">
              <input
                id="keyword-mode-toggle"
                type="checkbox"
                checked={isKeywordMode}
                onChange={(e) => { setIsKeywordMode(e.target.checked); if (e.target.checked) setIsNameMode(false); }}
                className="w-4 h-4 rounded border-gray-300 text-gray-900 accent-gray-900 cursor-pointer"
              />
              <span className="text-sm text-gray-600 group-hover:text-gray-900 transition-colors">
                Keyword Search
              </span>
            </label>

            {/* Party name mode toggle */}
            <label className="inline-flex items-center gap-2 cursor-pointer select-none group">
              <input
                id="name-mode-toggle"
                type="checkbox"
                checked={isNameMode}
                onChange={(e) => { setIsNameMode(e.target.checked); if (e.target.checked) setIsKeywordMode(false); }}
                className="w-4 h-4 rounded border-gray-300 text-gray-900 accent-gray-900 cursor-pointer"
              />
              <span className="text-sm text-gray-600 group-hover:text-gray-900 transition-colors flex items-center gap-1.5">
                <Users className="h-3.5 w-3.5" /> Search by Party Name
              </span>
            </label>

            {/* Filters toggle */}
            {!isNameMode && (
              <button
                type="button"
                onClick={() => setShowFilters(f => !f)}
                className={`inline-flex items-center gap-1.5 text-sm transition-colors select-none ${showFilters ? 'text-gray-900' : 'text-gray-500 hover:text-gray-900'}`}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
                Filters
                {activeFilterCount > 0 && (
                  <span className="bg-gray-900 text-white text-xs px-1.5 py-0.5 rounded-full leading-none">{activeFilterCount}</span>
                )}
                {showFilters ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              </button>
            )}
          </div>

          {/* Mode badges */}
          <div className="max-w-3xl mx-auto px-1 mt-1.5 flex flex-wrap gap-2">
            {isKeywordMode && <p className="text-xs text-amber-600">Exact keyword matches from our 46,456 case corpus</p>}
            {isNameMode && <p className="text-xs text-blue-600">Searching by party name / case title</p>}
          </div>

          {/* Collapsible Filters Panel */}
          {showFilters && !isNameMode && (
            <div className="max-w-3xl mx-auto mt-4 bg-white/80 border border-gray-200/60 rounded-2xl p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Filter Results</span>
                {activeFilterCount > 0 && (
                  <button onClick={clearFilters} className="text-xs text-gray-400 hover:text-gray-700 transition-colors">
                    Clear all
                  </button>
                )}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-gray-500 mb-1.5 font-medium">Court</label>
                  <select value={filterCourt} onChange={e => setFilterCourt(e.target.value)} className={selectStyle}>
                    <option value="">All Courts</option>
                    {COURTS.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1.5 font-medium">State / UT</label>
                  <select value={filterState} onChange={e => setFilterState(e.target.value)} className={selectStyle}>
                    <option value="">All States</option>
                    {STATES.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1.5 font-medium">Case Type</label>
                  <select value={filterCaseType} onChange={e => setFilterCaseType(e.target.value)} className={selectStyle}>
                    <option value="">All Types</option>
                    {CASE_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1.5 font-medium">Year From</label>
                  <input type="number" min="1950" max="2025" placeholder="e.g. 2010" value={filterYearMin} onChange={e => setFilterYearMin(e.target.value)} className={inputStyle} />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1.5 font-medium">Year To</label>
                  <input type="number" min="1950" max="2025" placeholder="e.g. 2023" value={filterYearMax} onChange={e => setFilterYearMax(e.target.value)} className={inputStyle} />
                </div>
              </div>
              <div className="mt-3 flex justify-end">
                <button
                  onClick={handleSearch}
                  className="flex items-center gap-1.5 bg-gray-900 hover:bg-gray-700 text-white px-4 py-2 rounded-xl text-sm font-medium transition-colors"
                >
                  <Search className="h-3.5 w-3.5" /> Apply Filters
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Suggested Queries — only when empty and not loading */}
      {!results.length && !loading && (
        <div className="py-10">
          <div className="max-w-3xl mx-auto px-4 sm:px-6">
            <p className="text-xs text-gray-400 text-center font-medium uppercase tracking-wider mb-4">Popular Topics</p>
            <div className="flex flex-wrap justify-center gap-2">
              {suggestedQueries.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => setQuery(suggestion)}
                  className="bg-white hover:bg-gray-50 text-gray-600 hover:text-gray-900 px-4 py-2 rounded-full border border-gray-200 hover:border-gray-300 transition-all duration-200 text-sm"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Skeleton loaders */}
      {loading && (
        <div className="py-8">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 space-y-3">
            {[1, 2, 3].map(i => <SkeletonCard key={i} />)}
          </div>
        </div>
      )}

      {/* Search Results */}
      {results.length > 0 && !loading && (
        <div className="py-8">
          <div className="max-w-5xl mx-auto px-4 sm:px-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-base font-semibold text-gray-900">
                {results.length} result{results.length !== 1 ? 's' : ''} found
                {isNameMode && <span className="ml-2 text-xs font-normal text-blue-500">by party name</span>}
              </h3>
              <div className="flex items-center gap-1.5 text-xs text-gray-400">
                <Clock className="h-3.5 w-3.5" />
                <span>{searchTime ? `${searchTime}s` : '—'}</span>
              </div>
            </div>

            <div className="space-y-3">
              {results.map((result, index) => {
                const id = result.document_id || result.case_id || result.filename;
                const isAnalyzing = analyzingId === id;
                return (
                  <div
                    key={index}
                    className="bg-white border border-gray-200/60 rounded-2xl p-5 sm:p-6 hover:border-gray-300 hover:shadow-sm transition-all duration-200"
                    style={{ animation: `fadeInUp 0.3s ease ${index * 0.05}s both` }}
                  >
                    <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mb-2">
                      <h4 className="text-sm font-semibold text-gray-900 flex-1">
                        {result.title || filenameToTitle(id)}
                      </h4>
                      <span className="self-start bg-gray-100 text-gray-600 px-2.5 py-1 rounded-full text-xs font-medium whitespace-nowrap">
                        {Math.round(result.similarity_percentage || result.score * 100 || 80)}% match
                      </span>
                    </div>
                    {(result.court || result.year) && (
                      <div className="text-xs text-gray-400 mb-3 flex flex-wrap gap-2">
                        {result.court && <span className="font-medium text-gray-600">{result.court}</span>}
                        {result.court && result.year && <span>•</span>}
                        {result.year && <span>{result.year}</span>}
                        {result.case_type && <><span>•</span><span>{result.case_type}</span></>}
                        {result.state && <><span>•</span><span>{result.state}</span></>}
                      </div>
                    )}
                    <p className="text-sm text-gray-500 mb-4 leading-relaxed line-clamp-3">
                      {result.top_chunk?.chunk_text || result.content || result.text || `Case ID: ${id}`}
                    </p>
                    <div className="flex flex-wrap items-center gap-2">
                      {/* Analyze */}
                      <button
                        className="flex items-center gap-1.5 bg-gray-900 hover:bg-gray-700 text-white px-4 py-2 rounded-full text-xs font-medium transition-colors disabled:opacity-50"
                        onClick={() => handleAnalyze(id)}
                        disabled={isAnalyzing}
                      >
                        {isAnalyzing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Brain className="h-3.5 w-3.5" />}
                        <span>Analyze in Assistant</span>
                      </button>
                      {/* View PDF */}
                      <button
                        className="flex items-center gap-1.5 bg-white border border-gray-200 hover:border-gray-300 text-gray-700 px-4 py-2 rounded-full text-xs font-medium transition-colors"
                        onClick={() => { if (id) setViewerPdf({ url: getPdfUrl(id), title: result.title || filenameToTitle(id) }); }}
                      >
                        <Eye className="h-3.5 w-3.5" />
                        <span>View PDF</span>
                      </button>
                      {/* Download */}
                      <a
                        href={id ? getPdfUrl(id) : '#'}
                        download={`${result.title || filenameToTitle(id)}.pdf`}
                        className="flex items-center gap-1.5 bg-white border border-gray-200 hover:border-gray-300 text-gray-700 px-4 py-2 rounded-full text-xs font-medium transition-colors"
                        onClick={(e) => { if (!id) e.preventDefault(); }}
                      >
                        <DownloadCloud className="h-3.5 w-3.5" />
                        <span>Download</span>
                      </a>
                      {/* Similar Cases — only for corpus results that have a document_id */}
                      {result.document_id && (
                        <button
                          className="flex items-center gap-1.5 bg-white border border-gray-200 hover:border-gray-300 text-gray-700 px-4 py-2 rounded-full text-xs font-medium transition-colors"
                          onClick={() => setSimilarModal({ documentId: result.document_id, caseTitle: result.title || filenameToTitle(id) })}
                        >
                          <Layers className="h-3.5 w-3.5" />
                          <span>Similar Cases</span>
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* PDF Viewer Modal */}
      {viewerPdf && (
        <PdfViewerModal
          url={viewerPdf.url}
          title={viewerPdf.title}
          onClose={() => setViewerPdf(null)}
        />
      )}

      {/* Similar Cases Modal */}
      {similarModal && (
        <SimilarCasesModal
          documentId={similarModal.documentId}
          caseTitle={similarModal.caseTitle}
          token={token}
          onClose={() => setSimilarModal(null)}
          onAnalyze={(id) => { setSimilarModal(null); handleAnalyze(id); }}
          onViewPdf={(id, title) => { setSimilarModal(null); setViewerPdf({ url: getPdfUrl(id), title }); }}
        />
      )}

      {/* Fade-in animation keyframes */}
      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

export default SearchPage;
