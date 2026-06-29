import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, FileText, Clock, Brain, Eye, Download, Loader2, DownloadCloud, AlertCircle, X } from 'lucide-react';
import { getPdfUrl } from '../config/apiClient';
import { casesApi } from '../config/apiClient';
import { useAuth } from '../context/AuthContext';

// ── PDF Viewer Modal ───────────────────────────────────────────────────────
// Fetches the PDF as a blob to bypass cross-origin iframe restrictions.

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
            <a 
              href={url} 
              download={title.toLowerCase().endsWith('.pdf') ? title : `${title}.pdf`} 
              className="p-2 text-gray-400 hover:text-gray-700 hover:bg-gray-50 rounded-lg transition-all" 
              title="Download PDF"
            >
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

function SearchPage() {
  const navigate = useNavigate();
  const { token } = useAuth();
  const [query, setQuery] = useState(() => {
    return sessionStorage.getItem('lexSearchQuery') || '';
  });
  const [results, setResults] = useState(() => {
    try {
      const saved = sessionStorage.getItem('lexSearchResults');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    sessionStorage.setItem('lexSearchQuery', query);
  }, [query]);

  useEffect(() => {
    sessionStorage.setItem('lexSearchResults', JSON.stringify(results));
  }, [results]);
  const [loading, setLoading] = useState(false);
  const [searchTime, setSearchTime] = useState(null);
  const [isKeywordMode, setIsKeywordMode] = useState(false);
  const [analyzingId, setAnalyzingId] = useState(null);
  const [viewerPdf, setViewerPdf] = useState(null); // { url, title }

  // Convert raw filenames like "abc__court__2019_Judgement.pdf" into readable titles
  const filenameToTitle = (filename) => {
    if (!filename) return 'Legal Case Document';
    return filename
      .replace(/\.pdf$/i, '')
      .replace(/__+/g, ' \u2014 ')
      .replace(/[_]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    const search_mode = isKeywordMode ? 'keyword' : 'hybrid';
    try {
      const data = await casesApi.search(query, token, 10, search_mode);
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
    setAnalyzingId(caseId);
    try {
      // V2: Create session, attach doc, get session_id
      const data = await casesApi.analyze(caseId, token);
      if (data.session_id) {
        navigate(`/assistant/${data.session_id}`);
      }
    } catch (err) {
      console.error('Analyze error:', err);
      alert('Failed to start analysis session.');
    } finally {
      setAnalyzingId(null);
    }
  };

  const suggestedQueries = [
    "Contract breach in employment law",
    "Property rights in joint ownership",
    "Criminal liability in corporate fraud",
    "Constitutional rights violation cases",
    "Intellectual property infringement"
  ];

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
                  placeholder="Enter your legal query or case details..."
                  className="flex-1 min-w-0 px-1 sm:px-2 py-2.5 text-sm text-gray-800 placeholder-gray-400 border-none outline-none bg-transparent"
                />
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="bg-gray-900 hover:bg-gray-700 disabled:bg-gray-300 text-white px-3 sm:px-5 py-2.5 rounded-xl
                           text-sm font-medium transition-colors flex-shrink-0 flex items-center gap-2"
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

          {/* Keyword mode toggle */}
          <div className="max-w-3xl mx-auto mt-3 px-1">
            <label className="inline-flex items-center gap-2.5 cursor-pointer select-none group">
              <input
                id="keyword-mode-toggle"
                type="checkbox"
                checked={isKeywordMode}
                onChange={(e) => setIsKeywordMode(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-gray-900 accent-gray-900 cursor-pointer"
              />
              <span className="text-sm text-gray-600 group-hover:text-gray-900 transition-colors">
                Keyword Search
                <span className="text-gray-400 font-normal"> — Exact match for case names, citations, act references</span>
              </span>
            </label>
            {isKeywordMode && (
              <p className="mt-1.5 ml-6 text-xs text-amber-600">
                Showing exact keyword matches from our 46,456 case corpus
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Suggested Queries */}
      {!results.length && !loading && (
        <div className="py-10">
          <div className="max-w-3xl mx-auto px-4 sm:px-6">
            <p className="text-xs text-gray-400 text-center font-medium uppercase tracking-wider mb-4">Popular Topics</p>
            <div className="flex flex-wrap justify-center gap-2">
              {suggestedQueries.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => setQuery(suggestion)}
                  className="bg-white hover:bg-gray-50 text-gray-600 hover:text-gray-900
                           px-4 py-2 rounded-full border border-gray-200 hover:border-gray-300
                           transition-all duration-200 text-sm"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search Results */}
      {results.length > 0 && (
        <div className="py-8">
          <div className="max-w-5xl mx-auto px-4 sm:px-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-base font-semibold text-gray-900">
                {results.length} results found
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
                  <div key={index} className="bg-white border border-gray-200/60 rounded-2xl p-5 sm:p-6
                                            hover:border-gray-300 hover:shadow-sm transition-all duration-200">
                    <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mb-2">
                      <h4 className="text-sm font-semibold text-gray-900 flex-1">
                        {result.title || filenameToTitle(id)}
                      </h4>
                      <span className="self-start bg-gray-100 text-gray-600 px-2.5 py-1 rounded-full text-xs font-medium whitespace-nowrap">
                        {Math.round((result.similarity_percentage || result.score * 100 || 80))}% match
                      </span>
                    </div>
                    {(result.court || result.year) && (
                      <div className="text-xs text-gray-400 mb-3 flex flex-wrap gap-2">
                        {result.court && <span className="font-medium text-gray-600">{result.court}</span>}
                        {result.court && result.year && <span>•</span>}
                        {result.year && <span>{result.year}</span>}
                        {result.case_type && <span>•</span>}
                        {result.case_type && <span>{result.case_type}</span>}
                      </div>
                    )}
                    <p className="text-sm text-gray-500 mb-4 leading-relaxed line-clamp-3">
                      {result.top_chunk?.chunk_text || result.content || result.text || `Case ID: ${id}`}
                    </p>
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        className="flex items-center gap-1.5 bg-gray-900 hover:bg-gray-700 text-white
                          px-4 py-2 rounded-full text-xs font-medium transition-colors disabled:opacity-50"
                        onClick={() => handleAnalyze(id)}
                        disabled={isAnalyzing}
                      >
                        {isAnalyzing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Brain className="h-3.5 w-3.5" />}
                        <span>Analyze in Assistant</span>
                      </button>
                      <button
                        className="flex items-center gap-1.5 bg-white border border-gray-200 hover:border-gray-300
                          text-gray-700 px-4 py-2 rounded-full text-xs font-medium transition-colors"
                        onClick={() => {
                          if (id) setViewerPdf({ url: getPdfUrl(id), title: result.title || filenameToTitle(id) });
                        }}
                      >
                        <Eye className="h-3.5 w-3.5" />
                        <span>View PDF</span>
                      </button>
                      <a
                        href={id ? getPdfUrl(id) : '#'}
                        download={(result.title || filenameToTitle(id)).toLowerCase().endsWith('.pdf') ? (result.title || filenameToTitle(id)) : `${result.title || filenameToTitle(id)}.pdf`}
                        className="flex items-center gap-1.5 bg-white border border-gray-200 hover:border-gray-300
                          text-gray-700 px-4 py-2 rounded-full text-xs font-medium transition-colors"
                        onClick={(e) => { if (!id) e.preventDefault(); }}
                      >
                        <DownloadCloud className="h-3.5 w-3.5" />
                        <span>Download</span>
                      </a>
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
    </div>
  );
}

export default SearchPage;
