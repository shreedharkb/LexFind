/**
 * Centralised API client for LexFind.
 * All requests target /api/* routes.
 * Automatically injects Authorization: Bearer <token> on authenticated calls.
 */

export const BASE = import.meta.env.VITE_API_BASE_URL ?? '';

/** Get the full URL for serving a corpus case PDF (search page — no auth needed). */
export const getPdfUrl = (filename) =>
  `${BASE}/api/search/pdf/${encodeURIComponent(filename)}`;

/** Get the full URL for serving any document PDF (assistant citations — requires auth token). */
export const getDocumentPdfUrl = (documentId) =>
  `${BASE}/api/documents/${encodeURIComponent(documentId)}/pdf`;


async function request(path, options = {}, token = null) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers ?? {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });

  // 204 No Content — no body to parse
  if (res.status === 204) return null;

  // Parse body regardless of status so we can return error detail
  let body;
  const ct = res.headers.get('content-type') ?? '';
  if (ct.includes('application/json')) {
    body = await res.json();
  } else if (ct.includes('text/event-stream')) {
    // For SSE, we don't parse as JSON here; the caller handles the stream
    return res; 
  } else {
    body = await res.text();
  }

  if (!res.ok) {
    const msg = body?.detail ?? body ?? `HTTP ${res.status}`;
    const err = new Error(typeof msg === 'object' ? JSON.stringify(msg) : msg);
    err.status = res.status;
    throw err;
  }
  return body;
}

// ── Auth ─── /api/auth/* ───────────────────────────────────────────────────
export const authApi = {
  register: (email, password) =>
    request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  login: (email, password) =>
    request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),
};

// ── Cases ─── /api/cases/* ────────────────────────────────────────────────
export const casesApi = {
  search: (query, token, top_k = 10, search_mode = 'hybrid') =>
    request('/api/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k, search_mode }),
    }, token),

  get: (caseId) =>
    request(`/api/cases/${encodeURIComponent(caseId)}`),
    
  analyze: (caseId, token) =>
    request(`/api/cases/${encodeURIComponent(caseId)}/analyze`, { method: 'POST' }, token),
};

// ── Sessions ─── /api/sessions/* ──────────────────────────────────────────
export const sessionsApi = {
  list: (token) =>
    request('/api/sessions', {}, token),

  create: (title, token) =>
    request('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    }, token),

  get: (id, token) =>
    request(`/api/sessions/${id}`, {}, token),

  rename: (id, title, token) =>
    request(`/api/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }, token),

  delete: (id, token) =>
    request(`/api/sessions/${id}`, { method: 'DELETE' }, token),

  attachDocument: (sessionId, documentId, token) =>
    request(`/api/sessions/${sessionId}/documents`, {
      method: 'POST',
      body: JSON.stringify({ document_id: documentId }),
    }, token),

  detachDocument: (sessionId, documentId, token) =>
    request(`/api/sessions/${sessionId}/documents/${documentId}`, { method: 'DELETE' }, token),
    
  getMessages: (sessionId, token) =>
    request(`/api/sessions/${sessionId}/messages`, {}, token),
    
  // POST for SSE stream
  sendMessageStream: (sessionId, content, token) =>
    request(`/api/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    }, token),
};

// ── Documents ─── /api/documents/* ────────────────────────────────────────
export const docsApi = {
  upload: (file, token) => {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${BASE}/api/documents/upload`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    }).then(async (res) => {
      const body = await res.json();
      if (!res.ok) throw new Error(body?.detail ?? `HTTP ${res.status}`);
      return body;
    });
  },

  getStatus: (documentId, token) =>
    request(`/api/documents/${documentId}/status`, {}, token),
};

// ── Backward Compat / Aliases ─────────────────────────────────────────────
export const chatApi = {
  ask: (sessionId, content, token) => sessionsApi.sendMessageStream(sessionId, content, token),
  history: (sessionId, token) => sessionsApi.getMessages(sessionId, token),
};
