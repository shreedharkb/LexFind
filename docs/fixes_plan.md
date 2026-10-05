# LexFind — Project Status Audit & Pending Fixes

Full audit of LexFind features, what's working, what's broken, and what still needs to be done — organized for mini project **mid-term** and **end-term** evaluation.

---

## Current Feature Status

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | **Landing Page** | ✅ Working | Hero, feature cards, CTA, mockup preview |
| 2 | **Auth (Register / Login)** | ✅ Working | JWT-based, email+password, protected routes |
| 3 | **Navigation + Routing** | ✅ Working | Responsive navbar, mobile menu, user dropdown |
| 4 | **Semantic Search (Qdrant)** | ✅ Working | 46k cases, hybrid mode, keyword toggle |
| 5 | **Search Results Display** | ✅ Working | Title, court, year, similarity %, chunk preview |
| 6 | **PDF Viewer Modal** | ⚠️ Partially | Works for local PDFs; Azure blob fetch may fail on CORS |
| 7 | **PDF Download** | ⚠️ Partially | Same Azure/local fallback issue as viewer |
| 8 | **Analyze in Assistant** | ✅ Working | Creates session, attaches doc, navigates to assistant |
| 9 | **AI Chat (General Legal Q&A)** | ✅ Working | LangGraph classifier → general_chat node → Groq LLM |
| 10 | **AI Chat (Corpus Search)** | ✅ Working | Searches Qdrant corpus, synthesizes multi-case answer |
| 11 | **AI Chat (Document-Specific)** | ✅ Working | Routes to Qdrant (legal_case) or pgvector (uploaded) |
| 12 | **SSE Streaming Response** | ⚠️ Partial | Sends full answer as single chunk, NOT token-by-token streaming |
| 13 | **Citations in Chat** | ✅ Working | Built from Qdrant payload, shown with View/Download buttons |
| 14 | **Session Management** | ✅ Working | Create, list, rename (auto-title), delete with confirmation |
| 15 | **PDF Upload (Private Docs)** | ⚠️ Needs Celery | Uploads to Azure Blob, but needs Celery+RabbitMQ running for processing |
| 16 | **Document Processing Pipeline** | ⚠️ Needs Celery | PyMuPDF extract → LangChain chunk → embed → pgvector |
| 17 | **Chat with Uploaded PDF** | ⚠️ Needs Processing | Only works after Celery processes the doc to "ready" status |
| 18 | **Session Document Attachments** | ✅ Working | Attach/detach, status polling, UI cards |
| 19 | **Intent Classifier (Guardrail)** | ✅ Working | Blocks non-legal queries, routes to correct node |
| 20 | **Azure Blob Storage** | ✅ Configured | Connection string set, upload/download service exists |
| 21 | **Docker Compose** | ✅ Available | PostgreSQL, RabbitMQ, Qdrant, backend services |
| 22 | **Vercel Deployment (Frontend)** | ✅ Configured | `vercel.json` with API rewrite proxy |
| 23 | **Azure VM Deployment (Backend)** | ✅ Configured | Nginx + SSL, deploy scripts exist |

---

## 🔴 Issues That Need Fixing

### 1. SSE Streaming Is Fake (Single-Chunk Delivery)

**File:** [sessions.py](file:///d:/LexFind/backend/app/api/sessions.py#L179-L211)

The `generate()` function sends the **entire answer as one chunk** after the LangGraph graph finishes. The frontend streaming code (SSE reader) works correctly, but the backend doesn't actually stream tokens — it waits for the full `ainvoke()` to complete.

**Impact:** The UI hangs with a spinner until the full response is ready (5-15 seconds), instead of showing text progressively like ChatGPT/Claude.

**Fix needed:** Either:
- **(A)** Switch to `astream_events` from LangGraph to emit partial tokens, OR
- **(B)** Use Groq's `stream=True` mode in each agent node and yield chunks back through SSE

> [!IMPORTANT]
> This is the most noticeable UX issue. Sagar's project has real token-by-token streaming — yours doesn't.

---

### 2. PDF Viewer / Download Fails for Azure-Hosted PDFs

**Files:** [search.py](file:///d:/LexFind/backend/app/api/search.py#L262-L312), [cases.py](file:///d:/LexFind/backend/app/api/cases.py#L309-L344)

The `serve_pdf` endpoints check local disk first, then fall back to Azure Blob. But:
- On production (VM without 46k PDFs on disk), it tries Azure — which works for the backend but the **frontend blob-fetch** may fail due to CORS or auth headers not being passed correctly.
- The search page's PDF viewer (`PdfViewerModal`) does a plain `fetch(url)` without auth headers, but `/api/search/pdf/{id}` doesn't require auth. The documents PDF endpoint (`/api/documents/{id}/pdf`) **does** require auth but the `AssistantPage` version passes the token correctly.

**Impact:** "Could not load PDF preview" error for many search result PDFs when running in production.

**Fix needed:**
- Ensure Azure Blob download works reliably with proper error handling
- Consider adding a streaming proxy that handles large PDFs without loading them fully into memory

---

### 3. Celery Worker Not Running Locally (PDF Upload Broken)

**Dependency:** Requires RabbitMQ + Celery worker running

When the user uploads a PDF, the backend calls `process_document_task.delay()` which sends a Celery task. Without a Celery worker running, the document stays in "processing" state forever.

**Impact:** The "Upload PDF" feature in the Assistant appears to work (file stages, uploads to Azure) but the document never becomes "ready", so the AI can't chat with it.

**Fix needed:**
- **(A)** For **local dev**: Start Celery and RabbitMQ via Docker (`docker compose up rabbitmq -d` + run the worker), OR
- **(B)** Add a **synchronous fallback** that processes small PDFs inline without Celery (for demo purposes)

> [!WARNING]
> This is a critical gap for the evaluation demo. If you can't show PDF upload → processing → chat, it's a major missing feature.

---

### 4. Session Title Auto-Rename Only Uses First Message

**File:** [sessions.py](file:///d:/LexFind/backend/app/api/sessions.py#L152-L154)

The session title is set to the first 40 characters of the first message. Sagar's project uses the LLM to generate a meaningful title (like "Contract Breach Analysis" instead of "What are the key provisions of...").

**Fix needed:** After the first AI response, call Groq with a short prompt to generate a 4-6 word session title.

---

### 5. No Loading Skeleton / Empty States Polish

The search page shows a blank area when no results exist (only "Popular Topics" chips). The assistant shows a basic empty state. These could be improved with:
- Animated skeleton loaders during search
- Better empty state illustrations
- Subtle entry animations for search results

---

### 6. No Search Filters UI (Court, Year, State, Case Type)

**Backend supports it:** [search.py](file:///d:/LexFind/backend/app/api/search.py#L49-L102) accepts `court`, `year_min`, `year_max`, `state`, `case_type`, `section_type` filters.

**Frontend doesn't use it:** [SearchPage.jsx](file:///d:/LexFind/frontend/src/pages/SearchPage.jsx#L116-L131) only sends `query`, `top_k`, and `search_mode`.

**Fix needed:** Add a collapsible filter panel below the search bar with dropdowns/inputs for court, year range, state, and case type.

> [!IMPORTANT]
> Sagar's project has search filters. This is an easy win for evaluation.

---

### 7. No "Similar Cases" Feature in UI

**Backend supports it:** [search.py](file:///d:/LexFind/backend/app/api/search.py#L182-L211) has `GET /api/search/similar/{document_id}`.

**Frontend doesn't use it.** There's no "Find Similar Cases" button on search results or in the assistant.

**Fix needed:** Add a "Similar Cases" button on each search result card that calls the API and shows related cases.

---

### 8. No Search-by-Name Feature in UI

**Backend supports it:** [search.py](file:///d:/LexFind/backend/app/api/search.py#L107-L141) has `POST /api/search/by-name`.

**Frontend doesn't use it.**

**Fix needed:** Add a toggle or separate input for "Search by Party Name" alongside the main search.

---

## 🟡 Minor Issues / Polish Items

| # | Issue | Priority |
|---|-------|----------|
| 1 | `LoginPage.jsx` — no password visibility toggle | Low |
| 2 | No "Forgot Password" flow (no endpoint exists) | Low |
| 3 | No user profile/settings page | Low |
| 4 | Session rename UI is missing (API exists via PATCH) | Medium |
| 5 | No pagination for search results (always returns max `top_k`) | Medium |
| 6 | No error boundary / 404 page — catch-all redirects to `/` | Low |
| 7 | Footer component is basic, could use more content | Low |
| 8 | No dark mode toggle | Low |
| 9 | The v1 FAISS-based `/api/cases/search` endpoint still exists alongside the v2 Qdrant `/api/search` — dead code | Low |
| 10 | Tests are minimal — only 2 test files (`test_routes.py`, `test_azure_integration.py`) | Medium |

---

## Recommended Priority for Evaluation

### Mid-Term Checklist (Show These Working)

- [x] Landing page with hero + features
- [x] User registration + login
- [x] Semantic search with results
- [x] "Analyze in Assistant" flow (search → chat)
- [x] AI chat with citations
- [x] Session management (create, list, delete)
- [ ] **Add search filters UI** (court, year, case type) — Easy win
- [ ] **Fix SSE streaming** to show tokens progressively — High impact

### End-Term Checklist (Full Feature Parity with Sagar's Project)

- [ ] **Real token-by-token streaming** in chat
- [ ] **PDF upload → process → chat** working end-to-end (needs Celery or sync fallback)
- [ ] **Search filters** (court, year, state, case type)
- [ ] **Similar cases** feature
- [ ] **Search by party name** feature
- [ ] **LLM-generated session titles**
- [ ] **Search result pagination**
- [ ] Polished loading states + animations
- [ ] Better error handling + error boundaries
- [ ] More comprehensive tests

---

## Open Questions

> [!IMPORTANT]
> **Which features do you want me to fix first?** I can start with the highest-impact items:
> 1. Add search filters UI (easy, 1-2 hours)
> 2. Fix streaming to show tokens progressively (medium, 2-3 hours)
> 3. Add synchronous PDF processing fallback for demos (medium, 1-2 hours)
> 4. Add similar cases + search by name buttons (easy, 1 hour each)

> [!NOTE]
> **About Sagar's project:** To match his features exactly, I'd need to know which specific features you've seen in his project that are missing here. If you can share his project URL or screenshots, I can do a precise feature-by-feature comparison.
