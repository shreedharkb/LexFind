# Local Demo Script

A practical walkthrough for demonstrating LexFind locally (e.g. for a project evaluation). Assumes the local Docker Compose stack is running — see [setup_and_installation.md](setup_and_installation.md) for full setup instructions.

## Before the demo

1. `docker compose up -d --build` from the repository root and wait for the stack to be healthy.
2. Verify: `curl http://localhost:8000/api/health` returns OK and `curl http://localhost:6333/collections` lists `legal_corpus_hybrid`.
3. Start the frontend: `cd frontend && npm run dev` and open `http://localhost:5173`.
4. Register a fresh account (or log in) beforehand so the demo starts on the search page, not the signup form.
5. Keep the fallback plan (bottom of this page) open in a second tab.

## The 6-step demo

### Step 1 — Registration & auth (10 seconds)
Log in / register. Point out that the API is fully authenticated with JWT, sessions are isolated per user, and the rate limiter is per-user.

### Step 2 — Semantic search (60 seconds)
Run a natural-language query, e.g. *"right to privacy and fundamental rights"*. Explain: the query is embedded with `all-mpnet-base-v2` and searched against 46,456 judgments in Qdrant with **hybrid retrieval** (dense + BM25 sparse fused via Reciprocal Rank Fusion). Show:
- the ranked results with court, year, and similarity score;
- the **Keyword Search** toggle (pure BM25) and the **party name / case title** toggle;
- the **Court / State / Case Type / Year** filters narrowing results.

### Step 3 — Read a case (30 seconds)
Open a result's **View PDF** to show the judgment text, then **Similar Cases** to show the vector-based related-cases modal.

### Step 4 — Document-specific chat (60 seconds)
Click **Analyze** on a result — it creates a chat session, attaches the case, and opens the Assistant page. Ask *"Summarize the facts of this case"*. Explain: the LangGraph agent classified the intent as document chat and retrieved only from this document (Qdrant filtered by `document_id`). Show the **Retrieved Sources** citations under the answer, with View PDF working for each source.

### Step 5 — Full-corpus synthesis (60 seconds)
In the same or a new session, ask a cross-case question, e.g. *"How has the Supreme Court's position on privacy evolved across cases?"*. Explain: the agent routes this to corpus search — hybrid retrieval over the full 46k corpus, top chunks deduplicated per case, and a synthesized answer citing multiple judgments inline.

### Step 6 — Private PDF upload + chat (90 seconds)
In a session, use **Upload** to attach a small PDF (have a 1–3 page PDF ready). Explain: a Celery worker extracts text with PyMuPDF, chunks it, embeds it, and stores it in the **private pgvector** store isolated by `owner_id`. Once status shows **ready**, ask a question about the file and show the grounded, cited answer.

Optional extras, time permitting:
- **General legal Q&A**: ask a knowledge question with no document attached (e.g. *"What is Article 21?"*) — no retrieval path.
- **Guardrail**: ask a non-legal question (e.g. *"Who won the cricket match?"*) and show the polite refusal — the classifier blocks off-topic input.
- **Session persistence**: reload the page and show the conversation history is still there (stored in PostgreSQL).

## Fallback plan (if something fails live)

| Symptom | Likely cause | What to do / say |
|---|---|---|
| Search returns no results or an error | Qdrant container down, or corpus not loaded | Restart Qdrant (`docker compose restart qdrant`). The corpus lives outside the repo in the git-ignored `data/` dir; if it was never loaded, demo search is unavailable — switch to Steps 4–6 with uploaded PDFs and say corpus indexing ran during development (46,456 judgments, ~1.16M chunks). |
| Chat hangs or errors | Groq API key missing / quota exhausted | Check `backend/.env` `GROQ_API_KEY`; the free tier has daily limits. Say LLM inference is via Groq's API, which needs internet + an API key. |
| Uploaded PDF stuck at "processing" | Celery worker down | Restart the worker (`docker compose restart celery-worker`). On Windows without Docker, run Celery with `-P solo`. |
| PDF viewer blank | CORS / blob fetch issue | The modal fetches the PDF as a blob with the JWT — hard-refresh the page; if it persists, demonstrate downloads instead. |
| 429 responses | Redis rate limiter tripped | Note that per-user rate limiting is a feature; wait 60 seconds or restart Redis to clear counters. |
| Login fails after stack restart | Database not migrated | `docker compose logs api` — the API container runs `alembic upgrade head` on start; give it a few seconds. |

## Honest talking points

- The 46k-judgment corpus is **not committed to the repository** (git-ignored); search demos depend on the locally loaded Qdrant collection.
- LLM calls go to **Groq's API** — chat needs an internet connection and a valid key.
- Hybrid retrieval raised excerpt Hit@5 from 35.9% (dense-only) to **82.3%** in our evaluation (192 sampled judgments × 3 query types); title-only queries remain the weakest category (26.0% Hit@5) — metadata filters are the recommended workaround. Full numbers: [rag_metrics.md](rag_metrics.md).
