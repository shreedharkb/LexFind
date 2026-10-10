"""
Node 9 — WebAugment

Runs when needs_web_search = True (after grader PASS + recency check).
Fetches recent web results and stores in state["web_results"].
Response generation node merges these with retrieved database chunks.

Failure policy: On any exception, set web_results=[] and continue.
Never crash the pipeline — web augmentation is optional enrichment, not critical path.

To activate: set TAVILY_API_KEY in .env
"""
import logging
import os

from app.agents.state import LexFindState

logger = logging.getLogger(__name__)


def _fetch_web_results(query: str) -> list:
    """Fetch web search results via Tavily. Falls back to empty list."""
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_key:
        logger.debug("WebAugment: no TAVILY_API_KEY set — skipping web search")
        return []
    try:
        from tavily import TavilyClient  # type: ignore
        client  = TavilyClient(api_key=tavily_key)
        results = client.search(
            query=f"{query} India law court judgment",
            max_results=3,
            search_depth="basic",
        )
        web_results = []
        for r in results.get("results", []):
            web_results.append({
                "source": "web",
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", "")[:600],
            })
        logger.info("WebAugment: fetched %d results for query '%s'", len(web_results), query[:60])
        return web_results
    except Exception as exc:
        logger.warning("WebAugment: Tavily fetch failed (%s)", exc)
        return []


def web_augment_node(state: LexFindState) -> LexFindState:
    """Fetch web results and attach to state. Safe — never crashes pipeline."""
    query = state.get("enhanced_query") or state["question"]
    try:
        web_results = _fetch_web_results(query)
    except Exception as exc:
        logger.warning("WebAugment: unexpected error (%s)", exc)
        web_results = []
        
    web_citations = []
    for r in web_results:
        web_citations.append({
            "source": "web",
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "excerpt": r.get("content", "")[:200] + "...",
            "document_title": r.get("title", ""),
            "filename": r.get("url", ""),
        })
        
    current_citations = state.get("citations") or []
    return {**state, "web_results": web_results, "citations": current_citations + web_citations}
