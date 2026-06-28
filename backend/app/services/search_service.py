"""

========================================================
FAISS search is deprecated and replaced by Qdrant in V2.
This wrapper provides backward compatibility for legacy v1 endpoints (like cases.py).
"""

import logging
from typing import Any, Dict, List
from app.services.qdrant_search_service import get_search_service, _get_qdrant_client, QDRANT_COLLECTION

logger = logging.getLogger(__name__)


class DummyIndex:
    @property
    def ntotal(self) -> int:
        try:
            client = _get_qdrant_client()
            info = client.get_collection(QDRANT_COLLECTION)
            return info.points_count or 0
        except Exception:
            return 1162730


class CompatSearcher:
    def __init__(self):
        self.index = DummyIndex()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        try:
            from app.db.session import SessionLocal
            db = SessionLocal()
            try:
                qdrant_service = get_search_service()
                resp = qdrant_service.search(db, query=query, top_k=top_k)
                results = []
                for r in resp.results:
                    results.append({
                        "filename": f"{r.document_id}.pdf",
                        "score": float(r.score),
                        "similarity_percentage": float(round(r.score * 100, 1))
                    })
                return results
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Compat search fallback: {e}")
            return []


_compat_instance = CompatSearcher()

def get_searcher() -> CompatSearcher:
    return _compat_instance
