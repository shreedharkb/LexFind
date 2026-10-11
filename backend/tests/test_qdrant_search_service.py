from unittest.mock import MagicMock

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVector,
)

from app.schemas.search_schemas import CaseResult, ChunkResult
from app.services import qdrant_search_service as search_service


def _case_result(document_id="document-1"):
    chunk = ChunkResult(
        chunk_id="chunk-1",
        chunk_text="A case excerpt.",
        chunk_index=0,
        section_type="facts",
        score=1.0,
    )
    return CaseResult(
        document_id=document_id,
        title="Example v State",
        score=1.0,
        top_chunk=chunk,
        all_chunks=[chunk],
    )


def test_hybrid_search_uses_named_dense_and_sparse_vectors(monkeypatch):
    response = MagicMock()
    response.json.return_value = {
        "result": {
            "points": [{
                "id": "00000000-0000-0000-0000-000000000001",
                "score": 0.9,
                "payload": {"document_id": "document-1"},
            }]
        }
    }
    query_filter = Filter(
        must=[FieldCondition(key="court", match=MatchValue(value="Supreme Court"))]
    )

    monkeypatch.setattr(search_service, "_embed", lambda _: [0.1, 0.2])
    monkeypatch.setattr(
        search_service,
        "_embed_sparse",
        lambda _: SparseVector(indices=[1, 2], values=[0.3, 0.4]),
    )
    post = MagicMock(return_value=response)
    monkeypatch.setattr(search_service.requests, "post", post)

    results = search_service.qdrant_hybrid_search("constitutional rights", query_filter, 20)

    assert results[0].score == 0.9
    body = post.call_args.kwargs["json"]
    assert body["query"] == {"fusion": "rrf"}
    assert body["prefetch"][0]["using"] == "dense"
    assert body["prefetch"][1]["using"] == "sparse"
    assert body["prefetch"][0]["limit"] == 20
    assert body["prefetch"][1]["limit"] == 20
    assert body["prefetch"][0]["filter"] == {
        "must": [{"key": "court", "match": {"value": "Supreme Court"}}]
    }


def test_hybrid_search_can_return_dense_vectors_for_displayed_similarity(monkeypatch):
    response = MagicMock()
    response.json.return_value = {
        "result": {"points": [{
            "id": "00000000-0000-0000-0000-000000000001",
            "score": 0.03,
            "vector": {"dense": [0.1, 0.2]},
            "payload": {"document_id": "document-1"},
        }]}
    }
    monkeypatch.setattr(search_service, "_embed", lambda _: [0.1, 0.2])
    monkeypatch.setattr(search_service, "_embed_sparse", lambda _: None)
    post = MagicMock(return_value=response)
    monkeypatch.setattr(search_service.requests, "post", post)

    points = search_service.qdrant_hybrid_search(
        "constitutional rights", None, 20, include_dense_vectors=True,
    )

    assert post.call_args.kwargs["json"]["with_vector"] == ["dense"]
    assert points[0].vector == {"dense": [0.1, 0.2]}


def test_hybrid_search_falls_back_to_dense_after_timeout(monkeypatch):
    dense_response = MagicMock()
    dense_response.json.return_value = {
        "result": {
            "points": [{
                "id": "00000000-0000-0000-0000-000000000001",
                "score": 0.8,
                "payload": {"document_id": "document-1"},
            }]
        }
    }

    monkeypatch.setattr(search_service, "_embed", lambda _: [0.1, 0.2])
    monkeypatch.setattr(search_service, "_embed_sparse", lambda _: None)
    post = MagicMock(side_effect=[search_service.requests.Timeout(), dense_response])
    monkeypatch.setattr(search_service.requests, "post", post)

    results = search_service.qdrant_hybrid_search("constitutional rights", None, 20)

    assert results[0].score == 0.8
    assert post.call_count == 2
    assert post.call_args_list[1].kwargs["json"] == {
        "query": [0.1, 0.2],
        "using": "dense",
        "limit": 20,
        "with_payload": True,
    }


def test_hybrid_search_uses_configured_qdrant_url_and_api_key(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"result": {"points": []}}

    monkeypatch.setattr(search_service, "QDRANT_URL", "https://demo.cloud.qdrant.io")
    monkeypatch.setattr(search_service, "QDRANT_API_KEY", "qdrant-api-key")
    monkeypatch.setattr(search_service, "_embed", lambda _: [0.1, 0.2])
    monkeypatch.setattr(search_service, "_embed_sparse", lambda _: None)
    post = MagicMock(return_value=response)
    monkeypatch.setattr(search_service.requests, "post", post)

    search_service.qdrant_hybrid_search("constitutional rights", None, 5)

    assert post.call_args.args[0] == (
        "https://demo.cloud.qdrant.io/collections/legal_corpus_hybrid/points/query"
    )
    assert post.call_args.kwargs["headers"] == {"api-key": "qdrant-api-key"}


def test_search_service_construction_does_not_load_models(monkeypatch):
    model_loader = MagicMock()
    qdrant_loader = MagicMock()
    bm25_loader = MagicMock()
    monkeypatch.setattr(search_service, "_get_embedding_model", model_loader)
    monkeypatch.setattr(search_service, "_get_qdrant_client", qdrant_loader)
    monkeypatch.setattr(search_service, "_get_bm25_model", bm25_loader)

    search_service.QdrantSearchService()

    model_loader.assert_not_called()
    qdrant_loader.assert_not_called()
    bm25_loader.assert_not_called()


def test_keyword_search_uses_sparse_vectors(monkeypatch):
    service = search_service.QdrantSearchService.__new__(search_service.QdrantSearchService)
    result = _case_result()
    sparse_search = MagicMock(return_value=[object()])

    monkeypatch.setattr(search_service, "qdrant_search_by_mode", sparse_search)
    monkeypatch.setattr(search_service, "_qdrant_hits_to_case_results", lambda *_args, **_kwargs: [result])

    response = service.search(MagicMock(), "contract", search_mode="keyword")

    assert response.results == [result]
    assert sparse_search.call_args.kwargs["mode"] == "sparse"


def test_hybrid_search_orders_results_by_displayed_similarity(monkeypatch):
    service = search_service.QdrantSearchService()
    lower_similarity = _case_result("document-low")
    lower_similarity.semantic_score = 0.71
    higher_similarity = _case_result("document-high")
    higher_similarity.semantic_score = 0.92

    monkeypatch.setattr(search_service, "qdrant_hybrid_search", MagicMock(return_value=[]))
    monkeypatch.setattr(search_service, "_embed", lambda _: [0.1, 0.2])
    monkeypatch.setattr(
        search_service,
        "_qdrant_hits_to_case_results",
        lambda *_args, **_kwargs: [lower_similarity, higher_similarity],
    )

    response = service.search(MagicMock(), "privacy rights", top_k=2)

    assert [result.document_id for result in response.results] == [
        "document-high",
        "document-low",
    ]


def test_name_search_uses_database_and_escapes_like_wildcards():
    row = (
        "document-1", "Example v State", "A_% Party", "State", "Supreme Court",
        2020, "2020 SCC 1", ["Judge One"], "Civil Appeal", "Delhi",
        "chunk-1", "A case excerpt.", 0, "facts",
    )
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [row]

    results = search_service._search_by_case_name_in_database(db, "A_%", 10)

    assert len(results) == 1
    assert results[0].petitioner == "A_% Party"
    assert results[0].top_chunk.chunk_id == "chunk-1"
    assert db.execute.call_args.args[1]["pattern"] == "%A" + chr(92) + "_" + chr(92) + "%%"


def test_similar_cases_queries_the_named_dense_vector(monkeypatch):
    client = MagicMock()
    client.scroll.return_value = ([], None)
    monkeypatch.setattr(search_service, "_get_qdrant_client", lambda: client)

    service = search_service.QdrantSearchService()
    response = service.get_similar_cases(MagicMock(), "document-1")

    # No source vectors means Qdrant is not queried; exercise the query path
    # with a single named dense vector below.
    assert response.results == []

    client.scroll.return_value = ([MagicMock(vector={"dense": [0.1, 0.2]})], None)
    client.query_points.return_value.points = []
    service.get_similar_cases(MagicMock(), "document-1")

    assert client.query_points.call_args.kwargs["using"] == "dense"
