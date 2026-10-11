from unittest.mock import MagicMock

from app.agents.nodes import classifier


def test_attached_document_session_routes_to_document_chat(monkeypatch):
    """An Analyze session must not send attachment messages to general chat."""
    groq_client = MagicMock()
    monkeypatch.setattr(classifier, "get_groq_client", groq_client)

    result = classifier.classifier_node({
        "question": "I have attached the case",
        "explicit_mode": "auto",
        "document_ids": ["00000000-0000-0000-0000-000000000001"],
    })

    assert result["is_legal"] is True
    assert result["intent"] == "document_chat"
    groq_client.assert_not_called()
