from types import SimpleNamespace
from unittest.mock import MagicMock

from app.agents.nodes.document_chat import _resolve_corpus_document_id


def test_resolve_corpus_document_id_uses_filename_for_legacy_attachment():
    db = MagicMock()
    db.execute.return_value.scalar.return_value = "00000000-0000-0000-0000-000000000123"
    document = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000999",
        blob_path="/app/data/pdfs/example-case.pdf",
    )

    result = _resolve_corpus_document_id(db, document)

    assert result == "00000000-0000-0000-0000-000000000123"
    assert db.execute.call_args.args[1] == {"filename": "example-case.pdf"}
