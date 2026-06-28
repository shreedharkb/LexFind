"""
Document Processing Service for LexFind V2.
Synchronous pipeline: extraction -> chunking -> embedding -> summarization

"""

import logging
import os
import uuid
from typing import List, Tuple

import fitz  # PyMuPDF
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db.models import DocumentChunk, DocumentEmbedding
from app.db.session import DatabaseSession
from app.services.blob_storage_service import blob_storage_service
from app.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

class DocumentProcessingError(Exception):
    pass


class DocumentProcessingService:
    @staticmethod
    def extract_text(pdf_bytes: bytes) -> List[Tuple[str, int]]:
        idx = pdf_bytes.find(b"%PDF-")
        if idx > 0: pdf_bytes = pdf_bytes[idx:]
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise DocumentProcessingError(f"Failed to open PDF: {exc}") from exc

        pages = [(text, i + 1) for i in range(len(doc)) if (text := doc[i].get_text("text").strip())]
        doc.close()

        if not pages: raise DocumentProcessingError("PDF contains no extractable text.")
        return pages

    @staticmethod
    def chunk_text(pages: List[Tuple[str, int]]) -> List[Tuple[str, int, int]]:
        chunks = []
        chunk_idx = 0
        for text, page_num in pages:
            for chunk in _SPLITTER.split_text(text):
                if chunk.strip():
                    chunks.append((chunk.strip(), page_num, chunk_idx))
                    chunk_idx += 1
        return chunks

    @staticmethod
    def store_chunks_and_embeddings(document_id: uuid.UUID, chunks: List[Tuple[str, int, int]], db) -> int:
        if not chunks: return 0
        embeddings = embed_texts([c[0] for c in chunks])

        chunk_records = [
            DocumentChunk(id=uuid.uuid4(), document_id=document_id, page_number=p, chunk_index=i, chunk_text=t, chunk_metadata={"char_count": len(t)})
            for (t, p, i) in chunks
        ]
        db.add_all(chunk_records)
        db.flush()

        embedding_records = [
            DocumentEmbedding(id=uuid.uuid4(), chunk_id=c.id, document_id=document_id, embedding=v.tolist())
            for c, v in zip(chunk_records, embeddings)
        ]
        db.add_all(embedding_records)
        db.commit()
        return len(chunk_records)

    @staticmethod
    def generate_summary(full_text: str) -> str:
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if not api_key: return "Summary generation skipped (no API key configured)."
        
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": f"You are an expert legal analyst. Provide a brief, structured summary of the following legal document excerpt. Highlight the main topic, parties involved, and the core issue or ruling.\n\nDocument text:\n{full_text[:8000]}"
                }],
                temperature=0.1, max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    def process_document(self, document_id: str, blob_path: str) -> None:
        doc_uuid = uuid.UUID(document_id)
        with DatabaseSession() as db:
            from app.db.crud.document_repository import get_document, update_status
            if not (doc := get_document(db, doc_uuid)) or doc.status not in ("uploaded", "processing", "failed"): return
            update_status(db, doc, "processing")

        try:
            pdf_bytes = blob_storage_service.download_pdf(blob_path)
            pages = self.extract_text(pdf_bytes)
            chunks = self.chunk_text(pages)
            if not chunks: raise DocumentProcessingError("No text chunks produced.")

            with DatabaseSession() as db: self.store_chunks_and_embeddings(doc_uuid, chunks, db)
            summary = self.generate_summary("\n\n".join(t for t, _ in pages))

            with DatabaseSession() as db:
                if doc := get_document(db, doc_uuid): update_status(db, doc, "ready", summary=summary)
            logger.info("Document %s processed successfully.", document_id)
        except Exception as exc:
            logger.exception("Processing failed for document %s: %s", document_id, exc)
            try:
                with DatabaseSession() as db:
                    if doc := get_document(db, doc_uuid): update_status(db, doc, "failed", error_message=str(exc))
            except Exception: pass
            raise exc

document_processing_service = DocumentProcessingService()
