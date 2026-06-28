"""
Azure Blob Storage service for LexFind.
Handles PDF upload, download, deletion, and existence checks.

"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    _AZURE_AVAILABLE = True
except ImportError:
    _AZURE_AVAILABLE = False


class BlobStorageError(Exception):
    pass


class BlobStorageService:
    def __init__(self, connection_string: Optional[str] = None, container_name: Optional[str] = None, use_local: Optional[bool] = None):
        self._conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self._container = container_name or os.getenv("AZURE_STORAGE_CONTAINER_NAME", os.getenv("AZURE_DATA_CONTAINER", "documents"))
        self._use_local = use_local if use_local is not None else (os.getenv("USE_LOCAL_FILES", "true").lower() == "true")
        
        _backend_dir = Path(__file__).resolve().parent.parent.parent
        self._local_root = _backend_dir / "data" / "uploaded_documents"
        self._local_root.mkdir(parents=True, exist_ok=True)
        self._blob_client = None

    def _get_blob_client(self):
        if not _AZURE_AVAILABLE: raise BlobStorageError("azure-storage-blob package is not installed.")
        if not self._conn_str: raise BlobStorageError("AZURE_STORAGE_CONNECTION_STRING is not set.")
        if self._blob_client is None: self._blob_client = BlobServiceClient.from_connection_string(self._conn_str)
        return self._blob_client

    @staticmethod
    def _generate_blob_path(user_id: str) -> str:
        return f"documents/{user_id}/{uuid.uuid4()}.pdf"

    def _local_path(self, blob_path: str) -> Path:
        if os.path.isabs(blob_path): return Path(blob_path)
        if blob_path.startswith("data/pdfs/"): return Path(__file__).resolve().parent.parent.parent / blob_path
        return self._local_root / blob_path

    def upload_pdf(self, file_bytes: bytes, user_id: str) -> str:
        blob_path = self._generate_blob_path(user_id)
        if self._use_local:
            dest = self._local_path(blob_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(file_bytes)
            return blob_path

        try:
            client = self._get_blob_client()
            blob = client.get_blob_client(container=self._container, blob=blob_path)
            blob.upload_blob(file_bytes, overwrite=False, content_settings=ContentSettings(content_type="application/pdf"))
            return blob_path
        except Exception as exc:
            raise BlobStorageError(f"Failed to upload PDF: {exc}") from exc

    def download_pdf(self, blob_path: str) -> bytes:
        if self._use_local:
            dest = self._local_path(blob_path)
            if not dest.exists(): raise BlobStorageError(f"Local file not found: {blob_path}")
            return dest.read_bytes()
        try:
            client = self._get_blob_client()
            return client.get_blob_client(container=self._container, blob=blob_path).download_blob().readall()
        except Exception as exc:
            raise BlobStorageError(f"Failed to download PDF '{blob_path}': {exc}") from exc

    def delete_pdf(self, blob_path: str) -> None:
        if self._use_local:
            dest = self._local_path(blob_path)
            if dest.exists(): dest.unlink()
            try: dest.parent.rmdir(); dest.parent.parent.rmdir()
            except OSError: pass
            return

        try:
            client = self._get_blob_client()
            client.get_blob_client(container=self._container, blob=blob_path).delete_blob(delete_snapshots="include")
        except Exception as exc:
            raise BlobStorageError(f"Failed to delete PDF '{blob_path}': {exc}") from exc

    def blob_exists(self, blob_path: str) -> bool:
        if self._use_local: return self._local_path(blob_path).exists()
        try:
            return self._get_blob_client().get_blob_client(container=self._container, blob=blob_path).exists()
        except Exception:
            return False

    def store_local_file(self, file_path: str, user_id: str) -> str:
        blob_path = self._generate_blob_path(user_id)
        dest = self._local_path(blob_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        return blob_path


blob_storage_service = BlobStorageService()
