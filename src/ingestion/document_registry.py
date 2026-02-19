"""
Document Registry for tracking indexed documents.

Enables incremental indexing:
- Track which files are indexed
- Detect new/changed/deleted files
- Support add/update/remove operations
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DocumentRecord:
    """Record of an indexed document."""
    file_hash: str
    filename: str
    filepath: str
    status: str  # "indexed", "pending", "failed", "deleted"
    indexed_at: Optional[str] = None
    file_size: int = 0
    page_count: int = 0
    chunk_count: int = 0
    chunk_ids: list = field(default_factory=list)
    domain: Optional[str] = None
    topics: list = field(default_factory=list)
    schema_version: str = "1.0"
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRecord":
        return cls(**data)


class DocumentRegistry:
    """
    Track all indexed documents for incremental updates.
    
    Features:
    - File hash-based change detection
    - Incremental indexing (only process new/changed files)
    - Document lifecycle management (add/update/remove)
    - Statistics and reporting
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self, registry_path: str = "data/processed/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load registry from disk or create new one."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "version": self.SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "documents": {},
            "stats": {
                "total_documents": 0,
                "total_chunks": 0,
                "indexed": 0,
                "pending": 0,
                "failed": 0,
            },
        }

    def _save_registry(self):
        """Save registry to disk."""
        self._registry["last_updated"] = datetime.now().isoformat()
        self._update_stats()
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, ensure_ascii=False, indent=2)

    def _update_stats(self):
        """Update statistics based on current documents."""
        docs = self._registry["documents"]
        self._registry["stats"] = {
            "total_documents": len(docs),
            "total_chunks": sum(d.get("chunk_count", 0) for d in docs.values()),
            "indexed": sum(1 for d in docs.values() if d.get("status") == "indexed"),
            "pending": sum(1 for d in docs.values() if d.get("status") == "pending"),
            "failed": sum(1 for d in docs.values() if d.get("status") == "failed"),
        }

    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """Compute SHA256 hash of file content."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_document(self, filepath: str) -> Optional[DocumentRecord]:
        """Get document record by filepath."""
        file_hash = self.compute_file_hash(filepath)
        if file_hash in self._registry["documents"]:
            return DocumentRecord.from_dict(self._registry["documents"][file_hash])
        return None

    def is_indexed(self, filepath: str) -> bool:
        """Check if file is already indexed."""
        doc = self.get_document(filepath)
        return doc is not None and doc.status == "indexed"

    def needs_update(self, filepath: str) -> bool:
        """Check if file needs to be (re)indexed."""
        path = Path(filepath)
        if not path.exists():
            return False
        doc = self.get_document(filepath)
        if doc is None:
            return True  # New file
        if doc.status in ("pending", "failed"):
            return True  # Retry
        return False  # Already indexed

    def get_pending_files(self, directory: str, pattern: str = "*.pdf") -> list[str]:
        """Get list of files that need indexing."""
        dir_path = Path(directory)
        all_files = list(dir_path.glob(pattern))
        return [str(f) for f in all_files if self.needs_update(str(f))]

    def get_all_indexed(self) -> list[DocumentRecord]:
        """Get all indexed documents."""
        return [
            DocumentRecord.from_dict(d)
            for d in self._registry["documents"].values()
            if d.get("status") == "indexed"
        ]

    def register_pending(self, filepath: str) -> DocumentRecord:
        """Register a file as pending indexing."""
        path = Path(filepath)
        file_hash = self.compute_file_hash(filepath)

        record = DocumentRecord(
            file_hash=file_hash,
            filename=path.name,
            filepath=str(path.absolute()),
            status="pending",
            file_size=path.stat().st_size,
        )

        self._registry["documents"][file_hash] = record.to_dict()
        self._save_registry()
        return record

    def register_indexed(
        self,
        filepath: str,
        chunk_ids: list[str],
        page_count: int,
        domain: Optional[str] = None,
        topics: Optional[list[str]] = None,
    ) -> DocumentRecord:
        """Register a successfully indexed document."""
        path = Path(filepath)
        file_hash = self.compute_file_hash(filepath)

        record = DocumentRecord(
            file_hash=file_hash,
            filename=path.name,
            filepath=str(path.absolute()),
            status="indexed",
            indexed_at=datetime.now().isoformat(),
            file_size=path.stat().st_size,
            page_count=page_count,
            chunk_count=len(chunk_ids),
            chunk_ids=chunk_ids,
            domain=domain,
            topics=topics or [],
            schema_version=self.SCHEMA_VERSION,
        )

        self._registry["documents"][file_hash] = record.to_dict()
        self._save_registry()
        return record

    def register_failed(self, filepath: str, error_message: str) -> DocumentRecord:
        """Register a failed indexing attempt."""
        path = Path(filepath)
        file_hash = self.compute_file_hash(filepath)

        record = DocumentRecord(
            file_hash=file_hash,
            filename=path.name,
            filepath=str(path.absolute()),
            status="failed",
            file_size=path.stat().st_size,
            error_message=error_message,
        )

        self._registry["documents"][file_hash] = record.to_dict()
        self._save_registry()
        return record

    def remove_document(self, filepath: str) -> Optional[list[str]]:
        """
        Remove document from registry.

        Returns:
            List of chunk IDs that should be removed from vector DB,
            or None if document not found.
        """
        file_hash = self.compute_file_hash(filepath)

        if file_hash not in self._registry["documents"]:
            return None

        doc = self._registry["documents"].pop(file_hash)
        self._save_registry()
        return doc.get("chunk_ids", [])

    def mark_deleted(self, filepath: str) -> bool:
        """Mark a document as deleted (soft delete)."""
        file_hash = self.compute_file_hash(filepath)

        if file_hash not in self._registry["documents"]:
            return False

        self._registry["documents"][file_hash]["status"] = "deleted"
        self._save_registry()
        return True

    def get_stats(self) -> dict:
        """Get registry statistics."""
        self._update_stats()
        return self._registry["stats"]

    def get_documents_by_domain(self, domain: str) -> list[DocumentRecord]:
        """Get all documents for a specific domain."""
        return [
            DocumentRecord.from_dict(d)
            for d in self._registry["documents"].values()
            if d.get("domain") == domain and d.get("status") == "indexed"
        ]

    def get_chunk_ids_by_domain(self, domain: str) -> list[str]:
        """Get all chunk IDs for a specific domain."""
        chunk_ids = []
        for doc in self._registry["documents"].values():
            if doc.get("domain") == domain and doc.get("status") == "indexed":
                chunk_ids.extend(doc.get("chunk_ids", []))
        return chunk_ids

    def cleanup_missing_files(self) -> list[str]:
        """
        Find and mark documents whose files no longer exist.

        Returns:
            List of filepaths that were marked as deleted.
        """
        deleted = []
        for file_hash, doc in list(self._registry["documents"].items()):
            if doc.get("status") == "indexed":
                if not Path(doc["filepath"]).exists():
                    self._registry["documents"][file_hash]["status"] = "deleted"
                    deleted.append(doc["filepath"])

        if deleted:
            self._save_registry()
        return deleted

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DocumentRegistry("
            f"indexed={stats['indexed']}, "
            f"pending={stats['pending']}, "
            f"failed={stats['failed']}, "
            f"chunks={stats['total_chunks']})"
        )

