"""
Document Indexer - Orchestrates the full ingestion pipeline.

Combines:
- PDFProcessor: Extract text from PDFs
- AdaptiveChunker: Create contextual chunks
- TopicTaxonomy: Classify chunks into topics
- DocumentRegistry: Track indexed documents

Outputs chunks ready for embedding and vector storage.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from .pdf_processor import PDFProcessor, ProcessedDocument
from .topic_taxonomy import TopicTaxonomy
from .document_registry import DocumentRegistry

# Import chunker from processing module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from processing.chunker import AdaptiveChunker

logger = logging.getLogger(__name__)


@dataclass
class IndexedChunk:
    """A chunk ready for embedding and vector storage."""
    id: str
    text: str  # Full text with context
    raw_text: str  # Original text without context

    # Source metadata
    source_file: str
    source_filename: str
    page_num: int
    chunk_index: int

    # Topic metadata
    domain: Optional[str] = None
    topics: list[str] = field(default_factory=list)

    # Structure metadata
    section_path: list[str] = field(default_factory=list)  # ["כיסויים", "נזקי רכוש"]
    content_type: str = "text"  # "text", "header", "table", "list"
    has_table: bool = False

    # Processing metadata
    char_count: int = 0
    chunk_size_used: int = 0
    has_context: bool = False
    previous_summary: str = ""

    # Timestamps
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "raw_text": self.raw_text,
            "source_file": self.source_file,
            "source_filename": self.source_filename,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
            "domain": self.domain,
            "topics": self.topics,
            "section_path": self.section_path,
            "content_type": self.content_type,
            "has_table": self.has_table,
            "char_count": self.char_count,
            "chunk_size_used": self.chunk_size_used,
            "has_context": self.has_context,
            "previous_summary": self.previous_summary,
            "indexed_at": self.indexed_at,
        }


@dataclass
class IndexingResult:
    """Result of indexing a document."""
    filepath: str
    filename: str
    success: bool
    chunks: list[IndexedChunk] = field(default_factory=list)
    page_count: int = 0
    chunk_count: int = 0
    total_chars: int = 0
    domain: Optional[str] = None
    topics: list[str] = field(default_factory=list)
    error: Optional[str] = None
    processing_time_ms: float = 0


class DocumentIndexer:
    """
    Orchestrates the full document ingestion pipeline.
    
    Pipeline:
    1. PDF Processing → Extract text per page
    2. Chunking → Create contextual chunks with summaries
    3. Classification → Assign topics to chunks
    4. Registration → Track in document registry
    """

    def __init__(
        self,
        registry_path: str = "data/processed/registry.json",
        chunks_path: str = "data/processed/chunks.json",
        taxonomy_path: Optional[str] = None,
        use_ocr: bool = False,
    ):
        """
        Initialize the document indexer.

        Args:
            registry_path: Path to document registry JSON
            chunks_path: Path to chunks JSON store
            taxonomy_path: Path to saved taxonomy (optional)
            use_ocr: Enable OCR for scanned PDFs
        """
        self.pdf_processor = PDFProcessor(use_ocr=use_ocr)
        self.chunker = AdaptiveChunker(doc_type="pdf")
        self.taxonomy = TopicTaxonomy()
        self.registry = DocumentRegistry(registry_path)
        self.chunks_path = Path(chunks_path)

        # Load saved taxonomy if provided
        if taxonomy_path and Path(taxonomy_path).exists():
            self.taxonomy = TopicTaxonomy.load(taxonomy_path)

        # Load existing chunks
        self._chunks_store: dict[str, dict] = {}
        self._load_chunks()

        logger.info(f"DocumentIndexer initialized: registry={registry_path}, chunks={chunks_path}")

    def _load_chunks(self):
        """Load existing chunks from JSON file."""
        if self.chunks_path.exists():
            try:
                import json
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for chunk in data.get("chunks", []):
                        self._chunks_store[chunk["id"]] = chunk
                logger.info(f"Loaded {len(self._chunks_store)} existing chunks")
            except Exception as e:
                logger.warning(f"Failed to load chunks: {e}")

    def _save_chunks(self):
        """Save all chunks to JSON file."""
        import json
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "exported_at": datetime.now().isoformat(),
            "total_chunks": len(self._chunks_store),
            "chunks": list(self._chunks_store.values()),
        }

        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(self._chunks_store)} chunks to {self.chunks_path}")

    def _build_section_map(self, doc: ProcessedDocument) -> dict[int, list[str]]:
        """
        Build a map of page numbers to section paths.

        Uses the structured items from the document to track
        which section each page belongs to.

        Args:
            doc: Processed document with structured items

        Returns:
            Dict mapping page_num -> section_path (list of section headers)
        """
        section_map: dict[int, list[str]] = {}
        current_section_path: list[str] = []

        for item in doc.structured_items:
            if item.item_type == "header":
                # Update section path based on header level
                if item.level < len(current_section_path):
                    current_section_path = current_section_path[:item.level]
                current_section_path.append(item.text)

            # Store the current section path for this page
            if item.page_num not in section_map:
                section_map[item.page_num] = current_section_path.copy()

        return section_map

    def index_processed_document(self, doc: ProcessedDocument) -> IndexingResult:
        """
        Index a pre-processed document (PDF or ASPX).

        Args:
            doc: ProcessedDocument from PDFProcessor or ASPXProcessor

        Returns:
            IndexingResult with chunks and metadata
        """
        import time
        start_time = time.time()

        filepath = doc.filepath
        path = Path(filepath)

        # Check if already indexed and up-to-date
        if self.registry.is_indexed(filepath) and not self.registry.needs_update(filepath):
            logger.info(f"Skipping {path.name} - already indexed")
            return IndexingResult(
                filepath=filepath,
                filename=path.name,
                success=True,
                error="Already indexed (skipped)",
            )

        if doc.error:
            self.registry.register_failed(filepath, doc.error)
            return IndexingResult(
                filepath=filepath,
                filename=path.name,
                success=False,
                error=doc.error,
            )

        # Use domain from document or infer from filepath
        domain = doc.domain or self.taxonomy.get_domain_from_filepath(filepath)

        # Build section map from structured items
        section_map = self._build_section_map(doc)

        # Chunk the document
        page_texts = doc.get_page_texts()
        doc_metadata = {
            "source_file": filepath,
            "source_filename": path.name,
            "domain": domain,
            "source_type": doc.metadata.get("source_type", "pdf"),
            "url": doc.metadata.get("url", ""),
        }

        raw_chunks = self.chunker.chunk_document(
            pages=page_texts,
            doc_metadata=doc_metadata,
            carry_context_across_pages=True,
        )

        # Create IndexedChunks with topic classification and structure
        indexed_chunks = []
        all_topics = set()

        for chunk in raw_chunks:
            chunk_id = f"{path.stem}_{uuid.uuid4().hex[:8]}"

            # Classify chunk text into topics
            chunk_topics = self.taxonomy.classify_text(chunk.get("raw_text", ""))
            all_topics.update(chunk_topics)

            # Get section path for this chunk
            page_num = chunk["metadata"]["page"]
            section_path = section_map.get(page_num, [])

            # Detect content type from chunk text
            raw_text = chunk.get("raw_text", chunk["text"])
            content_type = "text"
            has_table = False
            if raw_text.strip().startswith("|") or "[טבלה]" in raw_text:
                content_type = "table"
                has_table = True
            elif raw_text.strip().startswith("•") or raw_text.strip().startswith("-"):
                content_type = "list"
            elif raw_text.strip().startswith("##"):
                content_type = "header"

            indexed_chunk = IndexedChunk(
                id=chunk_id,
                text=chunk["text"],
                raw_text=chunk.get("raw_text", chunk["text"]),
                source_file=filepath,
                source_filename=path.name,
                page_num=page_num,
                chunk_index=chunk["metadata"]["chunk_index"],
                domain=domain,
                topics=chunk_topics[:5],
                section_path=section_path,
                content_type=content_type,
                has_table=has_table,
                char_count=chunk["metadata"]["char_count"],
                chunk_size_used=chunk["metadata"]["chunk_size_used"],
                has_context=chunk["metadata"]["has_context"],
                previous_summary=chunk.get("previous_summary", ""),
            )
            indexed_chunks.append(indexed_chunk)

        # Store chunks
        for chunk in indexed_chunks:
            self._chunks_store[chunk.id] = chunk.to_dict()

        # Register in document registry
        chunk_ids = [c.id for c in indexed_chunks]
        self.registry.register_indexed(
            filepath=filepath,
            chunk_ids=chunk_ids,
            page_count=doc.page_count,
            domain=domain,
            topics=list(all_topics)[:10],
        )

        # Save chunks to disk
        self._save_chunks()

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Indexed {path.name}: {doc.page_count} pages, "
            f"{len(indexed_chunks)} chunks, {doc.total_chars} chars, "
            f"{processing_time:.0f}ms"
        )

        return IndexingResult(
            filepath=filepath,
            filename=path.name,
            success=True,
            chunks=indexed_chunks,
            page_count=doc.page_count,
            chunk_count=len(indexed_chunks),
            total_chars=doc.total_chars,
            domain=domain,
            topics=list(all_topics)[:10],
            processing_time_ms=processing_time,
        )

    def index_document(self, filepath: str) -> IndexingResult:
        """
        Index a single document through the full pipeline.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            IndexingResult with chunks and metadata
        """
        import time
        start_time = time.time()
        
        path = Path(filepath)
        
        # Check if already indexed and up-to-date
        if self.registry.is_indexed(filepath) and not self.registry.needs_update(filepath):
            logger.info(f"Skipping {path.name} - already indexed")
            return IndexingResult(
                filepath=filepath,
                filename=path.name,
                success=True,
                error="Already indexed (skipped)",
            )
        
        # Step 1: Process PDF
        doc = self.pdf_processor.process(filepath)
        if doc.error:
            self.registry.register_failed(filepath, doc.error)
            return IndexingResult(
                filepath=filepath,
                filename=path.name,
                success=False,
                error=doc.error,
            )
        
        # Step 2: Infer domain from filename
        domain = self.taxonomy.get_domain_from_filepath(filepath)

        # Step 3: Build section map from structured items
        # Maps (page_num, char_offset) -> section_path
        section_map = self._build_section_map(doc)

        # Step 4: Chunk the document
        page_texts = doc.get_page_texts()
        doc_metadata = {
            "source_file": filepath,
            "source_filename": path.name,
            "domain": domain,
        }

        raw_chunks = self.chunker.chunk_document(
            pages=page_texts,
            doc_metadata=doc_metadata,
            carry_context_across_pages=True,
        )

        # Step 5: Create IndexedChunks with topic classification and structure
        indexed_chunks = []
        all_topics = set()

        for chunk in raw_chunks:
            chunk_id = f"{path.stem}_{uuid.uuid4().hex[:8]}"

            # Classify chunk text into topics
            chunk_topics = self.taxonomy.classify_text(chunk.get("raw_text", ""))
            all_topics.update(chunk_topics)

            # Get section path for this chunk
            page_num = chunk["metadata"]["page"]
            section_path = section_map.get(page_num, [])

            # Detect content type from chunk text
            raw_text = chunk.get("raw_text", chunk["text"])
            content_type = "text"
            has_table = False
            if raw_text.strip().startswith("|") or "[טבלה]" in raw_text:
                content_type = "table"
                has_table = True
            elif raw_text.strip().startswith("•") or raw_text.strip().startswith("-"):
                content_type = "list"
            elif raw_text.strip().startswith("##"):
                content_type = "header"

            indexed_chunk = IndexedChunk(
                id=chunk_id,
                text=chunk["text"],
                raw_text=chunk.get("raw_text", chunk["text"]),
                source_file=filepath,
                source_filename=path.name,
                page_num=page_num,
                chunk_index=chunk["metadata"]["chunk_index"],
                domain=domain,
                topics=chunk_topics[:5],  # Top 5 topics
                section_path=section_path,
                content_type=content_type,
                has_table=has_table,
                char_count=chunk["metadata"]["char_count"],
                chunk_size_used=chunk["metadata"]["chunk_size_used"],
                has_context=chunk["metadata"]["has_context"],
                previous_summary=chunk.get("previous_summary", ""),
            )
            indexed_chunks.append(indexed_chunk)

        # Step 5: Store chunks
        for chunk in indexed_chunks:
            self._chunks_store[chunk.id] = chunk.to_dict()

        # Step 6: Register in document registry
        chunk_ids = [c.id for c in indexed_chunks]
        self.registry.register_indexed(
            filepath=filepath,
            chunk_ids=chunk_ids,
            page_count=doc.page_count,
            domain=domain,
            topics=list(all_topics)[:10],
        )

        # Save chunks to disk
        self._save_chunks()

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Indexed {path.name}: {doc.page_count} pages, "
            f"{len(indexed_chunks)} chunks, {doc.total_chars} chars, "
            f"{processing_time:.0f}ms"
        )

        return IndexingResult(
            filepath=filepath,
            filename=path.name,
            success=True,
            chunks=indexed_chunks,
            page_count=doc.page_count,
            chunk_count=len(indexed_chunks),
            total_chars=doc.total_chars,
            domain=domain,
            topics=list(all_topics)[:10],
            processing_time_ms=processing_time,
        )

    def index_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        skip_indexed: bool = True,
    ) -> list[IndexingResult]:
        """
        Index all PDFs in a directory.

        Args:
            directory: Path to directory containing PDFs
            pattern: Glob pattern for matching files
            skip_indexed: Skip already indexed files

        Returns:
            List of IndexingResult objects
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        # Get pending files
        if skip_indexed:
            pdf_files = self.registry.get_pending_files(directory, pattern)
        else:
            pdf_files = [str(p) for p in dir_path.glob(pattern)]

        logger.info(f"Indexing {len(pdf_files)} files from {directory}")

        results = []
        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"[{i+1}/{len(pdf_files)}] Processing {Path(pdf_path).name}")
            try:
                result = self.index_document(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to index {pdf_path}: {e}")
                results.append(IndexingResult(
                    filepath=pdf_path,
                    filename=Path(pdf_path).name,
                    success=False,
                    error=str(e),
                ))

        # Summary
        successful = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunk_count for r in results)
        logger.info(f"Indexing complete: {successful}/{len(results)} files, {total_chunks} chunks")

        return results

    def get_all_chunks(self) -> list[dict]:
        """Get all indexed chunks from the chunk store."""
        return list(self._chunks_store.values())

    def get_chunk_count(self) -> int:
        """Get total number of stored chunks."""
        return len(self._chunks_store)

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        stats = self.registry.get_stats()
        stats["stored_chunks"] = len(self._chunks_store)
        return stats


# Quick test
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    indexer = DocumentIndexer()

    # Index a single file
    pdf_dir = "data/harel_pdfs/pdfs/"
    pdfs = list(Path(pdf_dir).glob("*.pdf"))

    if pdfs:
        result = indexer.index_document(str(pdfs[0]))
        print(f"\nResult: {result.filename}")
        print(f"  Success: {result.success}")
        print(f"  Pages: {result.page_count}")
        print(f"  Chunks: {result.chunk_count}")
        print(f"  Domain: {result.domain}")
        print(f"  Topics: {result.topics[:5]}")
        print(f"  Time: {result.processing_time_ms:.0f}ms")

        if result.chunks:
            print(f"\n  Sample chunk:")
            chunk = result.chunks[0]
            print(f"    ID: {chunk.id}")
            print(f"    Page: {chunk.page_num}")
            print(f"    Topics: {chunk.topics[:3]}")
            print(f"    Text: {chunk.text[:200]}...")

