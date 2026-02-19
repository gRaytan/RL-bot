"""
Document ingestion module for RAG pipeline.

Components:
- DocumentRegistry: Track indexed documents for incremental updates
- TopicTaxonomy: Define and manage topic hierarchy
- PDFProcessor: Extract text from PDFs using Docling
- DocumentIndexer: Orchestrate the full ingestion pipeline
- ToCBuilder: Build navigable Table of Contents from indexed chunks
"""

from .document_registry import DocumentRegistry
from .topic_taxonomy import TopicTaxonomy, Topic
from .pdf_processor import PDFProcessor, ProcessedDocument, PageContent
from .document_indexer import DocumentIndexer, IndexedChunk, IndexingResult
from .toc_builder import ToCBuilder, ToCEntry

__all__ = [
    "DocumentRegistry",
    "TopicTaxonomy",
    "Topic",
    "PDFProcessor",
    "ProcessedDocument",
    "PageContent",
    "DocumentIndexer",
    "IndexedChunk",
    "IndexingResult",
    "ToCBuilder",
    "ToCEntry",
]

