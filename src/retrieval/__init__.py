"""
Retrieval module for RAG pipeline.

Components:
- EmbeddingService: Generate embeddings using Nebius or OpenAI
- VectorStore: Store and search embeddings (ChromaDB)
- BM25Index: Sparse keyword search
- HybridRetriever: Combine semantic + keyword search with RRF
"""

from .embedding_service import EmbeddingService, EmbeddingConfig
from .vector_store import VectorStore, VectorStoreConfig, SearchResult
from .bm25_index import BM25Index, BM25Config, BM25Result
from .hybrid_retriever import HybridRetriever, HybridConfig, RetrievalResult
from .reranker import Reranker, RerankerConfig, RankedResult

__all__ = [
    "EmbeddingService",
    "EmbeddingConfig",
    "VectorStore",
    "VectorStoreConfig",
    "SearchResult",
    "BM25Index",
    "BM25Config",
    "BM25Result",
    "HybridRetriever",
    "HybridConfig",
    "RetrievalResult",
    "Reranker",
    "RerankerConfig",
    "RankedResult",
]

