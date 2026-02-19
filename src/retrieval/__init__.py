"""
Retrieval module for RAG pipeline.

Components:
- EmbeddingService: Generate embeddings using OpenAI or local models
- VectorStore: Store and search embeddings (ChromaDB/Milvus)
- HybridRetriever: Combine semantic + keyword search
"""

from .embedding_service import EmbeddingService, EmbeddingConfig
from .vector_store import VectorStore, VectorStoreConfig, SearchResult

__all__ = [
    "EmbeddingService",
    "EmbeddingConfig",
    "VectorStore",
    "VectorStoreConfig",
    "SearchResult",
]

