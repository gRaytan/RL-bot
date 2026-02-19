"""
Hybrid Retriever - Combines semantic search with BM25 keyword search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both retrieval methods.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from .embedding_service import EmbeddingService, EmbeddingConfig
from .vector_store import VectorStore, VectorStoreConfig, SearchResult
from .bm25_index import BM25Index, BM25Config, BM25Result

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid retriever."""
    semantic_weight: float = 0.6  # Weight for semantic search
    bm25_weight: float = 0.4  # Weight for BM25 search
    rrf_k: int = 60  # RRF constant (higher = more weight to lower ranks)
    semantic_top_k: int = 20  # Candidates from semantic search
    bm25_top_k: int = 20  # Candidates from BM25 search
    final_top_k: int = 10  # Final results after fusion


@dataclass
class RetrievalResult:
    """Single retrieval result with combined score."""
    id: str
    text: str
    metadata: dict
    combined_score: float
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results:
    RRF_score = sum(1 / (k + rank_i)) for each retrieval method

    Usage:
        retriever = HybridRetriever(embedding_service, vector_store, bm25_index)
        results = retriever.search("מה הכיסוי לגניבת רכב?", top_k=5)
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        config: Optional[HybridConfig] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.config = config or HybridConfig()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        domain_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Perform hybrid search combining semantic and BM25.

        Args:
            query: Search query
            top_k: Number of results (default: config.final_top_k)
            domain_filter: Filter by domain
            topic_filter: Filter by topic

        Returns:
            List of RetrievalResult sorted by combined score
        """
        top_k = top_k or self.config.final_top_k

        # Get semantic search results
        semantic_results = self._semantic_search(query, domain_filter, topic_filter)

        # Get BM25 results
        bm25_results = self._bm25_search(query, domain_filter)

        # Fuse results using RRF
        fused = self._rrf_fusion(semantic_results, bm25_results)

        # Return top-k
        return fused[:top_k]

    def _semantic_search(
        self,
        query: str,
        domain_filter: Optional[str],
        topic_filter: Optional[str],
    ) -> list[SearchResult]:
        """Perform semantic search."""
        query_embedding = self.embedding_service.embed(query)

        filter_dict = {}
        if domain_filter:
            filter_dict["domain"] = domain_filter
        if topic_filter:
            filter_dict["topics"] = topic_filter

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.config.semantic_top_k,
            filter=filter_dict if filter_dict else None,
        )

    def _bm25_search(
        self,
        query: str,
        domain_filter: Optional[str],
    ) -> list[BM25Result]:
        """Perform BM25 keyword search."""
        if not self.bm25_index.is_built:
            logger.warning("BM25 index not built - skipping keyword search")
            return []

        return self.bm25_index.search(
            query=query,
            top_k=self.config.bm25_top_k,
            domain_filter=domain_filter,
        )

    def _rrf_fusion(
        self,
        semantic_results: list[SearchResult],
        bm25_results: list[BM25Result],
    ) -> list[RetrievalResult]:
        """
        Fuse results using Reciprocal Rank Fusion.

        RRF score = w1 * (1 / (k + rank_semantic)) + w2 * (1 / (k + rank_bm25))
        """
        k = self.config.rrf_k
        w_semantic = self.config.semantic_weight
        w_bm25 = self.config.bm25_weight

        # Build lookup maps
        doc_scores = {}  # id -> {semantic_score, bm25_score, ranks, text, metadata}

        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            doc_scores[result.id] = {
                "text": result.text,
                "metadata": result.metadata,
                "semantic_score": result.score,
                "semantic_rank": rank,
                "bm25_score": None,
                "bm25_rank": None,
            }

        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            if result.id in doc_scores:
                doc_scores[result.id]["bm25_score"] = result.score
                doc_scores[result.id]["bm25_rank"] = rank
            else:
                doc_scores[result.id] = {
                    "text": result.text,
                    "metadata": result.metadata,
                    "semantic_score": None,
                    "semantic_rank": None,
                    "bm25_score": result.score,
                    "bm25_rank": rank,
                }

        # Calculate RRF scores
        results = []
        for doc_id, info in doc_scores.items():
            rrf_score = 0.0

            if info["semantic_rank"] is not None:
                rrf_score += w_semantic * (1.0 / (k + info["semantic_rank"]))

            if info["bm25_rank"] is not None:
                rrf_score += w_bm25 * (1.0 / (k + info["bm25_rank"]))

            results.append(RetrievalResult(
                id=doc_id,
                text=info["text"],
                metadata=info["metadata"],
                combined_score=rrf_score,
                semantic_score=info["semantic_score"],
                bm25_score=info["bm25_score"],
                semantic_rank=info["semantic_rank"],
                bm25_rank=info["bm25_rank"],
            ))

        # Sort by combined score (descending)
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results

    def search_semantic_only(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Search using only semantic search."""
        results = self._semantic_search(query, domain_filter, None)
        return [
            RetrievalResult(
                id=r.id,
                text=r.text,
                metadata=r.metadata,
                combined_score=r.score,
                semantic_score=r.score,
            )
            for r in results[:top_k]
        ]

    def search_bm25_only(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Search using only BM25."""
        results = self._bm25_search(query, domain_filter)
        return [
            RetrievalResult(
                id=r.id,
                text=r.text,
                metadata=r.metadata,
                combined_score=r.score,
                bm25_score=r.score,
            )
            for r in results[:top_k]
        ]
