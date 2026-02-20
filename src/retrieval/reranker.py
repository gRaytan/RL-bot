"""
Cross-Encoder Reranker - Re-rank retrieved documents using a cross-encoder model.

Uses sentence-transformers CrossEncoder for more accurate relevance scoring.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from .hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""
    # Multilingual cross-encoder trained on MS MARCO
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Maximum sequence length (query + document)
    max_length: int = 512
    # Device for inference (None = auto-detect)
    device: Optional[str] = None
    # Batch size for reranking
    batch_size: int = 32


@dataclass 
class RankedResult:
    """Result after cross-encoder reranking."""
    id: str
    text: str
    metadata: dict
    rerank_score: float  # Cross-encoder score
    original_score: float  # Original retrieval score
    original_rank: int


class Reranker:
    """
    Cross-encoder reranker for improving retrieval quality.
    
    Cross-encoders jointly encode query and document, providing
    more accurate relevance scores than bi-encoders (embeddings).
    
    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, retrieval_results, top_k=5)
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        
        logger.info(f"Loading cross-encoder: {self.config.model_name}")
        self._model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=self.config.device,
        )
        logger.info("Cross-encoder loaded successfully")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RankedResult]:
        """
        Rerank retrieval results using cross-encoder.
        
        Args:
            query: The search query
            results: List of retrieval results to rerank
            top_k: Number of results to return (default: all)
            
        Returns:
            List of RankedResult sorted by rerank_score
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, r.text) for r in results]
        
        # Get cross-encoder scores
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        
        # Create ranked results
        ranked = []
        for i, (result, score) in enumerate(zip(results, scores)):
            ranked.append(RankedResult(
                id=result.id,
                text=result.text,
                metadata=result.metadata,
                rerank_score=float(score),
                original_score=result.combined_score,
                original_rank=i + 1,
            ))
        
        # Sort by rerank score (descending)
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Return top-k
        if top_k:
            return ranked[:top_k]
        return ranked

    def rerank_with_context(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None,
        include_metadata: bool = True,
    ) -> list[RankedResult]:
        """
        Rerank with additional context from metadata.
        
        Includes source filename and domain in the text for better matching.
        """
        if not results:
            return []
        
        # Prepare enriched query-document pairs
        pairs = []
        for r in results:
            if include_metadata:
                # Add metadata context
                source = r.metadata.get("source_filename", "")
                domain = r.metadata.get("domain", "")
                enriched_text = f"[{domain}] {source}\n{r.text}"
            else:
                enriched_text = r.text
            pairs.append((query, enriched_text))
        
        # Get cross-encoder scores
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        
        # Create ranked results
        ranked = []
        for i, (result, score) in enumerate(zip(results, scores)):
            ranked.append(RankedResult(
                id=result.id,
                text=result.text,
                metadata=result.metadata,
                rerank_score=float(score),
                original_score=result.combined_score,
                original_rank=i + 1,
            ))
        
        # Sort by rerank score (descending)
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            return ranked[:top_k]
        return ranked

