"""
RAG Pipeline - End-to-end Retrieval-Augmented Generation.

Combines hybrid retrieval, cross-encoder reranking, and answer generation.
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass, field

from ..retrieval import (
    EmbeddingService, EmbeddingConfig,
    VectorStore, VectorStoreConfig,
    BM25Index, BM25Config,
    HybridRetriever, HybridConfig,
    Reranker, RerankerConfig,
    RankedResult,
)
from .answer_generator import AnswerGenerator, GeneratorConfig, GeneratedAnswer, Citation

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retrieval settings
    retrieval_top_k: int = 50  # Candidates from hybrid search (increased)
    rerank_top_k: int = 15  # After reranking (increased)
    final_context_k: int = 8  # Chunks sent to LLM (increased for more context)

    # Component configs
    embedding_config: Optional[EmbeddingConfig] = None
    vector_config: Optional[VectorStoreConfig] = None
    bm25_config: Optional[BM25Config] = None
    hybrid_config: Optional[HybridConfig] = None
    reranker_config: Optional[RerankerConfig] = None
    generator_config: Optional[GeneratorConfig] = None

    # Feature flags
    use_reranker: bool = True


@dataclass
class RAGResponse:
    """Complete RAG response with all metadata."""
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: str = "medium"

    # Timing info
    retrieval_time_ms: float = 0
    rerank_time_ms: float = 0
    generation_time_ms: float = 0
    total_time_ms: float = 0

    # Debug info
    retrieved_count: int = 0
    reranked_count: int = 0
    context_count: int = 0


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Pipeline:
    1. Hybrid retrieval (semantic + BM25)
    2. Cross-encoder reranking
    3. Answer generation with citations

    Usage:
        pipeline = RAGPipeline()
        response = pipeline.query("מה הכיסוי לגניבת רכב?")
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._init_components()

    def _init_components(self):
        """Initialize all pipeline components."""
        # Embedding service
        emb_config = self.config.embedding_config or EmbeddingConfig(
            provider="nebius",
            model="BAAI/bge-multilingual-gemma2",
            dimension=3584,
        )
        self.embedding_service = EmbeddingService(config=emb_config)

        # Vector store
        vec_config = self.config.vector_config or VectorStoreConfig(
            collection_name="harel_insurance_kb",
            persist_directory="data/vectordb",
            embedding_dimension=3584,
        )
        self.vector_store = VectorStore(config=vec_config)

        # BM25 index
        bm25_config = self.config.bm25_config or BM25Config(
            index_path="data/processed/bm25_index.pkl"
        )
        self.bm25_index = BM25Index(config=bm25_config)
        self.bm25_index.load()

        # Hybrid retriever
        hybrid_config = self.config.hybrid_config or HybridConfig(
            semantic_weight=0.6,
            bm25_weight=0.4,
            semantic_top_k=self.config.retrieval_top_k,
            bm25_top_k=self.config.retrieval_top_k,
            final_top_k=self.config.retrieval_top_k,
        )
        self.retriever = HybridRetriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            config=hybrid_config,
        )

        # Reranker (optional)
        if self.config.use_reranker:
            reranker_config = self.config.reranker_config or RerankerConfig()
            self.reranker = Reranker(config=reranker_config)
        else:
            self.reranker = None

        # Answer generator
        gen_config = self.config.generator_config or GeneratorConfig()
        self.generator = AnswerGenerator(config=gen_config)

        logger.info("RAG Pipeline initialized")

    def query(
        self,
        question: str,
        domain_filter: Optional[str] = None,
        use_reranker: Optional[bool] = None,
    ) -> RAGResponse:
        """
        Process a question through the full RAG pipeline.

        Args:
            question: User's question
            domain_filter: Optional domain filter (car, health, etc.)
            use_reranker: Override config.use_reranker

        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        use_rerank = use_reranker if use_reranker is not None else self.config.use_reranker

        # Step 1: Hybrid retrieval
        t0 = time.time()
        candidates = self.retriever.search(
            query=question,
            top_k=self.config.retrieval_top_k,
            domain_filter=domain_filter,
        )
        retrieval_time = (time.time() - t0) * 1000

        # Step 2: Reranking (optional)
        rerank_time = 0
        if use_rerank and self.reranker and candidates:
            t0 = time.time()
            reranked = self.reranker.rerank(
                query=question,
                results=candidates,
                top_k=self.config.rerank_top_k,
            )
            rerank_time = (time.time() - t0) * 1000
            context_results = reranked
        else:
            context_results = candidates[:self.config.rerank_top_k]

        # Step 3: Answer generation
        t0 = time.time()
        generated = self.generator.generate(
            question=question,
            context_results=context_results,
            max_context_chunks=self.config.final_context_k,
        )
        generation_time = (time.time() - t0) * 1000

        total_time = (time.time() - start_time) * 1000

        return RAGResponse(
            question=question,
            answer=generated.answer,
            citations=generated.citations,
            confidence=generated.confidence,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            retrieved_count=len(candidates),
            reranked_count=len(context_results),
            context_count=generated.context_used,
        )

    def query_simple(self, question: str, domain_filter: Optional[str] = None) -> str:
        """Simple query that returns just the answer text."""
        response = self.query(question, domain_filter)
        return response.answer
