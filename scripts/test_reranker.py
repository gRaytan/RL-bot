#!/usr/bin/env python3
"""
Test Cross-Encoder Reranker - Compare retrieval with and without reranking.

Usage:
    python scripts/test_reranker.py "query"
    python scripts/test_reranker.py "query" --domain car
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.retrieval import (
    EmbeddingService, EmbeddingConfig,
    VectorStore, VectorStoreConfig,
    BM25Index, BM25Config,
    HybridRetriever, HybridConfig,
    Reranker, RerankerConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def init_components():
    """Initialize all retrieval components."""
    # Embedding service
    embedding_config = EmbeddingConfig(
        provider="nebius",
        model="BAAI/bge-multilingual-gemma2",
        dimension=3584,
    )
    embedding_service = EmbeddingService(config=embedding_config)
    
    # Vector store
    vector_config = VectorStoreConfig(
        collection_name="harel_insurance_kb",
        persist_directory="data/vectordb",
        embedding_dimension=3584,
    )
    vector_store = VectorStore(config=vector_config)
    
    # BM25 index
    bm25_config = BM25Config(index_path="data/processed/bm25_index.pkl")
    bm25_index = BM25Index(config=bm25_config)
    bm25_index.load()
    
    # Hybrid retriever - get more candidates for reranking
    hybrid_config = HybridConfig(
        semantic_weight=0.6,
        bm25_weight=0.4,
        semantic_top_k=30,
        bm25_top_k=30,
        final_top_k=20,  # Get 20 candidates for reranking
    )
    retriever = HybridRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_index=bm25_index,
        config=hybrid_config,
    )
    
    # Reranker
    reranker = Reranker()
    
    return retriever, reranker


def main():
    parser = argparse.ArgumentParser(description="Test cross-encoder reranking")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--domain", type=str, help="Filter by domain")
    parser.add_argument("--top-k", type=int, default=5, help="Final results")
    parser.add_argument("--candidates", type=int, default=20, help="Candidates for reranking")
    
    args = parser.parse_args()
    
    retriever, reranker = init_components()
    
    print(f"\nQuery: {args.query}")
    if args.domain:
        print(f"Domain filter: {args.domain}")
    
    # Get candidates from hybrid search
    candidates = retriever.search(args.query, args.candidates, args.domain)
    
    print(f"\n{'='*70}")
    print(f"BEFORE RERANKING (Hybrid Search Top {args.top_k})")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(candidates[:args.top_k], 1):
        print(f"[{i}] Score: {r.combined_score:.6f}")
        print(f"    Domain: {r.metadata.get('domain', 'N/A')}")
        print(f"    Source: {r.metadata.get('source_filename', 'N/A')}")
        print(f"    Text: {r.text[:150]}...")
        print()
    
    # Rerank
    reranked = reranker.rerank(args.query, candidates, args.top_k)
    
    print(f"\n{'='*70}")
    print(f"AFTER RERANKING (Cross-Encoder Top {args.top_k})")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(reranked, 1):
        print(f"[{i}] Rerank: {r.rerank_score:.4f} (was rank {r.original_rank})")
        print(f"    Domain: {r.metadata.get('domain', 'N/A')}")
        print(f"    Source: {r.metadata.get('source_filename', 'N/A')}")
        print(f"    Text: {r.text[:150]}...")
        print()


if __name__ == "__main__":
    main()

