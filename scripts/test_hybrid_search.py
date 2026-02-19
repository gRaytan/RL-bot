#!/usr/bin/env python3
"""
Test Hybrid Search - Compare semantic, BM25, and hybrid retrieval.

Usage:
    python scripts/test_hybrid_search.py "query"
    python scripts/test_hybrid_search.py "query" --domain car
    python scripts/test_hybrid_search.py "query" --compare
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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def init_retriever():
    """Initialize hybrid retriever with all components."""
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
    
    # Hybrid retriever
    hybrid_config = HybridConfig(
        semantic_weight=0.6,
        bm25_weight=0.4,
        semantic_top_k=20,
        bm25_top_k=20,
        final_top_k=10,
    )
    retriever = HybridRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_index=bm25_index,
        config=hybrid_config,
    )
    
    return retriever


def print_results(results, title: str):
    """Print search results."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(results, 1):
        print(f"[{i}] Combined: {r.combined_score:.6f}", end="")
        if r.semantic_score is not None:
            print(f" | Semantic: {r.semantic_score:.4f} (rank {r.semantic_rank})", end="")
        if r.bm25_score is not None:
            print(f" | BM25: {r.bm25_score:.4f} (rank {r.bm25_rank})", end="")
        print()
        print(f"    Domain: {r.metadata.get('domain', 'N/A')}")
        print(f"    Source: {r.metadata.get('source_filename', 'N/A')}")
        print(f"    Text: {r.text[:150]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test hybrid search")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--domain", type=str, help="Filter by domain")
    parser.add_argument("--compare", action="store_true", help="Compare all methods")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    retriever = init_retriever()
    
    print(f"\nQuery: {args.query}")
    if args.domain:
        print(f"Domain filter: {args.domain}")
    
    if args.compare:
        # Compare all three methods
        semantic = retriever.search_semantic_only(args.query, args.top_k, args.domain)
        print_results(semantic, "SEMANTIC SEARCH (Vector Only)")
        
        bm25 = retriever.search_bm25_only(args.query, args.top_k, args.domain)
        print_results(bm25, "BM25 SEARCH (Keyword Only)")
        
        hybrid = retriever.search(args.query, args.top_k, args.domain)
        print_results(hybrid, "HYBRID SEARCH (RRF Fusion)")
    else:
        # Just hybrid search
        results = retriever.search(args.query, args.top_k, args.domain)
        print_results(results, "HYBRID SEARCH RESULTS")


if __name__ == "__main__":
    main()

