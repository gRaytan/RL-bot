#!/usr/bin/env python3
"""
Test RAG Pipeline - End-to-end question answering with retrieval.

Usage:
    python scripts/test_rag_pipeline.py "שאלה"
    python scripts/test_rag_pipeline.py "שאלה" --domain car
    python scripts/test_rag_pipeline.py "שאלה" --no-rerank
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

from src.rag import RAGPipeline, RAGConfig
from src.rag.answer_generator import GeneratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test RAG pipeline")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--domain", type=str, help="Filter by domain")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranking")
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="LLM model")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = RAGConfig(
        retrieval_top_k=30,
        rerank_top_k=10,
        final_context_k=5,
        use_reranker=not args.no_rerank,
        generator_config=GeneratorConfig(
            provider="nebius",
            model=args.model,
        ),
    )
    
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(config=config)
    
    print(f"\n{'='*70}")
    print(f"Question: {args.question}")
    if args.domain:
        print(f"Domain filter: {args.domain}")
    print(f"Reranking: {'enabled' if not args.no_rerank else 'disabled'}")
    print(f"{'='*70}\n")
    
    # Query
    response = pipeline.query(args.question, domain_filter=args.domain)
    
    # Print answer
    print("ANSWER:")
    print("-" * 70)
    print(response.answer)
    print("-" * 70)
    
    # Print citations
    print(f"\nCITATIONS ({len(response.citations)}):")
    for i, cite in enumerate(response.citations, 1):
        print(f"  [{i}] {cite.source_file}, page {cite.page_num}")
    
    # Print timing
    print(f"\nTIMING:")
    print(f"  Retrieval: {response.retrieval_time_ms:.0f}ms")
    print(f"  Reranking: {response.rerank_time_ms:.0f}ms")
    print(f"  Generation: {response.generation_time_ms:.0f}ms")
    print(f"  Total: {response.total_time_ms:.0f}ms")
    
    # Print stats
    print(f"\nSTATS:")
    print(f"  Retrieved: {response.retrieved_count} chunks")
    print(f"  Reranked: {response.reranked_count} chunks")
    print(f"  Context: {response.context_count} chunks")
    print(f"  Confidence: {response.confidence}")


if __name__ == "__main__":
    main()

