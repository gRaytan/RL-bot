#!/usr/bin/env python3
"""
Build BM25 Index - Create sparse keyword index for hybrid search.

Usage:
    python scripts/build_bm25_index.py                    # Build index
    python scripts/build_bm25_index.py --test "query"     # Test search
    python scripts/build_bm25_index.py --stats            # Show statistics
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import BM25Index, BM25Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_chunks(chunks_path: str = "data/processed/chunks.json") -> list[dict]:
    """Load chunks from JSON file."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


def build_index(chunks: list[dict], config: BM25Config):
    """Build BM25 index from chunks."""
    index = BM25Index(config=config)
    index.build_from_chunks(chunks, save=True)
    return index


def test_search(query: str, index: BM25Index, top_k: int = 5, domain: str = None):
    """Test BM25 search."""
    logger.info(f"Searching for: {query}")
    
    results = index.search(query, top_k=top_k, domain_filter=domain)
    
    print(f"\n{'='*60}")
    print(f"BM25 Search Results for: {query}")
    if domain:
        print(f"Filter: domain={domain}")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] Score: {result.score:.4f}")
        print(f"    ID: {result.id}")
        print(f"    Domain: {result.metadata.get('domain', 'N/A')}")
        print(f"    Source: {result.metadata.get('source_filename', 'N/A')}")
        print(f"    Text: {result.text[:200]}...")
        print()


def show_stats(index: BM25Index):
    """Show index statistics."""
    stats = index.get_stats()
    
    print(f"\n{'='*60}")
    print("BM25 INDEX STATISTICS")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for hybrid search")
    parser.add_argument("--chunks", default="data/processed/chunks.json", help="Path to chunks JSON")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--test", type=str, help="Test search with query")
    parser.add_argument("--domain", type=str, help="Filter by domain (for --test)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results for search")
    
    args = parser.parse_args()
    
    config = BM25Config(
        index_path="data/processed/bm25_index.pkl",
        k1=1.5,
        b=0.75,
    )
    
    index = BM25Index(config=config)
    
    # Try to load existing index
    if args.stats or args.test:
        if not index.load():
            logger.error("BM25 index not found. Run without --stats/--test to build it.")
            return
    
    if args.stats:
        show_stats(index)
        return
    
    if args.test:
        test_search(args.test, index, args.top_k, args.domain)
        return
    
    # Build index
    chunks = load_chunks(args.chunks)
    index = build_index(chunks, config)
    show_stats(index)


if __name__ == "__main__":
    main()

