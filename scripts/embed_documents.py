#!/usr/bin/env python3
"""
Embed Documents - Generate embeddings for all chunks and store in vector database.

Usage:
    python scripts/embed_documents.py                    # Embed all chunks
    python scripts/embed_documents.py --stats            # Show statistics
    python scripts/embed_documents.py --test "query"     # Test search
    python scripts/embed_documents.py --reset            # Reset vector store
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from src.retrieval import EmbeddingService, VectorStore, EmbeddingConfig, VectorStoreConfig

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


def embed_all_chunks(
    chunks: list[dict],
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    batch_size: int = 50,
):
    """Embed all chunks and store in vector database."""
    logger.info(f"Embedding {len(chunks)} chunks...")
    
    # Prepare data
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "domain": c.get("domain", "general"),
            "topics": ",".join(c.get("topics", [])),
            "source_file": c.get("source_file", ""),
            "source_filename": c.get("source_filename", ""),
            "page_num": c.get("page_num", 0),
            "chunk_index": c.get("chunk_index", 0),
        }
        for c in chunks
    ]
    
    # Generate embeddings in batches
    logger.info("Generating embeddings...")
    embeddings = embedding_service.embed_batch(texts, show_progress=True)
    
    # Store in vector database in batches (ChromaDB max batch size is 5461)
    STORE_BATCH_SIZE = 5000
    logger.info(f"Storing in vector database ({len(ids)} documents in batches of {STORE_BATCH_SIZE})...")

    for i in range(0, len(ids), STORE_BATCH_SIZE):
        batch_end = min(i + STORE_BATCH_SIZE, len(ids))
        batch_num = (i // STORE_BATCH_SIZE) + 1
        total_batches = (len(ids) + STORE_BATCH_SIZE - 1) // STORE_BATCH_SIZE

        vector_store.add_documents(
            ids=ids[i:batch_end],
            texts=texts[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        logger.info(f"Stored batch {batch_num}/{total_batches} ({batch_end}/{len(ids)} documents)")

    logger.info(f"Successfully embedded and stored {len(chunks)} chunks")


def test_search(
    query: str,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    top_k: int = 5,
    domain_filter: str = None,
):
    """Test search functionality."""
    logger.info(f"Searching for: {query}")
    
    # Generate query embedding
    query_embedding = embedding_service.embed(query)
    
    # Search
    filter_dict = {"domain": domain_filter} if domain_filter else None
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter=filter_dict,
    )
    
    print(f"\n{'='*60}")
    print(f"Search Results for: {query}")
    if domain_filter:
        print(f"Filter: domain={domain_filter}")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] Score: {result.score:.4f}")
        print(f"    ID: {result.id}")
        print(f"    Domain: {result.metadata.get('domain', 'N/A')}")
        print(f"    Source: {result.metadata.get('source_filename', 'N/A')}")
        print(f"    Text: {result.text[:200]}...")
        print()


def show_stats(vector_store: VectorStore):
    """Show vector store statistics."""
    stats = vector_store.get_stats()
    
    print(f"\n{'='*60}")
    print("VECTOR STORE STATISTICS")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Embed documents into vector store")
    parser.add_argument("--chunks", default="data/processed/chunks.json", help="Path to chunks JSON")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--test", type=str, help="Test search with query")
    parser.add_argument("--domain", type=str, help="Filter by domain (for --test)")
    parser.add_argument("--reset", action="store_true", help="Reset vector store before embedding")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results for search")
    
    args = parser.parse_args()
    
    # Initialize services - using Nebius bge-multilingual-gemma2 for Hebrew support
    embedding_config = EmbeddingConfig(
        provider="nebius",
        model="BAAI/bge-multilingual-gemma2",  # Good for Hebrew
        dimension=3584,  # bge-multilingual-gemma2 actual dimension
        batch_size=50,  # Conservative batch size for API
    )
    embedding_service = EmbeddingService(config=embedding_config)

    vector_config = VectorStoreConfig(
        collection_name="harel_insurance_kb",
        persist_directory="data/vectordb",
        embedding_dimension=3584,  # Match embedding model
    )
    vector_store = VectorStore(config=vector_config)

    if args.reset:
        logger.warning("Resetting vector store...")
        vector_store.delete_collection()
        vector_store = VectorStore(config=vector_config)

    if args.stats:
        show_stats(vector_store)
        return

    if args.test:
        if not embedding_service.is_available:
            logger.error("Embedding service not available - check LLM_API_KEY")
            return
        test_search(args.test, embedding_service, vector_store, args.top_k, args.domain)
        return

    # Embed all chunks
    if not embedding_service.is_available:
        logger.error("Embedding service not available - check LLM_API_KEY")
        return
    
    chunks = load_chunks(args.chunks)
    embed_all_chunks(chunks, embedding_service, vector_store)
    show_stats(vector_store)


if __name__ == "__main__":
    main()

