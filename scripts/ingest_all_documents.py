#!/usr/bin/env python3
"""
Ingest all documents (PDFs + ASPX pages) into the vector store.

Features:
- Processes both PDFs and ASPX pages
- Preserves document structure (headers, tables, lists)
- Carries context summary from previous chunk to next chunk
- Builds/extends Table of Contents (TOC)
- Rebuilds vector DB from scratch

Usage:
    python scripts/ingest_all_documents.py
    python scripts/ingest_all_documents.py --source pdf
    python scripts/ingest_all_documents.py --source aspx
    python scripts/ingest_all_documents.py --force  # Re-index all
"""

import argparse
import json
import logging
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_indexer import DocumentIndexer
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.aspx_processor import ASPXProcessor
from src.ingestion.toc_builder import ToCBuilder
from src.retrieval import EmbeddingService, VectorStore, BM25Index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_pdfs(indexer: DocumentIndexer, pdf_dir: str = "data/harel_pdfs/pdfs") -> int:
    """Ingest all PDF documents."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        logger.warning(f"PDF directory not found: {pdf_dir}")
        return 0
    
    processor = PDFProcessor()
    count = 0
    
    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        try:
            doc = processor.process(str(pdf_file))
            result = indexer.index_processed_document(doc)
            if result.success and result.chunk_count > 0:
                count += 1
                logger.info(f"  ✓ {pdf_file.name}: {result.chunk_count} chunks")
        except Exception as e:
            logger.error(f"  ✗ {pdf_file.name}: {e}")
    
    return count


def ingest_aspx(indexer: DocumentIndexer, aspx_dir: str = "data/raw/aspx") -> int:
    """Ingest all ASPX pages."""
    processor = ASPXProcessor()
    
    if not Path(aspx_dir).exists():
        logger.warning(f"ASPX directory not found: {aspx_dir}")
        return 0
    
    documents = processor.process_all()
    count = 0
    
    for doc in documents:
        try:
            result = indexer.index_processed_document(doc)
            if result.success and result.chunk_count > 0:
                count += 1
        except Exception as e:
            logger.error(f"  ✗ {doc.filename}: {e}")
    
    logger.info(f"Ingested {count} ASPX pages")
    return count


def embed_and_store(chunks_path: str = "data/processed/chunks.json"):
    """Embed chunks and store in vector store."""
    import json
    
    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    
    if not chunks:
        logger.warning("No chunks to embed")
        return
    
    logger.info(f"Embedding {len(chunks)} chunks...")
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    bm25_index = BM25Index()
    
    # Prepare data
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{
        "source_file": c.get("source_file", ""),
        "source_filename": c.get("source_filename", ""),
        "page_num": c.get("page_num", 1),
        "domain": c.get("domain", ""),
        "topics": ",".join(c.get("topics", [])),
        "section_path": " > ".join(c.get("section_path", [])),
        "content_type": c.get("content_type", "text"),
    } for c in chunks]
    
    # Generate embeddings
    embeddings = embedding_service.embed_batch(texts)
    
    # Store in vector store
    vector_store.add_documents(
        ids=ids,
        embeddings=embeddings,
        texts=texts,
        metadatas=metadatas,
    )
    
    # Build BM25 index
    bm25_index.add_documents(ids=ids, texts=texts, metadatas=metadatas)
    bm25_index.save()
    
    logger.info(f"Stored {len(chunks)} chunks in vector store and BM25 index")


def clear_vector_db():
    """Clear the ChromaDB vector store completely."""
    chroma_path = Path("data/vector_store")
    if chroma_path.exists():
        logger.info(f"Clearing vector store at {chroma_path}")
        shutil.rmtree(chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)

    # Also clear BM25 index
    bm25_path = Path("data/processed/bm25_index.pkl")
    if bm25_path.exists():
        bm25_path.unlink()
        logger.info("Cleared BM25 index")


def build_toc(chunks_path: str = "data/processed/chunks.json"):
    """Build Table of Contents from indexed chunks."""
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = list(data.get("chunks", {}).values())

    if not chunks:
        logger.warning("No chunks to build TOC from")
        return

    logger.info(f"Building TOC from {len(chunks)} chunks...")
    toc_builder = ToCBuilder()
    toc_builder.build_from_chunks(chunks)
    logger.info("TOC built and saved")


def main():
    parser = argparse.ArgumentParser(description="Ingest all documents")
    parser.add_argument("--source", choices=["all", "pdf", "aspx"], default="all",
                       help="Source type to ingest")
    parser.add_argument("--force", action="store_true",
                       help="Force re-indexing of all documents (clears vector DB)")
    parser.add_argument("--skip-embed", action="store_true",
                       help="Skip embedding step")
    parser.add_argument("--skip-toc", action="store_true",
                       help="Skip TOC building step")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Document Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source}")
    logger.info(f"Force: {args.force}")

    # Initialize indexer
    indexer = DocumentIndexer()

    if args.force:
        logger.info("\n--- Force mode: clearing all existing data ---")
        # Clear registry and chunks
        indexer.registry.clear()
        indexer._chunks_store.clear()
        indexer._save_chunks()
        # Clear vector DB
        clear_vector_db()
        # Clear TOC
        toc_path = Path("data/processed/toc.json")
        if toc_path.exists():
            toc_path.unlink()
            logger.info("Cleared TOC")

    total_docs = 0

    # Ingest PDFs
    if args.source in ["all", "pdf"]:
        logger.info("\n--- Ingesting PDFs ---")
        total_docs += ingest_pdfs(indexer)

    # Ingest ASPX
    if args.source in ["all", "aspx"]:
        logger.info("\n--- Ingesting ASPX Pages ---")
        total_docs += ingest_aspx(indexer)

    logger.info(f"\nTotal documents ingested: {total_docs}")
    logger.info(f"Total chunks: {len(indexer._chunks_store)}")

    # Build TOC
    if not args.skip_toc and len(indexer._chunks_store) > 0:
        logger.info("\n--- Building Table of Contents ---")
        build_toc()

    # Embed and store
    if not args.skip_embed and len(indexer._chunks_store) > 0:
        logger.info("\n--- Embedding and Storing in Vector DB ---")
        embed_and_store()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Documents processed: {total_docs}")
    logger.info(f"Total chunks: {len(indexer._chunks_store)}")

    # Show domain breakdown
    domain_counts = {}
    for chunk in indexer._chunks_store.values():
        domain = chunk.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    logger.info("\nChunks by domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {domain}: {count}")


if __name__ == "__main__":
    main()

