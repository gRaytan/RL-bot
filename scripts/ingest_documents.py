#!/usr/bin/env python3
"""
Document Ingestion CLI - Process PDFs and create chunks for RAG.

Usage:
    python scripts/ingest_documents.py                    # Index all pending PDFs
    python scripts/ingest_documents.py --all              # Re-index all PDFs
    python scripts/ingest_documents.py --file FILE.pdf    # Index single file
    python scripts/ingest_documents.py --stats            # Show indexing stats
    python scripts/ingest_documents.py --export           # Export chunks to JSON
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DocumentIndexer, IndexingResult


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def index_all(indexer: DocumentIndexer, pdf_dir: str, skip_indexed: bool = True):
    """Index all PDFs in directory."""
    print(f"\n{'='*60}")
    print(f"Indexing PDFs from: {pdf_dir}")
    print(f"Skip already indexed: {skip_indexed}")
    print(f"{'='*60}\n")
    
    results = indexer.index_directory(pdf_dir, skip_indexed=skip_indexed)
    
    # Summary
    successful = [r for r in results if r.success and r.chunk_count > 0]
    skipped = [r for r in results if r.success and r.chunk_count == 0]
    failed = [r for r in results if not r.success]
    
    total_chunks = sum(r.chunk_count for r in successful)
    total_pages = sum(r.page_count for r in successful)
    total_time = sum(r.processing_time_ms for r in successful)
    
    print(f"\n{'='*60}")
    print("INDEXING SUMMARY")
    print(f"{'='*60}")
    print(f"  Successful: {len(successful)} files")
    print(f"  Skipped:    {len(skipped)} files (already indexed)")
    print(f"  Failed:     {len(failed)} files")
    print(f"  Total pages: {total_pages}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total time: {total_time/1000:.1f}s")
    
    if failed:
        print(f"\nFailed files:")
        for r in failed:
            print(f"  - {r.filename}: {r.error}")
    
    return results


def index_single(indexer: DocumentIndexer, filepath: str):
    """Index a single PDF file."""
    print(f"\nIndexing: {filepath}")
    
    result = indexer.index_document(filepath)
    
    print(f"\nResult: {result.filename}")
    print(f"  Success: {result.success}")
    print(f"  Pages: {result.page_count}")
    print(f"  Chunks: {result.chunk_count}")
    print(f"  Domain: {result.domain}")
    print(f"  Topics: {result.topics[:5]}")
    print(f"  Time: {result.processing_time_ms:.0f}ms")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    return result


def show_stats(indexer: DocumentIndexer):
    """Show indexing statistics."""
    stats = indexer.get_stats()
    
    print(f"\n{'='*60}")
    print("INDEXING STATISTICS")
    print(f"{'='*60}")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total chunks (registry): {stats['total_chunks']}")
    print(f"  Stored chunks: {stats.get('stored_chunks', 0)}")
    print(f"  Indexed: {stats['indexed']}")
    print(f"  Pending: {stats['pending']}")
    print(f"  Failed: {stats['failed']}")


def export_chunks(indexer: DocumentIndexer, output_path: str):
    """Export all stored chunks to JSON file."""
    # Get all chunks from the indexer's chunk store
    all_chunks = indexer.get_all_chunks()

    # Save to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "exported_at": datetime.now().isoformat(),
            "total_chunks": len(all_chunks),
            "chunks": all_chunks,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nExported {len(all_chunks)} chunks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Document Ingestion CLI")
    parser.add_argument("--dir", default="data/harel_pdfs/pdfs/", help="PDF directory")
    parser.add_argument("--file", help="Single PDF file to index")
    parser.add_argument("--all", action="store_true", help="Re-index all files")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--export", help="Export chunks to JSON file")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for scanned PDFs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Initialize indexer
    indexer = DocumentIndexer(use_ocr=args.ocr)
    
    if args.stats:
        show_stats(indexer)
        return
    
    if args.file:
        result = index_single(indexer, args.file)
        results = [result]
    else:
        skip_indexed = not args.all
        results = index_all(indexer, args.dir, skip_indexed=skip_indexed)
    
    if args.export:
        export_chunks(indexer, args.export)
    
    show_stats(indexer)


if __name__ == "__main__":
    main()

