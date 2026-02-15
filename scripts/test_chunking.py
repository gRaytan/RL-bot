#!/usr/bin/env python3
"""
Test script to demonstrate dynamic page-size based chunking.
Shows how chunk size varies based on page content size.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.chunker import (
    AdaptiveChunker,
    get_chunk_config_for_page,
    get_dynamic_chunk_config,
)


def demo_dynamic_chunking():
    """Demonstrate how chunk size scales with page size."""
    print("=" * 70)
    print("DYNAMIC CHUNK SIZE CALCULATION")
    print("Formula: chunk_size = 256 + (page_chars / 8), clamped to [256, 1536]")
    print("=" * 70)
    print()
    
    # Test various page sizes
    test_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 15000]
    
    print(f"{'Page Chars':>12} | {'Chunk Size':>12} | {'Overlap':>10} | {'Est. Chunks':>12}")
    print("-" * 55)
    
    for page_chars in test_sizes:
        config = get_dynamic_chunk_config(page_chars)
        est_chunks = max(1, page_chars // (config.chunk_size - config.chunk_overlap))
        print(f"{page_chars:>12,} | {config.chunk_size:>12} | {config.chunk_overlap:>10} | {est_chunks:>12}")
    
    print()


def demo_mode_comparison():
    """Compare different chunking modes."""
    print("=" * 70)
    print("CHUNKING MODE COMPARISON")
    print("=" * 70)
    print()
    
    # Sample pages of different sizes
    pages = {
        "Small (cover page)": "א" * 300,
        "Medium (summary)": "א" * 1500,
        "Large (content)": "א" * 4000,
        "Very Large (table)": "א" * 10000,
    }
    
    modes = ["dynamic", "threshold", "fixed"]
    
    for page_name, page_text in pages.items():
        print(f"\n{page_name} ({len(page_text):,} chars):")
        print("-" * 50)
        for mode in modes:
            try:
                config = get_chunk_config_for_page(page_text, mode=mode)
                print(f"  {mode:>10}: chunk_size={config.chunk_size}, overlap={config.chunk_overlap}")
            except Exception as e:
                print(f"  {mode:>10}: Error - {e}")
    
    print()


def demo_document_chunking():
    """Demonstrate chunking a multi-page document."""
    print("=" * 70)
    print("MULTI-PAGE DOCUMENT CHUNKING")
    print("=" * 70)
    print()
    
    # Simulate a document with varying page sizes
    pages = [
        "כותרת המסמך\nביטוח רכב - פוליסה מקיפה",  # Small cover page
        "תקציר\n" + "זוהי פוליסת ביטוח מקיפה לרכב. " * 50,  # Medium summary
        "פרק 1: כיסויים\n" + "הפוליסה מכסה נזקי גוף ורכוש. " * 150,  # Large content
        "טבלת תעריפים\n" + "| גיל | תעריף |\n" * 200,  # Very large table
    ]
    
    chunker = AdaptiveChunker(doc_type="pdf")
    
    # Get stats first
    stats = chunker.get_chunking_stats(pages)
    
    print("Page Statistics:")
    print(f"{'Page':>6} | {'Chars':>8} | {'Chunk Size':>12} | {'Est. Chunks':>12}")
    print("-" * 50)
    
    for ps in stats["page_stats"]:
        print(f"{ps['page']:>6} | {ps['char_count']:>8,} | {ps['chunk_size']:>12} | {ps['estimated_chunks']:>12}")
    
    print(f"\nChunk Size Distribution: {stats['chunk_size_distribution']}")
    
    # Actually chunk the document
    chunks = chunker.chunk_document(pages, doc_metadata={"doc_type": "insurance_policy"})
    
    print(f"\nTotal Chunks Created: {len(chunks)}")
    print("\nSample Chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk {i+1}:")
        print(f"    Page: {chunk['metadata']['page']}")
        print(f"    Chunk Size Used: {chunk['metadata']['chunk_size_used']}")
        print(f"    Text Preview: {chunk['text'][:50]}...")


if __name__ == "__main__":
    demo_dynamic_chunking()
    demo_mode_comparison()
    demo_document_chunking()

