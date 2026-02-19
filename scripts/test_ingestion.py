#!/usr/bin/env python3
"""
Test script for document ingestion components.
Tests DocumentRegistry and TopicTaxonomy.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.document_registry import DocumentRegistry
from ingestion.topic_taxonomy import TopicTaxonomy


def test_document_registry():
    """Test DocumentRegistry functionality."""
    print("=" * 60)
    print("Testing DocumentRegistry")
    print("=" * 60)
    
    # Initialize registry (use temp path for testing)
    registry = DocumentRegistry("data/processed/test_registry.json")
    print(f"✓ Created registry: {registry}")
    
    # Get pending files
    pdf_dir = "data/harel_pdfs/pdfs"
    pending = registry.get_pending_files(pdf_dir)
    print(f"✓ Found {len(pending)} pending files to index")
    
    if pending:
        # Show first 5
        print("  First 5 pending files:")
        for f in pending[:5]:
            print(f"    - {Path(f).name}")
    
    # Test file hash
    if pending:
        test_file = pending[0]
        file_hash = registry.compute_file_hash(test_file)
        print(f"✓ Computed hash for {Path(test_file).name}: {file_hash[:16]}...")
    
    # Test stats
    stats = registry.get_stats()
    print(f"✓ Registry stats: {stats}")
    
    print()


def test_topic_taxonomy():
    """Test TopicTaxonomy functionality."""
    print("=" * 60)
    print("Testing TopicTaxonomy")
    print("=" * 60)
    
    # Initialize taxonomy
    taxonomy = TopicTaxonomy()
    print(f"✓ Created taxonomy: {taxonomy}")
    
    # Get root topics
    roots = taxonomy.get_root_topics()
    print(f"✓ Root topics ({len(roots)}):")
    for root in roots:
        children = taxonomy.get_children(root.id)
        print(f"    - {root.name_he} ({root.id}) - {len(children)} subtopics")
    
    # Test classification
    test_texts = [
        "ביטוח רכב מקיף כולל כיסוי לגניבה",
        "הגשת תביעה על נזק לדירה",
        "ביטול נסיעה לחו\"ל",
        "מספר הטלפון של מוקד תביעות",
        "ביטוח עובדים זרים לעסק",
    ]
    
    print(f"\n✓ Text classification:")
    for text in test_texts:
        topics = taxonomy.classify_text(text)
        top_topics = topics[:3] if topics else ["(no match)"]
        print(f"    \"{text[:40]}...\"")
        print(f"      → {', '.join(top_topics)}")
    
    # Test domain inference from filename
    test_files = [
        "פוליסת-ביטוח-רכב-מקיף.pdf",
        "טופס-תביעה-דירה.pdf",
        "ביטוח-נסיעות-לחול.pdf",
        "ביטוח-עובדים-זרים.pdf",
    ]
    
    print(f"\n✓ Domain inference from filename:")
    for filename in test_files:
        domain = taxonomy.get_domain_from_filepath(filename)
        print(f"    {filename} → {domain}")
    
    # Generate ToC
    toc = taxonomy.generate_toc()
    print(f"\n✓ Generated ToC with {len(toc)} root sections")
    
    # Save taxonomy
    taxonomy.save("data/processed/taxonomy.json")
    print(f"✓ Saved taxonomy to data/processed/taxonomy.json")
    
    print()


def test_integration():
    """Test integration between components."""
    print("=" * 60)
    print("Testing Integration")
    print("=" * 60)
    
    registry = DocumentRegistry("data/processed/test_registry.json")
    taxonomy = TopicTaxonomy()
    
    # Simulate indexing a document
    pdf_dir = "data/harel_pdfs/pdfs"
    pending = registry.get_pending_files(pdf_dir)
    
    if pending:
        test_file = pending[0]
        filename = Path(test_file).name
        
        # Infer domain
        domain = taxonomy.get_domain_from_filepath(test_file)
        print(f"✓ File: {filename}")
        print(f"  Domain: {domain}")
        
        # Simulate chunk IDs
        fake_chunk_ids = [f"chunk_{i:03d}" for i in range(5)]
        
        # Register as indexed
        record = registry.register_indexed(
            filepath=test_file,
            chunk_ids=fake_chunk_ids,
            page_count=10,
            domain=domain,
            topics=[domain],
        )
        print(f"  Registered: {record.status}, {record.chunk_count} chunks")
        
        # Verify it's now indexed
        is_indexed = registry.is_indexed(test_file)
        print(f"  Is indexed: {is_indexed}")
        
        # Check stats
        stats = registry.get_stats()
        print(f"  Registry stats: {stats}")
    
    print()


if __name__ == "__main__":
    test_document_registry()
    test_topic_taxonomy()
    test_integration()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

