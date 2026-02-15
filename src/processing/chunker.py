"""
Adaptive chunking based on page size and document type.
Implements intelligent chunking that adjusts chunk size based on content density.
"""

from dataclasses import dataclass
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class ChunkConfig:
    """Configuration for a specific chunk."""
    chunk_size: int
    chunk_overlap: int
    
    def __repr__(self):
        return f"ChunkConfig(size={self.chunk_size}, overlap={self.chunk_overlap})"


@dataclass
class PageSizeThreshold:
    """Threshold configuration for page-size based chunking."""
    max_page_chars: int
    chunk_size: int
    chunk_overlap: int


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_page_size_thresholds() -> List[PageSizeThreshold]:
    """Get page size thresholds from config."""
    config = load_config()
    page_config = config.get("document_processing", {}).get("page_size_chunking", {})
    
    if not page_config.get("enabled", False):
        return []
    
    thresholds = []
    for t in page_config.get("thresholds", []):
        thresholds.append(PageSizeThreshold(
            max_page_chars=t["max_page_chars"],
            chunk_size=t["chunk_size"],
            chunk_overlap=t["chunk_overlap"],
        ))
    
    # Sort by max_page_chars ascending
    thresholds.sort(key=lambda x: x.max_page_chars)
    return thresholds


def get_dynamic_chunk_config(page_chars: int, min_chunk: int = 256, max_chunk: int = 1536) -> ChunkConfig:
    """
    Calculate chunk size dynamically based on page character count.

    Formula: chunk_size = min_chunk + (page_chars / scale_factor)
    - Clamped between min_chunk and max_chunk
    - Overlap is 10% of chunk size

    Args:
        page_chars: Number of characters in the page
        min_chunk: Minimum chunk size (default: 256)
        max_chunk: Maximum chunk size (default: 1536)

    Returns:
        ChunkConfig with dynamically calculated chunk_size and overlap
    """
    # Scale factor: how many page chars per additional chunk token
    # ~4 chars per token, we want chunk to be ~1/4 to 1/2 of page
    scale_factor = 8

    # Dynamic calculation
    dynamic_size = min_chunk + (page_chars // scale_factor)

    # Clamp to bounds
    chunk_size = max(min_chunk, min(max_chunk, dynamic_size))

    # Round to nearest 64 for cleaner values
    chunk_size = (chunk_size // 64) * 64
    chunk_size = max(min_chunk, chunk_size)  # Ensure minimum

    # Overlap is 10% of chunk size
    chunk_overlap = max(25, chunk_size // 10)

    return ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def get_chunk_config_for_page(
    page_text: str,
    doc_type: str = "pdf",
    mode: str = "dynamic"  # "dynamic", "threshold", or "fixed"
) -> ChunkConfig:
    """
    Determine the optimal chunk configuration based on page size.

    Args:
        page_text: The text content of the page
        doc_type: Document type (pdf, aspx, table, list)
        mode: Chunking mode:
            - "dynamic": Chunk size scales with page size (recommended)
            - "threshold": Use predefined thresholds from config
            - "fixed": Use fixed chunk size per document type

    Returns:
        ChunkConfig with appropriate chunk_size and chunk_overlap
    """
    config = load_config()
    doc_config = config.get("document_processing", {})
    page_chars = len(page_text)

    # Mode 1: Dynamic calculation (recommended)
    if mode == "dynamic":
        page_config = doc_config.get("page_size_chunking", {})
        min_chunk = page_config.get("min_chunk_size", 256)
        max_chunk = page_config.get("max_chunk_size", 1536)
        return get_dynamic_chunk_config(page_chars, min_chunk, max_chunk)

    # Mode 2: Threshold-based (from config)
    if mode == "threshold":
        page_config = doc_config.get("page_size_chunking", {})
        if page_config.get("enabled", False):
            thresholds = get_page_size_thresholds()
            for threshold in thresholds:
                if page_chars <= threshold.max_page_chars:
                    return ChunkConfig(
                        chunk_size=threshold.chunk_size,
                        chunk_overlap=threshold.chunk_overlap,
                    )

    # Mode 3: Fixed per document type (fallback)
    chunk_sizes = doc_config.get("chunk_sizes", {})
    chunk_overlaps = doc_config.get("chunk_overlaps", {})
    
    return ChunkConfig(
        chunk_size=chunk_sizes.get(doc_type, doc_config.get("chunk_size", 1024)),
        chunk_overlap=chunk_overlaps.get(doc_type, doc_config.get("chunk_overlap", 100)),
    )


class AdaptiveChunker:
    """
    Adaptive text chunker that adjusts chunk size based on page/content size.
    """
    
    def __init__(self, doc_type: str = "pdf"):
        """
        Initialize the adaptive chunker.
        
        Args:
            doc_type: Document type for fallback chunking
        """
        self.doc_type = doc_type
        self.config = load_config().get("document_processing", {})
    
    def chunk_page(self, page_text: str, page_num: int = 1) -> List[dict]:
        """
        Chunk a single page with adaptive chunk size.
        
        Args:
            page_text: Text content of the page
            page_num: Page number for metadata
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunk_config = get_chunk_config_for_page(page_text, self.doc_type)
        chunks = []
        
        text = page_text.strip()
        if not text:
            return chunks
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + chunk_config.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence/paragraph boundary
            if end < len(text):
                # Look for natural break points
                for sep in ["\n\n", "\n", ". ", "। ", "؟ ", "? "]:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > chunk_config.chunk_size * 0.5:
                        chunk_text = chunk_text[:last_sep + len(sep)]
                        end = start + len(chunk_text)
                        break
            
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "chunk_size_used": chunk_config.chunk_size,
                    "char_count": len(chunk_text),
                },
            })
            
            chunk_idx += 1
            start = end - chunk_config.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def chunk_document(
        self,
        pages: List[str],
        doc_metadata: Optional[dict] = None
    ) -> List[dict]:
        """
        Chunk an entire document with multiple pages.
        Each page gets its own adaptive chunk size based on content density.

        Args:
            pages: List of page texts (index = page number - 1)
            doc_metadata: Optional metadata to include with each chunk

        Returns:
            List of all chunks with metadata
        """
        all_chunks = []
        doc_metadata = doc_metadata or {}

        for page_num, page_text in enumerate(pages, start=1):
            page_chunks = self.chunk_page(page_text, page_num)

            for chunk in page_chunks:
                chunk["metadata"].update(doc_metadata)
                chunk["metadata"]["total_pages"] = len(pages)
                all_chunks.append(chunk)

        # Add global chunk index
        for i, chunk in enumerate(all_chunks):
            chunk["metadata"]["global_chunk_index"] = i
            chunk["metadata"]["total_chunks"] = len(all_chunks)

        return all_chunks

    def get_chunking_stats(self, pages: List[str]) -> dict:
        """
        Get statistics about how a document would be chunked.

        Args:
            pages: List of page texts

        Returns:
            Dictionary with chunking statistics
        """
        stats = {
            "total_pages": len(pages),
            "page_stats": [],
            "chunk_size_distribution": {},
        }

        for page_num, page_text in enumerate(pages, start=1):
            config = get_chunk_config_for_page(page_text, self.doc_type)
            page_stat = {
                "page": page_num,
                "char_count": len(page_text),
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "estimated_chunks": max(1, len(page_text) // (config.chunk_size - config.chunk_overlap)),
            }
            stats["page_stats"].append(page_stat)

            # Track distribution
            size_key = str(config.chunk_size)
            stats["chunk_size_distribution"][size_key] = \
                stats["chunk_size_distribution"].get(size_key, 0) + 1

        return stats

