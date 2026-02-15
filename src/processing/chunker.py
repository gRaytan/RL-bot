"""
Adaptive chunking based on page size and document type.
Implements intelligent chunking that adjusts chunk size based on content density.

Features:
- Dynamic chunk sizing based on page content
- Contextual chunking: each chunk includes a summary of the previous chunk
  (instead of overlapping text - more token efficient)
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import yaml
from pathlib import Path


@dataclass
class ChunkConfig:
    """Configuration for a specific chunk."""
    chunk_size: int
    chunk_overlap: int = 0  # Default to 0 - we use summaries instead

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
    - No overlap - we use contextual summaries instead

    Args:
        page_chars: Number of characters in the page
        min_chunk: Minimum chunk size (default: 256)
        max_chunk: Maximum chunk size (default: 1536)

    Returns:
        ChunkConfig with dynamically calculated chunk_size (no overlap)
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

    # No overlap - we use contextual summaries instead
    return ChunkConfig(chunk_size=chunk_size, chunk_overlap=0)


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
    Uses contextual summaries instead of overlap for better token efficiency.
    """

    def __init__(
        self,
        doc_type: str = "pdf",
        summarizer: Optional[Callable[[str], str]] = None,
        summary_max_chars: int = 150,
    ):
        """
        Initialize the adaptive chunker.

        Args:
            doc_type: Document type for fallback chunking
            summarizer: Function to generate summaries (async or sync)
                       If None, uses simple extractive summary
            summary_max_chars: Maximum characters for context summary
        """
        self.doc_type = doc_type
        self.config = load_config().get("document_processing", {})
        self.summarizer = summarizer
        self.summary_max_chars = summary_max_chars

    def _extractive_summary(self, text: str) -> str:
        """
        Simple extractive summary - takes first sentence(s) up to max chars.
        Used when no LLM summarizer is provided.
        """
        if not text:
            return ""

        # Try to get first complete sentence(s)
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in ".!?。؟":
                sentences.append(current.strip())
                current = ""
                if len(" ".join(sentences)) >= self.summary_max_chars:
                    break

        summary = " ".join(sentences)
        if len(summary) > self.summary_max_chars:
            summary = summary[:self.summary_max_chars].rsplit(" ", 1)[0] + "..."

        return summary if summary else text[:self.summary_max_chars] + "..."

    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the text."""
        if self.summarizer:
            return self.summarizer(text)
        return self._extractive_summary(text)

    def chunk_page(
        self,
        page_text: str,
        page_num: int = 1,
        previous_context: str = "",
    ) -> List[dict]:
        """
        Chunk a single page with adaptive chunk size.
        Each chunk includes context from the previous chunk.

        Args:
            page_text: Text content of the page
            page_num: Page number for metadata
            previous_context: Summary of previous chunk for context

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
        current_context = previous_context

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

            # Build the full chunk with context
            chunk_with_context = ""
            if current_context:
                chunk_with_context = f"[הקשר קודם: {current_context}]\n\n{chunk_text.strip()}"
            else:
                chunk_with_context = chunk_text.strip()

            chunks.append({
                "text": chunk_with_context,
                "raw_text": chunk_text.strip(),  # Original text without context
                "previous_summary": current_context,
                "metadata": {
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "chunk_size_used": chunk_config.chunk_size,
                    "char_count": len(chunk_text),
                    "has_context": bool(current_context),
                },
            })

            # Generate summary of this chunk for the next one
            current_context = self._generate_summary(chunk_text.strip())

            chunk_idx += 1
            start = end  # No overlap - we use summaries instead
            if start >= len(text):
                break

        return chunks

    def chunk_document(
        self,
        pages: List[str],
        doc_metadata: Optional[dict] = None,
        carry_context_across_pages: bool = True,
    ) -> List[dict]:
        """
        Chunk an entire document with multiple pages.
        Each page gets its own adaptive chunk size based on content density.
        Context summaries are carried across chunks (and optionally across pages).

        Args:
            pages: List of page texts (index = page number - 1)
            doc_metadata: Optional metadata to include with each chunk
            carry_context_across_pages: If True, last chunk's summary carries to next page

        Returns:
            List of all chunks with metadata
        """
        all_chunks = []
        doc_metadata = doc_metadata or {}
        previous_context = ""  # Context from previous chunk

        for page_num, page_text in enumerate(pages, start=1):
            # Pass context from previous page/chunk
            page_chunks = self.chunk_page(
                page_text,
                page_num,
                previous_context=previous_context if carry_context_across_pages else ""
            )

            for chunk in page_chunks:
                chunk["metadata"].update(doc_metadata)
                chunk["metadata"]["total_pages"] = len(pages)
                all_chunks.append(chunk)

            # Carry last chunk's context to next page
            if page_chunks and carry_context_across_pages:
                last_chunk = page_chunks[-1]
                previous_context = self._generate_summary(last_chunk.get("raw_text", ""))

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

