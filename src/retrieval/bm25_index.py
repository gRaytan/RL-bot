"""
BM25 Index - Sparse keyword search for Hebrew insurance documents.

Uses rank_bm25 for efficient keyword matching, complementing semantic search.
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BM25Config:
    """Configuration for BM25 index."""
    index_path: str = "data/processed/bm25_index.pkl"
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization


@dataclass
class BM25Result:
    """Single BM25 search result."""
    id: str
    score: float
    text: str
    metadata: dict


class BM25Index:
    """
    BM25 sparse keyword index for Hebrew text.

    Usage:
        index = BM25Index()
        index.build_from_chunks(chunks)
        results = index.search("גניבת רכב", top_k=10)
    """

    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self._index = None
        self._documents = []  # List of (id, text, metadata)
        self._tokenized_corpus = []

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize Hebrew text for BM25.
        Simple whitespace + punctuation tokenization.
        """
        # Remove punctuation except Hebrew letters, numbers, spaces
        text = re.sub(r'[^\u0590-\u05FF\w\s]', ' ', text)
        # Split on whitespace
        tokens = text.lower().split()
        # Filter short tokens
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def build_from_chunks(self, chunks: list[dict], save: bool = True):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dicts with 'id', 'text', and metadata
            save: Whether to save index to disk
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed. Run: pip install rank-bm25")
            raise

        logger.info(f"Building BM25 index from {len(chunks)} chunks...")

        self._documents = []
        self._tokenized_corpus = []

        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            text = chunk.get("text", "")
            metadata = {
                "domain": chunk.get("domain", "general"),
                "topics": chunk.get("topics", []),
                "source_file": chunk.get("source_file", ""),
                "source_filename": chunk.get("source_filename", ""),
                "page_num": chunk.get("page_num", 0),
            }

            self._documents.append((chunk_id, text, metadata))
            self._tokenized_corpus.append(self._tokenize(text))

        # Build BM25 index
        self._index = BM25Okapi(
            self._tokenized_corpus,
            k1=self.config.k1,
            b=self.config.b,
        )

        logger.info(f"BM25 index built: {len(self._documents)} documents")

        if save:
            self.save()

    def search(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
    ) -> list[BM25Result]:
        """
        Search using BM25.

        Args:
            query: Search query
            top_k: Number of results
            domain_filter: Optional domain to filter by

        Returns:
            List of BM25Result
        """
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build_from_chunks() first.")

        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self._index.get_scores(query_tokens)

        # Get top-k indices
        results = []
        for idx in scores.argsort()[::-1]:
            if len(results) >= top_k:
                break

            doc_id, text, metadata = self._documents[idx]
            score = float(scores[idx])

            # Skip zero scores
            if score <= 0:
                continue

            # Apply domain filter
            if domain_filter and metadata.get("domain") != domain_filter:
                continue

            results.append(BM25Result(
                id=doc_id,
                score=score,
                text=text,
                metadata=metadata,
            ))

        return results

    def load(self) -> bool:
        """
        Load index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(self.config.index_path)
        if not path.exists():
            logger.warning(f"BM25 index not found at {path}")
            return False

        try:
            from rank_bm25 import BM25Okapi

            with open(path, "rb") as f:
                data = pickle.load(f)

            self._documents = data["documents"]
            self._tokenized_corpus = data["tokenized_corpus"]

            # Rebuild BM25 from tokenized corpus
            self._index = BM25Okapi(
                self._tokenized_corpus,
                k1=data["config"]["k1"],
                b=data["config"]["b"],
            )

            logger.info(f"BM25 index loaded: {len(self._documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    @property
    def is_built(self) -> bool:
        """Check if index is built."""
        return self._index is not None

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "document_count": len(self._documents),
            "index_path": self.config.index_path,
            "is_built": self.is_built,
        }

    def save(self):
        """Save index to disk."""
        path = Path(self.config.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": self._documents,
            "tokenized_corpus": self._tokenized_corpus,
            "config": {"k1": self.config.k1, "b": self.config.b},
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to {path}")

