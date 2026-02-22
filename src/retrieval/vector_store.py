"""
Vector Store - Store and search embeddings with metadata filtering.

Uses ChromaDB for local development, with option to switch to Milvus for production.

Features:
- Semantic search with cosine similarity
- Metadata filtering (by domain, topic, source file)
- Persistent storage
- Batch operations
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    collection_name: str = "harel_insurance_kb"
    persist_directory: str = "data/vectordb"
    embedding_dimension: int = 3584  # bge-multilingual-gemma2 actual dimension


class VectorStore:
    """
    Vector database for semantic search with metadata filtering.

    Usage:
        store = VectorStore()
        store.add_documents(chunks, embeddings)
        results = store.search(query_embedding, filter={"domain": "car"}, top_k=5)
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._collection = None
        self._client = None
        self._init_store()

    def _init_store(self):
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = Path(self.config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )

            count = self._collection.count()
            logger.info(
                f"VectorStore initialized: collection={self.config.collection_name}, "
                f"documents={count}"
            )
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            raise

    @property
    def count(self) -> int:
        """Get number of documents in collection."""
        return self._collection.count() if self._collection else 0

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
    ):
        """
        Add documents to the vector store.

        Args:
            ids: Unique document IDs
            texts: Document texts (for display)
            embeddings: Document embeddings
            metadatas: Optional metadata for each document
        """
        if not self._collection:
            raise RuntimeError("VectorStore not initialized")

        if not ids:
            return

        # Prepare metadata - ChromaDB requires flat structure
        flat_metadatas = []
        for meta in (metadatas or [{}] * len(ids)):
            flat = {}
            for k, v in meta.items():
                if isinstance(v, list):
                    # Convert list to comma-separated string
                    flat[k] = ",".join(str(x) for x in v)
                elif isinstance(v, (str, int, float, bool)):
                    flat[k] = v
                else:
                    flat[k] = str(v)
            flat_metadatas.append(flat)

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=flat_metadatas,
        )

        logger.info(f"Added {len(ids)} documents to vector store")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter (e.g., {"domain": "car"})

        Returns:
            List of SearchResult objects
        """
        if not self._collection:
            raise RuntimeError("VectorStore not initialized")

        # Build ChromaDB where clause
        where = None
        if filter:
            where = self._build_where_clause(filter)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distance, convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Cosine distance to similarity

                search_results.append(SearchResult(
                    id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return search_results

    def _build_where_clause(self, filter: dict) -> dict:
        """Build ChromaDB where clause from filter dict."""
        conditions = []

        for key, value in filter.items():
            if isinstance(value, list):
                # OR condition for list values
                conditions.append({
                    "$or": [{key: {"$contains": v}} for v in value]
                })
            elif key == "topics":
                # Topics are stored as comma-separated string
                conditions.append({key: {"$contains": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None

    def search_by_topic(
        self,
        query_embedding: list[float],
        topic_id: str,
        top_k: int = 5,
        include_subtopics: bool = True,
    ) -> list[SearchResult]:
        """
        Search within a specific topic (from ToC).

        Args:
            query_embedding: Query embedding vector
            topic_id: Topic ID (e.g., "car/comprehensive/coverage")
            top_k: Number of results
            include_subtopics: Include child topics in search

        Returns:
            List of SearchResult objects
        """
        # Filter by topic - uses $contains since topics are comma-separated
        filter = {"topics": topic_id}
        return self.search(query_embedding, top_k=top_k, filter=filter)

    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """Get documents by their IDs."""
        if not self._collection or not ids:
            return []

        results = self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        search_results = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                search_results.append(SearchResult(
                    id=doc_id,
                    text=results["documents"][i] if results["documents"] else "",
                    score=1.0,  # Direct lookup, full score
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                ))

        return search_results

    def delete_collection(self):
        """Delete the entire collection (use with caution!)."""
        if self._client and self._collection:
            self._client.delete_collection(self.config.collection_name)
            self._collection = None
            logger.warning(f"Deleted collection: {self.config.collection_name}")

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "collection_name": self.config.collection_name,
            "document_count": self.count,
            "embedding_dimension": self.config.embedding_dimension,
            "persist_directory": self.config.persist_directory,
        }

