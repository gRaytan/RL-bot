"""
Table of Contents Builder - Creates navigable ToC from indexed chunks.

The ToC enables:
1. Structured navigation by topic/domain
2. Jump to specific sections
3. See chunk counts per topic
4. Filter search by topic
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .topic_taxonomy import TopicTaxonomy

logger = logging.getLogger(__name__)


@dataclass
class ToCEntry:
    """A single entry in the Table of Contents."""
    id: str
    name_he: str
    name_en: str
    parent_id: Optional[str] = None
    chunk_count: int = 0
    chunk_ids: list[str] = field(default_factory=list)
    children: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name_he": self.name_he,
            "name_en": self.name_en,
            "parent_id": self.parent_id,
            "chunk_count": self.chunk_count,
            "chunk_ids": self.chunk_ids,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


class ToCBuilder:
    """
    Builds and maintains the Table of Contents from indexed chunks.

    The ToC maps topics to chunks, enabling:
    - Navigation: "Show me all topics in car insurance"
    - Filtering: "Search only in car/comprehensive/coverage"
    - Statistics: "How many chunks cover health insurance?"
    """

    def __init__(
        self,
        taxonomy: Optional[TopicTaxonomy] = None,
        toc_path: str = "data/processed/toc.json",
    ):
        self.taxonomy = taxonomy or TopicTaxonomy()
        self.toc_path = Path(toc_path)
        self._toc: dict[str, ToCEntry] = {}
        self._topic_chunks: dict[str, list[str]] = defaultdict(list)

        # Load existing ToC if available
        self._load()

    def _load(self):
        """Load existing ToC from file."""
        if self.toc_path.exists():
            try:
                with open(self.toc_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._topic_chunks = defaultdict(list, data.get("topic_chunks", {}))
                    logger.info(f"Loaded ToC with {len(self._topic_chunks)} topics")
            except Exception as e:
                logger.warning(f"Failed to load ToC: {e}")

    def save(self):
        """Save ToC to file."""
        self.toc_path.parent.mkdir(parents=True, exist_ok=True)

        toc_data = {
            "toc": self.generate(),
            "topic_chunks": dict(self._topic_chunks),
            "stats": self.get_stats(),
        }

        with open(self.toc_path, "w", encoding="utf-8") as f:
            json.dump(toc_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved ToC to {self.toc_path}")

    def add_chunk(self, chunk_id: str, topics: list[str]):
        """Add a chunk to the ToC under its topics."""
        for topic_id in topics:
            if chunk_id not in self._topic_chunks[topic_id]:
                self._topic_chunks[topic_id].append(chunk_id)

    def build_from_chunks(self, chunks: list[dict]):
        """Build ToC from a list of indexed chunks."""
        logger.info(f"Building ToC from {len(chunks)} chunks")

        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            topics = chunk.get("topics", [])
            self.add_chunk(chunk_id, topics)

        self.save()
        logger.info(f"ToC built with {len(self._topic_chunks)} topics")

    def generate(self) -> dict:
        """Generate the full ToC structure."""
        toc = {}

        for root in self.taxonomy.get_root_topics():
            entry = self._build_entry(root.id)
            toc[root.id] = entry.to_dict()

        return toc

    def _build_entry(self, topic_id: str) -> ToCEntry:
        """Build a ToC entry for a topic and its children."""
        topic = self.taxonomy.get_topic(topic_id)
        if not topic:
            return ToCEntry(id=topic_id, name_he="", name_en="")

        # Get chunks for this topic
        chunk_ids = self._topic_chunks.get(topic_id, [])

        # Build children
        children = {}
        for child in self.taxonomy.get_children(topic_id):
            children[child.id] = self._build_entry(child.id)

        # Count includes children
        total_chunks = len(chunk_ids)
        for child_entry in children.values():
            total_chunks += child_entry.chunk_count

        return ToCEntry(
            id=topic_id,
            name_he=topic.name_he,
            name_en=topic.name_en,
            parent_id=topic.parent_id,
            chunk_count=total_chunks,
            chunk_ids=chunk_ids,
            children=children,
        )

    def get_stats(self) -> dict:
        """Get ToC statistics."""
        total_chunks = sum(len(chunks) for chunks in self._topic_chunks.values())
        topics_with_chunks = sum(1 for chunks in self._topic_chunks.values() if chunks)

        return {
            "total_topics": len(self.taxonomy._topics),
            "topics_with_chunks": topics_with_chunks,
            "total_chunk_assignments": total_chunks,
            "domains": {
                root.id: {
                    "name_he": root.name_he,
                    "chunk_count": len(self.get_chunks_for_topic(root.id)),
                }
                for root in self.taxonomy.get_root_topics()
            },
        }

    def print_toc(self, max_depth: int = 2):
        """Print a human-readable ToC."""
        print("\n" + "=" * 60)
        print("TABLE OF CONTENTS")
        print("=" * 60)

        for root in self.taxonomy.get_root_topics():
            self._print_entry(root.id, depth=0, max_depth=max_depth)

        print("=" * 60)

    def _print_entry(self, topic_id: str, depth: int, max_depth: int):
        """Print a single ToC entry."""
        if depth > max_depth:
            return

        topic = self.taxonomy.get_topic(topic_id)
        if not topic:
            return

        chunk_count = len(self.get_chunks_for_topic(topic_id))
        indent = "  " * depth

        print(f"{indent}• {topic.name_he} ({topic.name_en}) [{chunk_count} chunks]")

        for child in self.taxonomy.get_children(topic_id):
            self._print_entry(child.id, depth + 1, max_depth)


    def get_chunks_for_topic(
        self,
        topic_id: str,
        include_children: bool = True
    ) -> list[str]:
        """Get all chunk IDs for a topic."""
        chunks = list(self._topic_chunks.get(topic_id, []))

        if include_children:
            for child in self.taxonomy.get_children(topic_id):
                chunks.extend(self.get_chunks_for_topic(child.id, include_children=True))

        return list(set(chunks))  # Deduplicate

