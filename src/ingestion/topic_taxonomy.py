"""
Topic Taxonomy for insurance knowledge base.

Defines hierarchical topic structure for:
- Document classification
- ToC navigation
- Filtered retrieval
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Topic:
    """A topic in the taxonomy hierarchy."""
    id: str                          # e.g., "car/comprehensive/coverage"
    name_en: str                     # English name
    name_he: str                     # Hebrew name
    keywords_he: list[str] = field(default_factory=list)  # Hebrew keywords for classification
    keywords_en: list[str] = field(default_factory=list)  # English keywords
    parent_id: Optional[str] = None  # Parent topic ID
    children: list[str] = field(default_factory=list)     # Child topic IDs
    description: Optional[str] = None
    chunk_ids: list[str] = field(default_factory=list)    # Chunks tagged with this topic

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Topic":
        return cls(**data)


# Default taxonomy for Harel Insurance
DEFAULT_TAXONOMY = {
    # ===== CAR INSURANCE =====
    "car": Topic(
        id="car",
        name_en="Car Insurance",
        name_he="ביטוח רכב",
        keywords_he=["רכב", "מכונית", "נהיגה", "תאונת דרכים", "רישיון"],
        keywords_en=["car", "vehicle", "auto", "driving"],
    ),
    "car/mandatory": Topic(
        id="car/mandatory",
        name_en="Mandatory Insurance",
        name_he="ביטוח חובה",
        keywords_he=["חובה", "חוק", "פיצויים", "נפגעי תאונות"],
        parent_id="car",
    ),
    "car/comprehensive": Topic(
        id="car/comprehensive",
        name_en="Comprehensive Insurance",
        name_he="ביטוח מקיף",
        keywords_he=["מקיף", "גניבה", "נזק עצמי", "אובדן גמור"],
        parent_id="car",
    ),
    "car/comprehensive/coverage": Topic(
        id="car/comprehensive/coverage",
        name_en="Coverage",
        name_he="כיסויים",
        keywords_he=["כיסוי", "מכוסה", "כולל", "פוליסה מכסה"],
        parent_id="car/comprehensive",
    ),
    "car/comprehensive/exclusions": Topic(
        id="car/comprehensive/exclusions",
        name_en="Exclusions",
        name_he="חריגים",
        keywords_he=["חריג", "לא מכוסה", "אינו מכסה", "למעט", "פרט ל"],
        parent_id="car/comprehensive",
    ),
    "car/comprehensive/deductible": Topic(
        id="car/comprehensive/deductible",
        name_en="Deductible",
        name_he="השתתפות עצמית",
        keywords_he=["השתתפות עצמית", "דמי השתתפות"],
        parent_id="car/comprehensive",
    ),
    "car/third_party": Topic(
        id="car/third_party",
        name_en="Third Party",
        name_he="צד ג׳",
        keywords_he=["צד ג", "צד שלישי", "נזק לאחרים"],
        parent_id="car",
    ),
    "car/claims": Topic(
        id="car/claims",
        name_en="Claims",
        name_he="תביעות",
        keywords_he=["תביעה", "הגשת תביעה", "פתיחת תיק", "דיווח נזק"],
        parent_id="car",
    ),
    
    # ===== APARTMENT INSURANCE =====
    "apartment": Topic(
        id="apartment",
        name_en="Apartment Insurance",
        name_he="ביטוח דירה",
        keywords_he=["דירה", "בית", "מגורים", "נכס"],
        keywords_en=["apartment", "home", "house", "property"],
    ),
    "apartment/structure": Topic(
        id="apartment/structure",
        name_en="Structure",
        name_he="מבנה",
        keywords_he=["מבנה", "קירות", "גג", "יסודות"],
        parent_id="apartment",
    ),
    "apartment/contents": Topic(
        id="apartment/contents",
        name_en="Contents",
        name_he="תכולה",
        keywords_he=["תכולה", "רהיטים", "מכשירי חשמל", "ציוד"],
        parent_id="apartment",
    ),
    "apartment/coverage": Topic(
        id="apartment/coverage",
        name_en="Coverage",
        name_he="כיסויים",
        keywords_he=["כיסוי", "מכוסה", "כולל"],
        parent_id="apartment",
    ),
    "apartment/exclusions": Topic(
        id="apartment/exclusions",
        name_en="Exclusions",
        name_he="חריגים",
        keywords_he=["חריג", "לא מכוסה", "אינו מכסה"],
        parent_id="apartment",
    ),
    "apartment/claims": Topic(
        id="apartment/claims",
        name_en="Claims",
        name_he="תביעות",
        keywords_he=["תביעה", "הגשת תביעה", "דיווח נזק"],
        parent_id="apartment",
    ),
    
    # ===== HEALTH INSURANCE =====
    "health": Topic(
        id="health",
        name_en="Health Insurance",
        name_he="ביטוח בריאות",
        keywords_he=["בריאות", "רפואי", "רפואה", "חולה", "מחלה"],
        keywords_en=["health", "medical", "healthcare"],
    ),
    "health/surgery": Topic(
        id="health/surgery",
        name_en="Surgery",
        name_he="ניתוחים",
        keywords_he=["ניתוח", "ניתוחים", "אשפוז", "בית חולים"],
        parent_id="health",
    ),
    "health/medications": Topic(
        id="health/medications",
        name_en="Medications",
        name_he="תרופות",
        keywords_he=["תרופה", "תרופות", "מרשם"],
        parent_id="health",
    ),

    # ===== TRAVEL INSURANCE =====
    "travel": Topic(
        id="travel",
        name_en="Travel Insurance",
        name_he="ביטוח נסיעות",
        keywords_he=["נסיעות", "חו\"ל", "טיסה", "נסיעה לחו\"ל"],
        keywords_en=["travel", "abroad", "flight", "trip"],
    ),
    "travel/medical": Topic(
        id="travel/medical",
        name_en="Medical Coverage",
        name_he="כיסוי רפואי",
        keywords_he=["רפואי", "אשפוז בחו\"ל", "טיפול רפואי"],
        parent_id="travel",
    ),
    "travel/cancellation": Topic(
        id="travel/cancellation",
        name_en="Trip Cancellation",
        name_he="ביטול נסיעה",
        keywords_he=["ביטול", "ביטול נסיעה", "ביטול טיסה"],
        parent_id="travel",
    ),
    "travel/luggage": Topic(
        id="travel/luggage",
        name_en="Luggage",
        name_he="מטען",
        keywords_he=["מטען", "מזוודה", "אובדן מטען", "כבודה"],
        parent_id="travel",
    ),

    # ===== BUSINESS INSURANCE =====
    "business": Topic(
        id="business",
        name_en="Business Insurance",
        name_he="ביטוח עסקים",
        keywords_he=["עסק", "עסקים", "חברה", "מושב", "עובדים"],
        keywords_en=["business", "company", "commercial"],
    ),
    "business/liability": Topic(
        id="business/liability",
        name_en="Professional Liability",
        name_he="אחריות מקצועית",
        keywords_he=["אחריות מקצועית", "רשלנות", "נזק ללקוח"],
        parent_id="business",
    ),
    "business/property": Topic(
        id="business/property",
        name_en="Business Property",
        name_he="רכוש עסקי",
        keywords_he=["רכוש עסקי", "ציוד", "מלאי", "מבנה עסק"],
        parent_id="business",
    ),
    "business/employees": Topic(
        id="business/employees",
        name_en="Employee Insurance",
        name_he="ביטוח עובדים",
        keywords_he=["עובדים", "עובדים זרים", "תאונות עבודה"],
        parent_id="business",
    ),

    # ===== LIFE INSURANCE =====
    "life": Topic(
        id="life",
        name_en="Life Insurance",
        name_he="ביטוח חיים",
        keywords_he=["חיים", "ביטוח חיים", "מוות", "נכות"],
        keywords_en=["life", "death", "disability"],
    ),

    # ===== DENTAL INSURANCE =====
    "dental": Topic(
        id="dental",
        name_en="Dental Insurance",
        name_he="ביטוח שיניים",
        keywords_he=["שיניים", "רופא שיניים", "טיפול שיניים"],
        keywords_en=["dental", "teeth", "dentist"],
    ),

    # ===== MORTGAGE INSURANCE =====
    "mortgage": Topic(
        id="mortgage",
        name_en="Mortgage Insurance",
        name_he="ביטוח משכנתא",
        keywords_he=["משכנתא", "הלוואה", "בנק"],
        keywords_en=["mortgage", "loan"],
    ),

    # ===== GENERAL =====
    "general": Topic(
        id="general",
        name_en="General",
        name_he="כללי",
        keywords_he=["כללי", "מידע", "שירות"],
    ),
    "general/claims": Topic(
        id="general/claims",
        name_en="How to File Claims",
        name_he="הגשת תביעות",
        keywords_he=["תביעה", "הגשת תביעה", "איך להגיש", "טופס תביעה"],
        parent_id="general",
    ),
    "general/cancellation": Topic(
        id="general/cancellation",
        name_en="Policy Cancellation",
        name_he="ביטול פוליסה",
        keywords_he=["ביטול", "ביטול פוליסה", "סיום ביטוח"],
        parent_id="general",
    ),
    "general/contact": Topic(
        id="general/contact",
        name_en="Contact Information",
        name_he="פרטי התקשרות",
        keywords_he=["טלפון", "מוקד", "פקס", "מייל", "כתובת", "יצירת קשר"],
        parent_id="general",
    ),
}


class TopicTaxonomy:
    """
    Manage the topic taxonomy for document classification and navigation.

    Features:
    - Hierarchical topic structure
    - Keyword-based classification
    - ToC index generation
    - Extensible (add new topics at runtime)
    """

    def __init__(self, taxonomy_path: Optional[str] = None):
        """
        Initialize taxonomy.

        Args:
            taxonomy_path: Path to custom taxonomy JSON. If None, uses default.
        """
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else None
        self._topics: dict[str, Topic] = {}
        self._load_taxonomy()

    def _load_taxonomy(self):
        """Load taxonomy from file or use default."""
        if self.taxonomy_path and self.taxonomy_path.exists():
            with open(self.taxonomy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for topic_id, topic_data in data.get("topics", {}).items():
                    self._topics[topic_id] = Topic.from_dict(topic_data)
        else:
            # Use default taxonomy
            self._topics = {k: v for k, v in DEFAULT_TAXONOMY.items()}

        # Build parent-child relationships
        self._build_hierarchy()

    def _build_hierarchy(self):
        """Build parent-child relationships."""
        for topic_id, topic in self._topics.items():
            if topic.parent_id and topic.parent_id in self._topics:
                parent = self._topics[topic.parent_id]
                if topic_id not in parent.children:
                    parent.children.append(topic_id)

    def save(self, path: Optional[str] = None):
        """Save taxonomy to JSON file."""
        save_path = Path(path) if path else self.taxonomy_path
        if not save_path:
            save_path = Path("data/processed/taxonomy.json")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "topics": {k: v.to_dict() for k, v in self._topics.items()},
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        """Get topic by ID."""
        return self._topics.get(topic_id)

    def get_root_topics(self) -> list[Topic]:
        """Get all root-level topics (no parent)."""
        return [t for t in self._topics.values() if t.parent_id is None]

    def get_children(self, topic_id: str) -> list[Topic]:
        """Get child topics of a topic."""
        topic = self._topics.get(topic_id)
        if not topic:
            return []
        return [self._topics[cid] for cid in topic.children if cid in self._topics]

    def get_all_descendants(self, topic_id: str) -> list[Topic]:
        """Get all descendant topics (recursive)."""
        descendants = []
        children = self.get_children(topic_id)
        for child in children:
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child.id))
        return descendants

    def classify_text(self, text: str) -> list[str]:
        """
        Classify text into topics based on keyword matching.

        Returns:
            List of topic IDs that match, ordered by specificity (most specific first).
        """
        text_lower = text.lower()
        matches = []

        for topic_id, topic in self._topics.items():
            score = 0
            # Check Hebrew keywords
            for kw in topic.keywords_he:
                if kw in text:
                    score += 1
            # Check English keywords
            for kw in topic.keywords_en:
                if kw.lower() in text_lower:
                    score += 1

            if score > 0:
                # More specific topics (deeper in hierarchy) get priority
                depth = topic_id.count("/")
                matches.append((topic_id, score, depth))

        # Sort by depth (descending) then score (descending)
        matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return [m[0] for m in matches]

    def add_topic(self, topic: Topic) -> bool:
        """Add a new topic to the taxonomy."""
        if topic.id in self._topics:
            return False

        self._topics[topic.id] = topic

        # Update parent's children list
        if topic.parent_id and topic.parent_id in self._topics:
            parent = self._topics[topic.parent_id]
            if topic.id not in parent.children:
                parent.children.append(topic.id)

        return True

    def add_chunk_to_topic(self, topic_id: str, chunk_id: str):
        """Associate a chunk with a topic."""
        if topic_id in self._topics:
            if chunk_id not in self._topics[topic_id].chunk_ids:
                self._topics[topic_id].chunk_ids.append(chunk_id)

    def get_chunks_for_topic(self, topic_id: str, include_descendants: bool = True) -> list[str]:
        """Get all chunk IDs for a topic (optionally including descendants)."""
        chunk_ids = []

        topic = self._topics.get(topic_id)
        if topic:
            chunk_ids.extend(topic.chunk_ids)

            if include_descendants:
                for desc in self.get_all_descendants(topic_id):
                    chunk_ids.extend(desc.chunk_ids)

        return list(set(chunk_ids))  # Dedupe

    def get_domain_from_filepath(self, filepath: str) -> Optional[str]:
        """Infer domain from filename."""
        filename = Path(filepath).name.lower()

        domain_keywords = {
            "car": ["רכב", "חובה", "מקיף", "צד-ג"],
            "apartment": ["דירה", "מבנה", "תכולה", "דירות"],
            "health": ["בריאות", "רפואי", "רפואה"],
            "travel": ["נסיעות", "חול", "טיסה"],
            "business": ["עסק", "עסקים", "מושב", "עובדים-זרים"],
            "life": ["חיים"],
            "dental": ["שיניים"],
            "mortgage": ["משכנתא"],
        }

        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if kw in filename:
                    return domain

        return "general"

    def generate_toc(self) -> dict:
        """Generate Table of Contents structure for navigation."""
        toc = {}

        for root in self.get_root_topics():
            toc[root.id] = {
                "name_he": root.name_he,
                "name_en": root.name_en,
                "chunk_count": len(root.chunk_ids),
                "subtopics": self._generate_subtoc(root.id),
            }

        return toc

    def _generate_subtoc(self, topic_id: str) -> dict:
        """Recursively generate subtopic structure."""
        subtoc = {}

        for child in self.get_children(topic_id):
            subtoc[child.id] = {
                "name_he": child.name_he,
                "name_en": child.name_en,
                "chunk_count": len(child.chunk_ids),
                "subtopics": self._generate_subtoc(child.id),
            }

        return subtoc

    def __repr__(self) -> str:
        roots = len(self.get_root_topics())
        total = len(self._topics)
        return f"TopicTaxonomy(roots={roots}, total_topics={total})"

