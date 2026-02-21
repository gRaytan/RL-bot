"""
ASPX Processor - Process scraped ASPX pages for ingestion.

Converts scraped JSON files into ProcessedDocument format for indexing.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .pdf_processor import ProcessedDocument, PageContent, StructuredItem

logger = logging.getLogger(__name__)


# Map ASPX domain names to our standard domain names
DOMAIN_MAPPING = {
    "car": "car",
    "life": "life",
    "travel": "travel",
    "health": "health",
    "dental": "health",  # Map dental to health
    "mortgage": "apartment",  # Map mortgage to apartment
    "business": "business",
    "apartment": "apartment",
    "personal-accident": "life",  # Map to life
    "long-term-care": "health",  # Map to health
    "foreign": "travel",  # Map foreign workers to travel
    "loss-of-working-ability": "life",  # Map to life
    "diseases-disabilities": "health",  # Map to health
}


@dataclass
class ASPXProcessorConfig:
    """Configuration for ASPX processor."""
    input_dir: str = "data/raw/aspx"
    min_content_length: int = 100  # Minimum content length to process


class ASPXProcessor:
    """
    Process scraped ASPX pages into ProcessedDocument format.
    
    Usage:
        processor = ASPXProcessor()
        documents = processor.process_all()
    """

    def __init__(self, config: Optional[ASPXProcessorConfig] = None):
        self.config = config or ASPXProcessorConfig()
        self.input_dir = Path(self.config.input_dir)
        logger.info(f"ASPXProcessor initialized. Input: {self.input_dir}")

    def process_page(self, page_path: Path, domain: str) -> Optional[ProcessedDocument]:
        """Process a single ASPX page JSON file."""
        try:
            with open(page_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content_text = data.get("content_text", "")
            
            # Skip pages with too little content
            if len(content_text) < self.config.min_content_length:
                return None
            
            # Convert structured content to StructuredItem objects
            structured_items = []
            for item in data.get("structured_content", []):
                structured_items.append(StructuredItem(
                    item_type=item.get("type", "text"),
                    text=item.get("text", ""),
                    page_num=1,  # ASPX pages are single-page
                    level=item.get("level", 0),
                    section_path=[],  # Will be built during indexing
                ))
            
            # Create page content
            page_content = PageContent(
                page_num=1,
                text=content_text,
                char_count=len(content_text),
                has_tables=len(data.get("tables", [])) > 0,
                headers=[item["text"] for item in data.get("structured_content", [])
                        if item.get("type") == "header"],
                structured_items=structured_items,
            )
            
            # Map domain to standard domain
            standard_domain = DOMAIN_MAPPING.get(domain, domain)
            
            return ProcessedDocument(
                filename=page_path.name,
                filepath=str(page_path),
                page_count=1,
                pages=[page_content],
                total_chars=len(content_text),
                has_tables=page_content.has_tables,
                detected_headers=[item["text"] for item in data.get("structured_content", [])
                                 if item.get("type") == "header"],
                processing_method="aspx",
                structured_items=structured_items,
                domain=standard_domain,
                metadata={
                    "url": data.get("url", ""),
                    "title": data.get("title", ""),
                    "original_domain": domain,
                    "source_type": "aspx",
                    "scraped_at": data.get("scraped_at", ""),
                },
            )
        except Exception as e:
            logger.error(f"Error processing {page_path}: {e}")
            return None

    def process_domain(self, domain: str) -> list[ProcessedDocument]:
        """Process all pages for a specific domain."""
        domain_dir = self.input_dir / domain
        if not domain_dir.exists():
            logger.warning(f"Domain directory not found: {domain_dir}")
            return []
        
        documents = []
        for page_file in sorted(domain_dir.glob("page_*.json")):
            doc = self.process_page(page_file, domain)
            if doc:
                documents.append(doc)
        
        logger.info(f"Processed {len(documents)} pages from domain: {domain}")
        return documents

    def process_all(self) -> list[ProcessedDocument]:
        """Process all ASPX pages from all domains."""
        all_documents = []
        
        for domain_dir in sorted(self.input_dir.iterdir()):
            if domain_dir.is_dir():
                domain = domain_dir.name
                documents = self.process_domain(domain)
                all_documents.extend(documents)
        
        logger.info(f"Total ASPX documents processed: {len(all_documents)}")
        return all_documents

    def get_domain_stats(self) -> dict[str, int]:
        """Get statistics about scraped pages per domain."""
        stats = {}
        for domain_dir in sorted(self.input_dir.iterdir()):
            if domain_dir.is_dir():
                count = len(list(domain_dir.glob("page_*.json")))
                stats[domain_dir.name] = count
        return stats

