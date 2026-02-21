"""
Web Scraper for Harel Insurance ASPX pages.

Scrapes content from all insurance type pages with structure preservation:
- Section headers and hierarchy
- Tables (converted to markdown)
- Lists and bullet points
- Legal clauses and conditions
"""

import logging
import time
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin, urlparse
import json

import requests
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Insurance types to scrape
INSURANCE_TYPES = [
    "car",
    "life", 
    "travel",
    "health",
    "dental",
    "mortgage",
    "business",
    "apartment",
    "personal-accident",
    "long-term-care",
    "foreign",
    "loss-of-working-ability",
    "diseases-disabilities",
]

BASE_URL = "https://www.harel-group.co.il/insurance/"


@dataclass
class ScrapedPage:
    """A scraped web page with extracted content."""
    url: str
    title: str
    domain: str  # Insurance type
    content_text: str  # Plain text content
    content_html: str  # Original HTML
    structured_content: list[dict] = field(default_factory=list)  # Structured items
    tables: list[str] = field(default_factory=list)  # Tables as markdown
    links: list[str] = field(default_factory=list)  # Child page links
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


@dataclass
class ScraperConfig:
    """Configuration for web scraper."""
    max_depth: int = 3
    delay_seconds: float = 0.5
    timeout_seconds: int = 15
    max_pages_per_domain: int = 50
    output_dir: str = "data/raw/aspx"


class HarelWebScraper:
    """
    Scrape Harel insurance web pages with structure preservation.
    
    Usage:
        scraper = HarelWebScraper()
        pages = scraper.scrape_all()
        scraper.save_pages(pages)
    """

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.visited_urls: set[str] = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"HarelWebScraper initialized. Output: {self.config.output_dir}")

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page and return HTML content."""
        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
            return None

    def _extract_text_with_structure(self, soup: BeautifulSoup) -> tuple[str, list[dict]]:
        """Extract text content while preserving structure."""
        structured_items = []
        text_parts = []
        
        # Find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if not main_content:
            main_content = soup.body or soup
        
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'table', 'div']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                text = element.get_text(strip=True)
                if text:
                    level = int(element.name[1])
                    structured_items.append({
                        "type": "header",
                        "level": level,
                        "text": text,
                    })
                    text_parts.append(f"\n{'#' * level} {text}\n")
            elif element.name == 'li':
                text = element.get_text(strip=True)
                if text:
                    structured_items.append({"type": "list", "text": text})
                    text_parts.append(f"• {text}")
            elif element.name == 'table':
                table_md = self._table_to_markdown(element)
                if table_md:
                    structured_items.append({"type": "table", "text": table_md})
                    text_parts.append(f"\n{table_md}\n")
            elif element.name in ['p', 'div']:
                text = element.get_text(strip=True)
                if text and len(text) > 20:  # Skip short divs
                    structured_items.append({"type": "text", "text": text})
                    text_parts.append(text)
        
        return "\n".join(text_parts), structured_items

    def _table_to_markdown(self, table: Tag) -> str:
        """Convert HTML table to markdown format."""
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cells.append(td.get_text(strip=True).replace('|', '\\|'))
            if cells:
                rows.append('| ' + ' | '.join(cells) + ' |')
        
        if len(rows) >= 2:
            # Add header separator
            header = rows[0]
            separator = '|' + '|'.join(['---'] * header.count('|')) + '|'
            rows.insert(1, separator)

        return '\n'.join(rows) if rows else ""

    def _extract_child_links(self, soup: BeautifulSoup, base_url: str, domain: str) -> list[str]:
        """Extract child page links within the same insurance domain."""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(base_url, href)

            # Only include links within the same insurance domain
            if (f'/insurance/{domain}' in absolute_url and
                absolute_url not in self.visited_urls and
                not absolute_url.endswith('.pdf') and
                'harel-group.co.il' in absolute_url):
                links.append(absolute_url)

        return list(set(links))

    def _extract_tables(self, soup: BeautifulSoup) -> list[str]:
        """Extract all tables as markdown."""
        tables = []
        for table in soup.find_all('table'):
            md = self._table_to_markdown(table)
            if md:
                tables.append(md)
        return tables

    def scrape_page(self, url: str, domain: str) -> Optional[ScrapedPage]:
        """Scrape a single page."""
        if url in self.visited_urls:
            return None

        self.visited_urls.add(url)
        html = self._fetch_page(url)

        if not html:
            return ScrapedPage(
                url=url, title="", domain=domain,
                content_text="", content_html="",
                error="Failed to fetch page"
            )

        soup = BeautifulSoup(html, 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Extract structured content
        content_text, structured_content = self._extract_text_with_structure(soup)

        # Extract tables
        tables = self._extract_tables(soup)

        # Extract child links
        child_links = self._extract_child_links(soup, url, domain)

        return ScrapedPage(
            url=url,
            title=title,
            domain=domain,
            content_text=content_text,
            content_html=html,
            structured_content=structured_content,
            tables=tables,
            links=child_links,
        )

    def scrape_domain(self, domain: str, max_depth: int = None) -> list[ScrapedPage]:
        """Scrape all pages for a specific insurance domain."""
        max_depth = max_depth or self.config.max_depth
        pages = []
        start_url = f"{BASE_URL}{domain}"

        logger.info(f"Scraping domain: {domain} from {start_url}")

        # BFS crawl
        queue = [(start_url, 0)]  # (url, depth)

        while queue and len(pages) < self.config.max_pages_per_domain:
            url, depth = queue.pop(0)

            if depth > max_depth or url in self.visited_urls:
                continue

            page = self.scrape_page(url, domain)
            if page and not page.error:
                pages.append(page)
                logger.info(f"  [{len(pages)}] Scraped: {url[:80]}...")

                # Add child links to queue
                if depth < max_depth:
                    for link in page.links:
                        if link not in self.visited_urls:
                            queue.append((link, depth + 1))

            # Rate limiting
            time.sleep(self.config.delay_seconds)

        logger.info(f"Domain {domain}: scraped {len(pages)} pages")
        return pages

    def scrape_all(self, domains: list[str] = None) -> list[ScrapedPage]:
        """Scrape all insurance domains."""
        domains = domains or INSURANCE_TYPES
        all_pages = []

        for domain in domains:
            pages = self.scrape_domain(domain)
            all_pages.extend(pages)

        logger.info(f"Total pages scraped: {len(all_pages)}")
        return all_pages

    def save_pages(self, pages: list[ScrapedPage]) -> Path:
        """Save scraped pages to JSON files organized by domain."""
        output_dir = Path(self.config.output_dir)

        # Group by domain
        by_domain: dict[str, list[ScrapedPage]] = {}
        for page in pages:
            if page.domain not in by_domain:
                by_domain[page.domain] = []
            by_domain[page.domain].append(page)

        # Save each domain
        for domain, domain_pages in by_domain.items():
            domain_dir = output_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)

            # Save index
            index_data = {
                "domain": domain,
                "page_count": len(domain_pages),
                "scraped_at": datetime.now().isoformat(),
                "pages": [
                    {
                        "url": p.url,
                        "title": p.title,
                        "text_length": len(p.content_text),
                        "table_count": len(p.tables),
                    }
                    for p in domain_pages
                ]
            }

            with open(domain_dir / "index.json", 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)

            # Save each page
            for i, page in enumerate(domain_pages):
                page_data = {
                    "url": page.url,
                    "title": page.title,
                    "domain": page.domain,
                    "content_text": page.content_text,
                    "structured_content": page.structured_content,
                    "tables": page.tables,
                    "scraped_at": page.scraped_at,
                }

                filename = f"page_{i:03d}.json"
                with open(domain_dir / filename, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(pages)} pages to {output_dir}")
        return output_dir

