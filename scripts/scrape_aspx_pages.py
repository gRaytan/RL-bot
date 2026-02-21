#!/usr/bin/env python3
"""
Scrape ASPX pages from Harel insurance website.

Usage:
    python scripts/scrape_aspx_pages.py
    python scripts/scrape_aspx_pages.py --domains car,health,apartment
    python scripts/scrape_aspx_pages.py --max-depth 2 --max-pages 20
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.web_scraper import HarelWebScraper, ScraperConfig, INSURANCE_TYPES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Scrape Harel insurance ASPX pages")
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help=f"Comma-separated list of domains to scrape. Available: {', '.join(INSURANCE_TYPES)}"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum crawl depth (default: 3)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum pages per domain (default: 50)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/aspx",
        help="Output directory (default: data/raw/aspx)"
    )
    
    args = parser.parse_args()
    
    # Parse domains
    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
        invalid = [d for d in domains if d not in INSURANCE_TYPES]
        if invalid:
            logger.error(f"Invalid domains: {invalid}")
            logger.info(f"Available domains: {INSURANCE_TYPES}")
            sys.exit(1)
    
    # Create config
    config = ScraperConfig(
        max_depth=args.max_depth,
        max_pages_per_domain=args.max_pages,
        delay_seconds=args.delay,
        output_dir=args.output_dir,
    )
    
    # Run scraper
    logger.info("=" * 60)
    logger.info("Harel Insurance ASPX Page Scraper")
    logger.info("=" * 60)
    logger.info(f"Domains: {domains or 'ALL'}")
    logger.info(f"Max depth: {config.max_depth}")
    logger.info(f"Max pages per domain: {config.max_pages_per_domain}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)
    
    scraper = HarelWebScraper(config)
    pages = scraper.scrape_all(domains)
    
    # Save results
    output_path = scraper.save_pages(pages)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 60)
    
    # Count by domain
    domain_counts = {}
    for page in pages:
        domain_counts[page.domain] = domain_counts.get(page.domain, 0) + 1
    
    logger.info(f"\nPages scraped by domain:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain}: {count} pages")
    
    logger.info(f"\nTotal pages: {len(pages)}")
    logger.info(f"Output directory: {output_path}")
    
    # Count tables
    total_tables = sum(len(p.tables) for p in pages)
    logger.info(f"Total tables extracted: {total_tables}")


if __name__ == "__main__":
    main()

