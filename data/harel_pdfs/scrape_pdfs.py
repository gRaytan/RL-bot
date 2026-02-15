#!/usr/bin/env python3
"""
Scrape PDF documents from Harel insurance website
"""
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
from pathlib import Path

# Configuration
BASE_URL = "https://www.harel-group.co.il/insurance/"
PDF_PATTERN = re.compile(r'https://media\.harel-group\.co\.il/media/[^"\']+\.pdf', re.IGNORECASE)
OUTPUT_FILE = "pdf_urls.txt"
visited_urls = set()
pdf_urls = set()
max_depth = 3

def get_page_content(url):
    """Fetch page content with proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_pdf_urls(html_content, base_url):
    """Extract PDF URLs from HTML content"""
    # Search for PDF URLs using regex
    found_pdfs = PDF_PATTERN.findall(html_content)
    return set(found_pdfs)

def extract_links(html_content, base_url):
    """Extract all links from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = set()

    for tag in soup.find_all(['a', 'link'], href=True):
        href = tag['href']
        absolute_url = urljoin(base_url, href)

        # Only include URLs from the same domain
        if 'harel-group.co.il' in absolute_url:
            links.add(absolute_url)

    return links

def crawl_site(start_url, current_depth=0):
    """Recursively crawl the site to find PDF URLs"""
    global visited_urls, pdf_urls

    if current_depth > max_depth or start_url in visited_urls:
        return

    print(f"[Depth {current_depth}] Crawling: {start_url}")
    visited_urls.add(start_url)

    html_content = get_page_content(start_url)
    if not html_content:
        return

    # Extract PDF URLs
    found_pdfs = extract_pdf_urls(html_content, start_url)
    if found_pdfs:
        print(f"  Found {len(found_pdfs)} PDFs on this page")
        pdf_urls.update(found_pdfs)

    # Extract and follow links
    if current_depth < max_depth:
        links = extract_links(html_content, start_url)
        insurance_links = [l for l in links if '/insurance' in l and l not in visited_urls]

        print(f"  Found {len(insurance_links)} insurance-related links to crawl")

        for link in insurance_links[:20]:  # Limit to avoid too many requests
            time.sleep(0.5)  # Be polite
            crawl_site(link, current_depth + 1)

def main():
    print(f"Starting PDF extraction from {BASE_URL}")
    print(f"Max depth: {max_depth}\n")

    crawl_site(BASE_URL)

    print(f"\n{'='*60}")
    print(f"Crawling complete!")
    print(f"Total pages visited: {len(visited_urls)}")
    print(f"Total PDFs found: {len(pdf_urls)}")
    print(f"{'='*60}\n")

    # Save PDF URLs to file
    output_path = Path(__file__).parent / OUTPUT_FILE
    with open(output_path, 'w') as f:
        for pdf_url in sorted(pdf_urls):
            f.write(f"{pdf_url}\n")
            print(pdf_url)

    print(f"\nPDF URLs saved to: {output_path}")

if __name__ == "__main__":
    main()
