#!/usr/bin/env python3
"""
Download all PDFs from the extracted URL list
"""
import requests
import time
from pathlib import Path
from urllib.parse import urlparse
import sys

def download_pdf(url, output_dir):
    """Download a single PDF file"""
    try:
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name

        output_path = output_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            print(f"  ✓ Already exists: {filename}")
            return True

        # Download the file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Save the file
        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"  ✓ Downloaded: {filename} ({len(response.content) / 1024:.1f} KB)")
        return True

    except Exception as e:
        print(f"  ✗ Error downloading {url}: {e}")
        return False

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    pdf_urls_file = base_dir / "pdf_urls.txt"
    output_dir = base_dir / "pdfs"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Read PDF URLs
    with open(pdf_urls_file, 'r') as f:
        pdf_urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(pdf_urls)} PDFs to download")
    print(f"Output directory: {output_dir}\n")

    # Download each PDF
    successful = 0
    failed = 0

    for i, url in enumerate(pdf_urls, 1):
        print(f"[{i}/{len(pdf_urls)}] {url}")
        if download_pdf(url, output_dir):
            successful += 1
        else:
            failed += 1

        # Be polite - add delay between requests
        if i < len(pdf_urls):
            time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(pdf_urls)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
