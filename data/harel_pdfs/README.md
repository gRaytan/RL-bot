# Harel Insurance PDFs

This directory contains insurance-related PDF documents extracted from Harel's website (https://www.harel-group.co.il/insurance/).

## Summary

- **Total PDFs**: 72
- **Source**: https://www.harel-group.co.il/insurance/
- **Extraction Date**: 2026-02-15
- **Extraction Method**: Python web scraping with BeautifulSoup
- **Pattern**: `https://media.harel-group.co.il/media/***.pdf`

## Directory Structure

```
harel_pdfs/
├── README.md              # This file
├── scrape_pdfs.py         # Web scraper script
├── download_pdfs.py       # PDF download script
├── pdf_urls.txt           # List of all PDF URLs
├── pdfs/                  # Downloaded PDF files (72 files)
├── crawl/                 # Temporary crawl data
└── venv/                  # Python virtual environment
```

## Insurance Categories Covered

The PDFs cover various insurance domains including:

- **Car Insurance** (ביטוח רכב)
  - Comprehensive policies
  - Third-party liability
  - Compulsory insurance
  - Claims and forms

- **Personal Accident Insurance** (תאונות אישיות)
  - General policies
  - Scuba diving
  - Extreme sports
  - Family coverage

- **Home Insurance** (ביטוח דירה)
  - Property damage claims
  - Third-party liability

- **Life Insurance** (ביטוח חיים)
  - Mortgage insurance
  - Policy terms

- **Health & Dental** (בריאות ושיניים)
  - Health insurance policies
  - Dental insurance

- **Business Insurance** (ביטוח לעסק)
  - Third-party claims
  - Policy cancellations

- **Travel Insurance** (ביטוח נסיעות לחוץ)
  - Foreign travel policies

## Document Types

- Policy documents (פוליסות)
- Terms and conditions (תנאים)
- Claim forms (טפסי תביעה)
- Cancellation forms (טפסי ביטול)
- Disclosure documents (גילוי נאות)
- Claims procedures (נהלים)
- FAQ and guidelines (שאלות ותשובות)

## Usage

To re-run the extraction:

```bash
# Activate virtual environment
source venv/bin/activate

# Scrape PDF URLs
python scrape_pdfs.py

# Download PDFs
python download_pdfs.py
```

## Technical Details

- **Web Scraping**: BeautifulSoup4 + Requests
- **Crawl Depth**: 3 levels
- **Pages Visited**: 48
- **Rate Limiting**: 0.5s delay between requests
- **Total Download Size**: ~12.5 MB
