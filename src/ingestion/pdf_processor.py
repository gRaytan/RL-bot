"""
PDF Processor using Docling for document parsing.

Extracts text from PDFs with structure preservation:
- Page-by-page extraction
- Table detection and formatting
- Hebrew RTL text handling
- Header/section detection for ToC
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# Docling imports
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not available. Install with: pip install docling")

# Fallback to pypdf
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class StructuredItem:
    """A structured item extracted from a document."""
    item_type: str  # "header", "text", "table", "list"
    text: str
    page_num: int
    level: int = 0  # Hierarchy level (0 = top level)
    section_path: list[str] = field(default_factory=list)  # ["כיסויים", "נזקי רכוש"]


@dataclass
class PageContent:
    """Content extracted from a single page."""
    page_num: int
    text: str
    char_count: int
    has_tables: bool = False
    has_headers: bool = False
    headers: list[str] = field(default_factory=list)
    structured_items: list[StructuredItem] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    """Result of processing a PDF document."""
    filepath: str
    filename: str
    page_count: int
    pages: list[PageContent]
    total_chars: int
    has_tables: bool = False
    detected_headers: list[str] = field(default_factory=list)
    processing_method: str = "docling"  # "docling", "pypdf", or "aspx"
    error: Optional[str] = None
    structured_items: list[StructuredItem] = field(default_factory=list)
    domain: Optional[str] = None  # Insurance domain (car, health, etc.)
    metadata: dict = field(default_factory=dict)  # Additional metadata (url, title, etc.)

    def get_page_texts(self) -> list[str]:
        """Get list of page texts for chunking."""
        return [p.text for p in self.pages]

    def get_structured_text(self) -> str:
        """Get text with structure markers preserved."""
        parts = []
        for item in self.structured_items:
            if item.item_type == "header":
                # Add header with level indicator
                prefix = "#" * (item.level + 1)
                parts.append(f"\n{prefix} {item.text}\n")
            elif item.item_type == "table":
                parts.append(f"\n[טבלה]\n{item.text}\n")
            elif item.item_type == "list":
                parts.append(f"• {item.text}")
            else:
                parts.append(item.text)
        return "\n".join(parts)


class PDFProcessor:
    """
    Process PDF documents using Docling with pypdf fallback.
    
    Features:
    - Structure-aware parsing (tables, headers, lists)
    - Page-by-page extraction
    - Hebrew text support
    - Automatic fallback to pypdf if Docling fails
    """

    def __init__(self, use_ocr: bool = False):
        """
        Initialize PDF processor.
        
        Args:
            use_ocr: Enable OCR for scanned PDFs (slower but more accurate)
        """
        self.use_ocr = use_ocr
        self._converter = None
        
        if DOCLING_AVAILABLE:
            self._init_docling()

    def _init_docling(self):
        """Initialize Docling converter with options."""
        try:
            # Configure PDF pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.use_ocr
            pipeline_options.do_table_structure = True
            
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
            logger.info("Docling converter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Docling: {e}")
            self._converter = None

    def process(self, filepath: str) -> ProcessedDocument:
        """
        Process a PDF file and extract content.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            ProcessedDocument with extracted pages
        """
        path = Path(filepath)
        
        if not path.exists():
            return ProcessedDocument(
                filepath=str(path),
                filename=path.name,
                page_count=0,
                pages=[],
                total_chars=0,
                error=f"File not found: {filepath}"
            )

        # Try Docling first
        if self._converter:
            try:
                return self._process_with_docling(path)
            except Exception as e:
                logger.warning(f"Docling failed for {path.name}: {e}, falling back to pypdf")

        # Fallback to pypdf
        if PYPDF_AVAILABLE:
            try:
                return self._process_with_pypdf(path)
            except Exception as e:
                logger.error(f"pypdf also failed for {path.name}: {e}")
                return ProcessedDocument(
                    filepath=str(path),
                    filename=path.name,
                    page_count=0,
                    pages=[],
                    total_chars=0,
                    error=str(e)
                )

        return ProcessedDocument(
            filepath=str(path),
            filename=path.name,
            page_count=0,
            pages=[],
            total_chars=0,
            error="No PDF processing library available"
        )

    def _process_with_docling(self, path: Path) -> ProcessedDocument:
        """Process PDF using Docling with structure-aware extraction."""
        logger.info(f"Processing with Docling: {path.name}")

        result = self._converter.convert(str(path))
        doc = result.document

        # Extract structured items per page
        page_texts: dict[int, list[str]] = {}
        page_has_tables: dict[int, bool] = {}
        page_headers: dict[int, list[str]] = {}
        page_structured_items: dict[int, list[StructuredItem]] = {}
        all_structured_items: list[StructuredItem] = []

        # Track current section hierarchy
        current_section_path: list[str] = []

        for item, level in doc.iterate_items():
            # Get page number from provenance
            page_no = 1  # default
            if hasattr(item, 'prov') and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, 'page_no'):
                    page_no = prov.page_no

            # Initialize page data
            if page_no not in page_texts:
                page_texts[page_no] = []
                page_has_tables[page_no] = False
                page_headers[page_no] = []
                page_structured_items[page_no] = []

            item_type = type(item).__name__

            # Handle different item types
            if 'SectionHeader' in item_type or 'Heading' in item_type or 'Title' in item_type:
                # Section header - update hierarchy
                text = item.text if hasattr(item, 'text') and item.text else ""
                if text:
                    # Adjust section path based on level
                    if level < len(current_section_path):
                        current_section_path = current_section_path[:level]
                    current_section_path.append(text)

                    page_texts[page_no].append(f"\n## {text}\n")
                    page_headers[page_no].append(text)

                    structured_item = StructuredItem(
                        item_type="header",
                        text=text,
                        page_num=page_no,
                        level=level,
                        section_path=current_section_path.copy(),
                    )
                    page_structured_items[page_no].append(structured_item)
                    all_structured_items.append(structured_item)

            elif 'Table' in item_type:
                # Table - export as markdown
                page_has_tables[page_no] = True
                table_text = ""
                if hasattr(item, 'export_to_markdown'):
                    try:
                        table_text = item.export_to_markdown()
                    except Exception:
                        table_text = "[טבלה]"
                elif hasattr(item, 'text') and item.text:
                    table_text = item.text
                else:
                    table_text = "[טבלה]"

                page_texts[page_no].append(f"\n{table_text}\n")

                structured_item = StructuredItem(
                    item_type="table",
                    text=table_text,
                    page_num=page_no,
                    level=level,
                    section_path=current_section_path.copy(),
                )
                page_structured_items[page_no].append(structured_item)
                all_structured_items.append(structured_item)

            elif 'List' in item_type:
                # List item - preserve bullet structure
                text = item.text if hasattr(item, 'text') and item.text else ""
                if text:
                    page_texts[page_no].append(f"• {text}")

                    structured_item = StructuredItem(
                        item_type="list",
                        text=text,
                        page_num=page_no,
                        level=level,
                        section_path=current_section_path.copy(),
                    )
                    page_structured_items[page_no].append(structured_item)
                    all_structured_items.append(structured_item)

            else:
                # Regular text
                text = item.text if hasattr(item, 'text') and item.text else ""
                if text:
                    page_texts[page_no].append(text)

                    structured_item = StructuredItem(
                        item_type="text",
                        text=text,
                        page_num=page_no,
                        level=level,
                        section_path=current_section_path.copy(),
                    )
                    page_structured_items[page_no].append(structured_item)
                    all_structured_items.append(structured_item)

        # Build PageContent objects
        pages = []
        all_headers = []
        total_chars = 0
        has_any_tables = False

        for page_no in sorted(page_texts.keys()):
            text = "\n".join(page_texts[page_no])
            char_count = len(text)
            total_chars += char_count
            has_tables = page_has_tables.get(page_no, False)
            headers = page_headers.get(page_no, [])
            structured_items = page_structured_items.get(page_no, [])

            if has_tables:
                has_any_tables = True
            all_headers.extend(headers)

            pages.append(PageContent(
                page_num=page_no,
                text=text,
                char_count=char_count,
                has_tables=has_tables,
                has_headers=len(headers) > 0,
                headers=headers,
                structured_items=structured_items,
            ))

        return ProcessedDocument(
            filepath=str(path),
            filename=path.name,
            page_count=len(pages),
            pages=pages,
            total_chars=total_chars,
            has_tables=has_any_tables,
            detected_headers=all_headers,
            processing_method="docling",
            structured_items=all_structured_items,
        )

    def _process_with_pypdf(self, path: Path) -> ProcessedDocument:
        """Process PDF using pypdf as fallback."""
        logger.info(f"Processing with pypdf: {path.name}")

        reader = PdfReader(str(path))
        pages = []
        total_chars = 0

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            char_count = len(text)
            total_chars += char_count

            pages.append(PageContent(
                page_num=i + 1,
                text=text,
                char_count=char_count,
                has_tables=False,  # pypdf doesn't detect tables
                has_headers=False,
            ))

        return ProcessedDocument(
            filepath=str(path),
            filename=path.name,
            page_count=len(pages),
            pages=pages,
            total_chars=total_chars,
            has_tables=False,
            processing_method="pypdf"
        )

    def process_directory(
        self,
        directory: str,
        pattern: str = "*.pdf"
    ) -> list[ProcessedDocument]:
        """
        Process all PDFs in a directory.

        Args:
            directory: Path to directory containing PDFs
            pattern: Glob pattern for matching files

        Returns:
            List of ProcessedDocument objects
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        pdf_files = list(dir_path.glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        results = []
        for pdf_path in pdf_files:
            try:
                doc = self.process(str(pdf_path))
                results.append(doc)
                logger.info(f"Processed {pdf_path.name}: {doc.page_count} pages, {doc.total_chars} chars")
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results.append(ProcessedDocument(
                    filepath=str(pdf_path),
                    filename=pdf_path.name,
                    page_count=0,
                    pages=[],
                    total_chars=0,
                    error=str(e)
                ))

        return results


# Quick test
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    processor = PDFProcessor(use_ocr=False)

    if len(sys.argv) > 1:
        # Process specific file
        doc = processor.process(sys.argv[1])
        print(f"\n{'='*50}")
        print(f"File: {doc.filename}")
        print(f"Pages: {doc.page_count}")
        print(f"Total chars: {doc.total_chars}")
        print(f"Method: {doc.processing_method}")
        if doc.error:
            print(f"Error: {doc.error}")
        else:
            print(f"\nFirst 500 chars:")
            print(doc.pages[0].text[:500] if doc.pages else "No content")
    else:
        # Process sample directory
        docs = processor.process_directory("data/harel_pdfs/pdfs/")
        print(f"\nProcessed {len(docs)} documents")
        for doc in docs[:3]:
            print(f"  - {doc.filename}: {doc.page_count} pages, {doc.total_chars} chars")

