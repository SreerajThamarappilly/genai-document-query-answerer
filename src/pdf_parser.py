# src/pdf_parser.py
"""
PDF parsing utilities for the GenAI document QA system.

This module defines the ``PDFParser`` class used to extract textual content from PDF files.
It uses PyMuPDF for born-digital PDFs and optionally falls back to OCR via Tesseract for scanned pages.
If pytesseract is not installed, the OCR fallback logs a warning and returns an empty string.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
from PIL import Image
from logger import get_logger

# Optional OCR
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore

logger = get_logger(__name__)


@dataclass
class Section:
    """Represents a logical section of a document."""
    text: str
    page: int
    section_type: str = "paragraph"


class PDFParser:
    """Parse PDF documents into sections labelled as paragraphs, tables or forms."""

    def parse(self, pdf_path: str) -> List[Section]:
        sections: List[Section] = []
        try:
            document = fitz.open(pdf_path)
        except Exception as exc:
            logger.error(f"Could not open PDF '{pdf_path}': {exc}")
            return sections

        logger.info(f"Parsing document '{pdf_path}' with {document.page_count} pages.")
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            page_idx = page_number + 1
            blocks = page.get_text("blocks")
            if not blocks:
                # Fallback to OCR if PyMuPDF finds no text and OCR is available
                logger.warning(f"Page {page_idx} has no embedded text; attempting OCR.")
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    sections.append(Section(ocr_text, page_idx, "paragraph"))
                continue
            # Sort blocks by y (row) then x (column) to maintain reading order
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            for block in blocks:
                _, _, _, _, text, _, _ = block
                text = text.strip()
                if not text:
                    continue
                sec_type = self._classify_text(text)
                sections.append(Section(text, page_idx, sec_type))
        document.close()
        logger.info(f"Finished parsing '{pdf_path}'. Extracted {len(sections)} sections.")
        return sections

    def _ocr_page(self, page: fitz.Page) -> str:
        """Render a page to an image and run OCR via pytesseract."""
        if pytesseract is None:
            logger.warning("pytesseract not installed; cannot OCR scanned pages.")
            return ""
        try:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as exc:
            logger.error(f"OCR failed on page {page.number+1}: {exc}")
            return ""

    def _classify_text(self, text: str) -> str:
        """Heuristically classify text as table, form or paragraph."""
        stripped = re.sub(r"\s+", " ", text)
        lines = text.splitlines()
        # Table: many lines with identical counts of delimiters
        if len(lines) >= 3:
            delim_counts = [line.count(",") + line.count("\t") + line.count("|") for line in lines[:8]]
            if len(set(delim_counts)) == 1 and delim_counts[0] > 0:
                return "table"
        # Form: many lines contain a colon with a short label
        form_like = 0
        for line in lines:
            if ":" in line and len(line.split(":")[0]) < 30:
                form_like += 1
        if form_like >= max(3, len(lines)//2):
            return "form"
        return "paragraph"
