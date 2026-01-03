# src/chunker.py
"""
Text chunking utilities.

This class splits text into manageable chunks with overlap.  
It respects paragraph boundaries and then uses simple sentence splitting for long paragraphs.  
Overlap preserves context across boundaries.
"""

from __future__ import annotations
import math
import re
from typing import List
from logger import get_logger
from pdf_parser import Section

logger = get_logger(__name__)

class TextChunker:
    def __init__(self, max_chunk_size: int, overlap: int):
        if max_chunk_size <= 0 or overlap < 0 or overlap >= max_chunk_size:
            raise ValueError("Invalid chunk size or overlap")
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_sections(self, sections: List[Section]) -> List[Section]:
        chunks: List[Section] = []
        for sec in sections:
            text = re.sub(r"\r\n|\r", "\n", sec.text).strip()
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            for para in paragraphs:
                if len(para) <= self.max_chunk_size:
                    chunks.append(Section(para, sec.page, sec.section_type))
                else:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(current) + len(sent) + 1 > self.max_chunk_size:
                            chunk_text = current.strip()
                            if chunk_text:
                                chunks.append(Section(chunk_text, sec.page, sec.section_type))
                            overlap_text = chunk_text[-self.overlap:] if self.overlap > 0 else ""
                            current = overlap_text + " " + sent
                        else:
                            current += (" " if current else "") + sent
                    if current.strip():
                        chunks.append(Section(current.strip(), sec.page, sec.section_type))
        avg_len = math.floor(sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0
        logger.info(f"Chunked {len(sections)} sections into {len(chunks)} chunks (avg length {avg_len} chars).")
        return chunks
