# tests/test_system.py
"""
Unit tests validating chunking, vector search and gatekeeping.
Run with `pytest -q` in project root.
"""

from __future__ import annotations
import numpy as np

import sys
from pathlib import Path
# Add the src folder to sys.path if it's not already there
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from pdf_parser import Section
from chunker import TextChunker
from vector_store import VectorStore
from query_engine import QueryEngine

def test_chunker_overlap():
    chunker = TextChunker(max_chunk_size=50, overlap=10)
    text = "The quick brown fox jumps over the lazy dog. " * 5
    sec = Section(text, 1, "paragraph")
    chunks = chunker.chunk_sections([sec])
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        overlap = a.text[-10:].strip()
        assert overlap in b.text

def test_vector_store_nn():
    store = VectorStore()
    v1 = np.array([1.0, 0.0, 0.0], dtype="float32")
    v2 = np.array([0.0, 1.0, 0.0], dtype="float32")
    v3 = np.array([0.0, 0.0, 1.0], dtype="float32")
    sections = [Section("A", 1), Section("B", 1), Section("C", 1)]
    store.add_vectors(np.vstack([v1, v2, v3]), sections)
    q = np.array([0.1, 0.9, 0.0], dtype="float32")
    results = store.search(q, 1)
    assert results[0][0].text == "B"

def test_query_engine_gatekeeping():
    engine = QueryEngine()
    fact = Section("The sky is blue.", 1, "paragraph")
    emb = engine.embedder.embed([fact.text])
    engine.vector_store.add_vectors(emb, [fact])
    engine.document_loaded = True
    ans_rel = engine.answer_query("What colour is the sky?")
    assert "blue" in ans_rel.lower() or "cannot" in ans_rel.lower()
    ans_irrel = engine.answer_query("How many wheels does a car have?")
    assert "relevant" in ans_irrel.lower() or "cannot" in ans_irrel.lower()
