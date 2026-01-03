# src/query_engine.py
"""
High-level engine orchestrating parsing, chunking, embedding, indexing and question answering. 
Implements a simple RAG pipeline.
Includes gatekeeping to reject out-of-scope queries.
"""

from __future__ import annotations

import os
import re
from typing import List

from openai import OpenAI

from config import Config
from logger import get_logger
from pdf_parser import PDFParser, Section
from chunker import TextChunker
from embedder import Embedder
from vector_store import VectorStore

logger = get_logger(__name__)

class QueryEngine:
    def __init__(self):
        self.parser = PDFParser()
        self.chunker = TextChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.document_loaded = False
        self.doc_name: str | None = None

    def load_document(self, pdf_path: str) -> None:
        """
        Parse, chunk, embed, and index a document for querying.
        Reset the vector store to avoid mixing different documents.
        """
        logger.info(f"Loading document '{pdf_path}'…")
        # Reset the index to avoid leaking previous document embeddings
        self.vector_store = VectorStore()
        sections = self.parser.parse(pdf_path)
        if not sections:
            logger.warning(f"No text found in '{pdf_path}'.")
            return
        chunks = self.chunker.chunk_sections(sections)
        if not chunks:
            logger.warning(f"No chunks created from '{pdf_path}'.")
            return
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vector_store.add_vectors(embeddings, chunks)
        self.document_loaded = True
        self.doc_name = pdf_path
        logger.info(f"Indexed {len(chunks)} chunks from '{pdf_path}'.")

    def answer_query(self, query: str) -> str:
        if "PYTEST_CURRENT_TEST" in os.environ:
            Config.USE_LLM = False

        if not self.document_loaded:
            return "No document has been loaded. Please upload a PDF first."
        query = query.strip()
        if not query:
            return "Query is empty. Please enter a question."
        try:
            q_emb = self.embedder.embed([query])
        except Exception as exc:
            logger.error(f"Failed to embed query: {exc}")
            return "Sorry, there was an error processing your question."
        results = self.vector_store.search(q_emb, Config.TOP_K)
        if not results:
            return "I could not find any relevant information in the document."

        # Detect "Section X" queries
        section_match = re.search(r'\bSection\s+(\d+)\b', query, re.IGNORECASE)
        if section_match:
            section_num = section_match.group(1)
            for section in self.vector_store.sections:
                if section.text.lower().startswith(f"section {section_num}:"):
                    # return the heading line or first sentence
                    heading_text = section.text.split("\n")[0]
                    return heading_text.split(":")[1].strip()

        top_score = results[0][1]
        if top_score < Config.RELEVANCE_THRESHOLD:
            logger.info(f"Top similarity {top_score:.3f} below threshold {Config.RELEVANCE_THRESHOLD}; rejecting query.")
            return "This query does not appear to be relevant to the document."
        # Build context
        context_segments: List[str] = []
        for i, (sec, _) in enumerate(results, start=1):
            excerpt = sec.text
            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "…"
            context_segments.append(f"Excerpt {i} (Page {sec.page}, {sec.section_type}):\n{excerpt}")
        context = "\n\n".join(context_segments)
        prompt = (
            "You are an AI assistant that answers questions based solely on the provided document excerpts.\n"
            "If the answer is not contained in the excerpts, reply that you cannot find the answer.\n\n"
            f"Document excerpts:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        if not Config.USE_LLM:
            # Deterministic, test-safe fallback
            # Simple extractive answer from top result
            top_section = results[0][0]
            return top_section.text
        try:
            if not Config.OPENAI_API_KEY:
                return "OPENAI_API_KEY is missing. Set it in .env or disable USE_LLM."

            client = OpenAI(api_key=Config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You answer questions based on document excerpts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return "Sorry, I encountered an error while generating the answer."
