# src/vector_store.py
"""
Vector store abstraction using FAISS.

Stores L2-normalised embeddings and their Section metadata, supports similarity search, and can persist the index on disk.
Switch between exact (flat) and approximate (HNSW) indexes via environment variables (INDEX_TYPE, HNSW_M, HNSW_EF_CONSTRUCTION).
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import faiss

from config import Config
from logger import get_logger
from pdf_parser import Section

logger = get_logger(__name__)


class VectorStore:
    def __init__(self):
        self.index = None
        self.sections: List[Section] = []
        self.dim = None

    def _to_float32_2d(self, x) -> np.ndarray:
        """
        Normalize input to a 2D float32 numpy array.
        Accepts list[list[float]] or numpy arrays.
        """
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            # single vector -> make it 2D
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Embeddings must be 2D. Got shape={arr.shape}, ndim={arr.ndim}")
        return arr

    def add_vectors(self, embeddings, chunks: List[Section]) -> None:
        emb = self._to_float32_2d(embeddings)

        if self.index is None:
            self.dim = emb.shape[1]
            logger.info("Initialising flat index (exact similarity search).")
            # cosine similarity requires normalized vectors if using inner product
            self.index = faiss.IndexFlatIP(self.dim)

        # Optional: normalize for cosine similarity
        faiss.normalize_L2(emb)

        self.index.add(emb)
        self.sections.extend(chunks)

    def search(self, query_embedding, top_k: int) -> List[Tuple[Section, float]]:
        q = self._to_float32_2d(query_embedding)
        faiss.normalize_L2(q)

        distances, indices = self.index.search(q, top_k)
        results: List[Tuple[Section, float]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            results.append((self.sections[idx], float(score)))
        return results
