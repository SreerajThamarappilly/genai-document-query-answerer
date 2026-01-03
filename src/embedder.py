# src/embedder.py
"""
Embedding layer used by the RAG pipeline.

Supports:
- Local embeddings via sentence-transformers (default for offline / free usage)
- OpenAI embeddings via the OpenAI Python SDK v1+ (openai>=1.0.0)

Key engineering goals:
- Deterministic behavior in tests
- Clear logging + hard failures when config is incomplete
- Single responsibility: return embeddings for a list[str]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from config import Config
from logger import get_logger

logger = get_logger(__name__)

# Local embedding imports (optional)
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

# OpenAI SDK import (optional)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


@dataclass
class Embedder:
    """
    Embedder is a strategy-style wrapper.
    It selects OpenAI embeddings OR local SentenceTransformer embeddings, based on Config.USE_OPENAI.
    """

    def __post_init__(self) -> None:
        self.use_openai = Config.USE_OPENAI

        if self.use_openai:
            if OpenAI is None:
                raise ImportError("openai package is required when USE_OPENAI=True")
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set but USE_OPENAI=True")

            # OpenAI v1+ client
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.model_name = Config.OPENAI_EMBEDDING_MODEL
            logger.info(f"Embedder using OpenAI model '{self.model_name}'.")
            self.local_model = None
        else:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required when USE_OPENAI=False "
                    "(install sentence-transformers + compatible huggingface_hub)"
                )
            self.model_name = Config.LOCAL_EMBEDDING_MODEL
            logger.info(f"Loading local sentence transformer model '{self.model_name}'…")
            self.local_model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded model '{self.model_name}' successfully.")
            self.client = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of strings to embedding vectors.

        Time complexity:
            - OpenAI: O(n) API calls worth of work (batched in 1 request)
            - Local: O(n * sequence_length) transformer forward passes

        Returns:
            List of vectors (List[List[float]]) aligned with input texts.
        """
        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            return []

        if self.use_openai:
            # OpenAI v1+ embeddings endpoint
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=cleaned,
            )
            # resp.data is a list in the same order as input
            return [item.embedding for item in resp.data]

        # Local embeddings
        assert self.local_model is not None
        vectors = self.local_model.encode(
            cleaned,
            normalize_embeddings=True,  # helps cosine similarity in FAISS
            show_progress_bar=False,
        )
        # sentence-transformers may return numpy array; convert to Python lists
        return [v.tolist() for v in vectors]
