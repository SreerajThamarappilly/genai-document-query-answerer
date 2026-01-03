# src/config.py
"""
Central configuration module.

Sets defaults for embedding models, chunk sizes, retrieval settings, LLM usage, and index persistence.  
Values can be overridden by environment variables.
"""

from __future__ import annotations
from dotenv import load_dotenv
from pathlib import Path
import os

# Loads .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

class Config:
    # Embedding configuration
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "False").lower() == "true"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "16"))

    # LLM configuration
    USE_LLM: bool = os.getenv("USE_LLM", "True").lower() == "true"
    USE_LANGCHAIN: bool = os.getenv("USE_LANGCHAIN", "False").lower() == "true"
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))

    # Persistence
    PERSIST_INDEX: bool = os.getenv("PERSIST_INDEX", "False").lower() == "true"
    INDEX_DIR: str = os.getenv("INDEX_DIR", "./indices")
