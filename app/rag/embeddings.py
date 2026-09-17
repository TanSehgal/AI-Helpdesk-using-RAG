"""Wraps the HuggingFace sentence-transformer embedding model.

A single embedding function instance is shared across the app (both for
building the index and for querying it) so that vectors are always produced
by the same model.
"""
from functools import lru_cache

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.utils.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Returns a cached sentence-transformer embedding function for ChromaDB."""
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
