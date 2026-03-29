"""
SIGMA Embedding Pipeline.
Uses OpenAI's text-embedding-3-small for embeddings.
"""

import os
from functools import lru_cache
from typing import Any

from config import settings

# Module-level client cache
_embedding_client: Any = None


def get_embedding_client():
    """
    Get or create the OpenAI client for embeddings.

    Returns:
        OpenAI client instance.
    """
    global _embedding_client
    if _embedding_client is None:
        try:
            from openai import OpenAI

            api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            _embedding_client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    return _embedding_client


@lru_cache(maxsize=1000)
def embed_text(text: str) -> tuple[float, ...]:
    """
    Embed a single text string.

    Args:
        text: Text to embed (will be truncated to 8000 chars).

    Returns:
        Embedding vector as tuple (for hashability in cache).
    """
    # Truncate to 8000 chars for model safety
    text = text[:8000]

    client = get_embedding_client()
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
    )
    return tuple(response.data[0].embedding)


def embed_text_list(text: str) -> list[float]:
    """
    Embed a single text string and return as list.

    Args:
        text: Text to embed.

    Returns:
        Embedding vector as list[float].
    """
    return list(embed_text(text))


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Batch embed multiple texts at once.

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    client = get_embedding_client()
    embeddings = []

    # Process in chunks of 100 (OpenAI limit)
    chunk_size = 100
    for i in range(0, len(texts), chunk_size):
        chunk = [t[:8000] for t in texts[i : i + chunk_size]]  # Truncate each text
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=chunk,
        )
        for item in response.data:
            embeddings.append(item.embedding)

    return embeddings
