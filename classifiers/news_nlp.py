"""
SIGMA News NLP module.
Re-exports entity extraction from data/ingestion/et_news.py for convenience.
"""

from data.ingestion.et_news import (
    EVENT_KEYWORDS,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    ETNewsParser,
)

__all__ = ["ETNewsParser", "POSITIVE_KEYWORDS", "NEGATIVE_KEYWORDS", "EVENT_KEYWORDS"]
