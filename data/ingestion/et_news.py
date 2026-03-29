"""
SIGMA ET News Parser.
Parses Economic Times RSS feed and extracts entities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser

from models.events import EventType, RawEvent

# Load ticker map
_ticker_map: dict[str, str] = {}
_ticker_map_path = Path(__file__).parent / "ticker_map.json"
if _ticker_map_path.exists():
    with open(_ticker_map_path, "r") as f:
        _ticker_map = json.load(f)

# Sentiment keywords
POSITIVE_KEYWORDS = [
    "beats", "upgrade", "profit", "growth", "acquisition", "buyback",
    "dividend", "record", "surge", "rally", "gains", "bullish", "outperform",
    "strong", "robust", "accelerate", "expand", "improve",
]

NEGATIVE_KEYWORDS = [
    "miss", "downgrade", "loss", "default", "penalty", "SEBI order",
    "fraud", "resign", "crash", "plunge", "bearish", "weak", "decline",
    "slump", "warning", "concern", "risk", "investigation",
]

EVENT_KEYWORDS = [
    "rate cut", "merger", "acquisition", "default", "SEBI order",
    "earnings", "quarterly results", "RBI", "repo rate", "inflation",
    "GDP", "FII", "DII", "buyback", "dividend", "split", "bonus",
    "DPCO", "drug price", "pricing cap",
]


class ETNewsParser:
    """
    Economic Times news parser for SIGMA.
    Parses RSS feeds and extracts relevant entities and sentiment.
    """

    def __init__(self):
        self._nlp = None

    def _get_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                # Fall back to None if spaCy not available
                self._nlp = False
        return self._nlp if self._nlp else None

    async def fetch_rss_events(
        self, feed_url: str = "https://economictimes.indiatimes.com/markets/rss.cms"
    ) -> list[RawEvent]:
        """
        Fetch and parse ET RSS feed.

        Args:
            feed_url: URL of the RSS feed.

        Returns:
            List of RawEvent objects for each news item.
        """
        events = []

        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link = entry.get("link", "")
                published = entry.get("published", "")

                # Parse published date
                timestamp = datetime.now()
                if published:
                    try:
                        timestamp = datetime(*entry.published_parsed[:6])
                    except (TypeError, AttributeError):
                        pass

                # Extract entities from title + summary
                combined_text = f"{title} {summary}"
                entities = self.extract_entities(combined_text)

                # Classify sentiment
                sentiment = self.classify_news_sentiment(combined_text)

                # Determine primary ticker (if any)
                tickers = entities.get("tickers", [])
                primary_ticker = tickers[0] if tickers else "MARKET"

                raw_payload = {
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "published": published,
                    "entities": entities,
                    "sentiment": sentiment,
                }

                events.append(
                    RawEvent(
                        ticker=primary_ticker,
                        exchange="NSE",
                        event_type=EventType.NEWS_EVENT,
                        timestamp=timestamp,
                        raw_payload=raw_payload,
                        source="ET_NEWS_RSS",
                        parse_confidence=1.0 if tickers else 0.7,
                    )
                )

        except Exception as e:
            print(f"Error fetching RSS feed: {e}")

        return events

    def extract_entities(self, text: str) -> dict[str, Any]:
        """
        Extract entities from text using spaCy and keyword matching.

        Args:
            text: Text to extract entities from.

        Returns:
            Dict with tickers, money_figures, event_keywords, sentiment_keywords.
        """
        tickers = []
        money_figures = []
        percent_figures = []
        event_keywords_found = []
        sentiment_keywords_found = []

        # Try spaCy NER
        nlp = self._get_nlp()
        if nlp:
            doc = nlp(text)

            for ent in doc.ents:
                if ent.label_ == "ORG":
                    # Try to map organization to ticker
                    org_name = ent.text.strip()
                    if org_name in _ticker_map:
                        tickers.append(_ticker_map[org_name])
                    else:
                        # Try partial match
                        for name, ticker in _ticker_map.items():
                            if name.lower() in org_name.lower() or org_name.lower() in name.lower():
                                tickers.append(ticker)
                                break

                elif ent.label_ == "MONEY":
                    money_figures.append(ent.text)

                elif ent.label_ == "PERCENT":
                    percent_figures.append(ent.text)

        # Fallback: Direct keyword matching for tickers
        if not tickers:
            text_lower = text.lower()
            for name, ticker in _ticker_map.items():
                if name.lower() in text_lower:
                    tickers.append(ticker)

        # Deduplicate tickers
        tickers = list(dict.fromkeys(tickers))

        # Match event keywords
        text_lower = text.lower()
        for keyword in EVENT_KEYWORDS:
            if keyword.lower() in text_lower:
                event_keywords_found.append(keyword)

        # Match sentiment keywords
        for keyword in POSITIVE_KEYWORDS:
            if keyword.lower() in text_lower:
                sentiment_keywords_found.append(f"+{keyword}")

        for keyword in NEGATIVE_KEYWORDS:
            if keyword.lower() in text_lower:
                sentiment_keywords_found.append(f"-{keyword}")

        return {
            "tickers": tickers,
            "money_figures": money_figures,
            "percent_figures": percent_figures,
            "event_keywords": event_keywords_found,
            "sentiment_keywords": sentiment_keywords_found,
        }

    def classify_news_sentiment(self, text: str) -> str:
        """
        Classify news sentiment using lexicon-based approach.

        Args:
            text: Text to classify.

        Returns:
            "positive", "negative", or "neutral".
        """
        text_lower = text.lower()

        positive_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw.lower() in text_lower)
        negative_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw.lower() in text_lower)

        score = positive_hits - negative_hits

        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"
