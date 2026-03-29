"""
SIGMA Data Agent.
Entry point for data collection, deduplication, and structuring.
"""

import hashlib
from datetime import date, datetime, timedelta
from typing import Any

from audit.logger import audit_logger
from data.ingestion.et_news import ETNewsParser
from data.ingestion.nse_feed import NSEFeed
from data.ingestion.sebi_filings import SEBIFilings
from models.events import RawEvent
from models.state import SigmaState

# Fallback dedup store (in-memory)
_dedup_store: dict[str, datetime] = {}
_dedup_ttl_hours = 4


class DataAgent:
    """
    Data Agent for SIGMA.
    Collects, deduplicates, and structures all incoming data.
    """

    def __init__(self):
        self.nse_feed = NSEFeed()
        self.sebi_filings = SEBIFilings()
        self.et_news = ETNewsParser()

        # Try to use fakeredis, fallback to dict
        self._redis = None
        try:
            import fakeredis
            self._redis = fakeredis.FakeStrictRedis()
        except ImportError:
            pass

    async def run(self, state: SigmaState) -> dict[str, Any]:
        """
        LangGraph node function. Returns partial state update.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with raw_events and audit_trail.
        """
        all_events: list[RawEvent] = []

        try:
            # 1. Fetch today's bulk deals
            trading_day = self.get_last_trading_day()
            bulk_events = await self.sebi_filings.fetch_bulk_deals(trading_day)
            promoter_events = self.sebi_filings.extract_promoter_deals(bulk_events)
            all_events.extend(promoter_events)

        except Exception as e:
            print(f"Error fetching bulk deals: {e}")

        try:
            # 2. Fetch recent ET news
            news_events = await self.et_news.fetch_rss_events()
            all_events.extend(news_events)

        except Exception as e:
            print(f"Error fetching news: {e}")

        try:
            # 3. Fetch Bhavcopy (today or last trading day)
            trading_day = self.get_last_trading_day()
            bhavcopy_events = await self.nse_feed.fetch_bhavcopy(trading_day)
            # Cap bhavcopy for demo speed
            all_events.extend(bhavcopy_events[:100])

        except Exception as e:
            print(f"Error fetching bhavcopy: {e}")

        # 4. Deduplicate using event_id hash
        unique_events = self._deduplicate(all_events)

        # 5. Write to audit log
        audit_logger.log_agent_run(
            "DataAgent",
            input_count=len(all_events),
            output_count=len(unique_events),
        )

        return {
            "raw_events": unique_events,
            "audit_trail": [
                {
                    "agent": "DataAgent",
                    "event_count": len(unique_events),
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    def _deduplicate(self, events: list[RawEvent]) -> list[RawEvent]:
        """
        Deduplicate events using SHA-256 hash.

        Args:
            events: List of events to deduplicate.

        Returns:
            Deduplicated list of events.
        """
        seen: set[str] = set()
        unique: list[RawEvent] = []

        for e in events:
            # Create idempotency key
            key_data = f"{e.ticker}{e.event_type}{e.timestamp.strftime('%Y%m%d%H')}"
            key = hashlib.sha256(key_data.encode()).hexdigest()

            # Check in Redis or fallback store
            if self._redis:
                if self._redis.get(key):
                    continue
                self._redis.setex(key, _dedup_ttl_hours * 3600, "1")
            else:
                if key in _dedup_store:
                    # Check TTL
                    if datetime.now() - _dedup_store[key] < timedelta(hours=_dedup_ttl_hours):
                        continue
                _dedup_store[key] = datetime.now()

            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    @staticmethod
    def get_last_trading_day() -> date:
        """Return today if weekday, else last Friday."""
        d = date.today()
        while d.weekday() > 4:  # Saturday=5, Sunday=6
            d -= timedelta(days=1)
        return d

    async def close(self) -> None:
        """Clean up resources."""
        await self.nse_feed.close()
        await self.sebi_filings.close()
