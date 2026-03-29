"""
SIGMA SEBI Filings Scraper.
Fetches bulk deals, block deals, and insider trades from NSE.
"""

import asyncio
from datetime import date, datetime
from typing import Any

import aiohttp

from data.ingestion.nse_feed import NSE_HEADERS, NSEFeed
from models.events import EventType, RawEvent

# Rate limiting
_last_request_time = 0.0
_rate_limit_seconds = 0.5

# Promoter patterns for filtering
PROMOTER_PATTERNS = [
    "promoter",
    "director",
    "chairman",
    "managing director",
    "cmd",
    "founder",
    "ceo",
    "cfo",
    "coo",
]


class SEBIFilings:
    """
    SEBI filings scraper for bulk deals, block deals, and insider trades.
    """

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._nse_feed = NSEFeed()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=NSE_HEADERS)
        return self._session

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        global _last_request_time
        now = asyncio.get_event_loop().time()
        elapsed = now - _last_request_time
        if elapsed < _rate_limit_seconds:
            await asyncio.sleep(_rate_limit_seconds - elapsed)
        _last_request_time = asyncio.get_event_loop().time()

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
        await self._nse_feed.close()

    async def fetch_bulk_deals(self, target_date: date) -> list[RawEvent]:
        """
        Fetch bulk deals from NSE API.

        Args:
            target_date: Date to fetch bulk deals for.

        Returns:
            List of RawEvent objects for each bulk deal.
        """
        date_str = target_date.strftime("%d-%m-%Y")
        url = f"https://www.nseindia.com/api/historical/bulk-deals?from={date_str}&to={date_str}"

        events = []

        try:
            await self._rate_limit()
            session = await self._get_session()

            # First request to get cookies
            async with session.get("https://www.nseindia.com", timeout=aiohttp.ClientTimeout(total=10)):
                pass

            await self._rate_limit()

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                deals = data.get("data", [])

                for deal in deals:
                    symbol = deal.get("symbol", "")
                    if not symbol:
                        continue

                    client_name = deal.get("clientName", "")
                    deal_type = deal.get("dealType", deal.get("buySell", ""))
                    quantity = deal.get("quantity", 0)
                    price = deal.get("price", deal.get("avgPrice", 0))
                    remarks = deal.get("remarks", "")

                    # Determine parse confidence
                    if client_name and client_name.lower() not in ["others", "other", "client"]:
                        parse_confidence = 1.0
                    else:
                        parse_confidence = 0.7

                    # Try to get price vs prev close
                    price_vs_prev_close = None
                    try:
                        quote = await self._nse_feed.get_live_quote(symbol)
                        if quote and quote.get("lastPrice"):
                            price_vs_prev_close = ((float(price) - quote["lastPrice"]) / quote["lastPrice"]) * 100
                    except Exception:
                        pass

                    raw_payload = {
                        "symbol": symbol,
                        "clientName": client_name,
                        "dealType": deal_type,
                        "quantity": quantity,
                        "price": price,
                        "remarks": remarks,
                        "price_vs_prev_close": price_vs_prev_close,
                    }

                    events.append(
                        RawEvent(
                            ticker=symbol,
                            exchange="NSE",
                            event_type=EventType.BULK_DEAL,
                            timestamp=datetime.combine(target_date, datetime.min.time()),
                            raw_payload=raw_payload,
                            source="NSE_BULK_DEAL_API",
                            parse_confidence=parse_confidence,
                        )
                    )

        except Exception as e:
            print(f"Error fetching bulk deals: {e}")

        return events

    def extract_promoter_deals(self, events: list[RawEvent]) -> list[RawEvent]:
        """
        Filter bulk deals for promoter-related transactions.

        Args:
            events: List of RawEvent objects to filter.

        Returns:
            Filtered list with promoter-flagged deals.
        """
        promoter_deals = []

        for event in events:
            if event.event_type != EventType.BULK_DEAL:
                continue

            client_name = event.raw_payload.get("clientName", "").lower()

            is_promoter = False

            # Check for promoter patterns
            for pattern in PROMOTER_PATTERNS:
                if pattern in client_name:
                    is_promoter = True
                    break

            # Check for corporate promoter patterns
            if not is_promoter:
                if client_name.endswith("ltd") or client_name.endswith("limited"):
                    is_promoter = True
                elif "private limited" in client_name:
                    is_promoter = True
                elif "pvt ltd" in client_name or "pvt. ltd" in client_name:
                    is_promoter = True

            if is_promoter:
                # Create new event with promoter flag
                updated_payload = dict(event.raw_payload)
                updated_payload["is_promoter_flagged"] = True
                promoter_deals.append(
                    RawEvent(
                        event_id=event.event_id,
                        ticker=event.ticker,
                        exchange=event.exchange,
                        event_type=event.event_type,
                        timestamp=event.timestamp,
                        raw_payload=updated_payload,
                        source=event.source,
                        parse_confidence=event.parse_confidence,
                    )
                )

        return promoter_deals

    async def fetch_insider_trades(self, ticker: str) -> list[RawEvent]:
        """
        Fetch insider trades for a specific ticker.

        Args:
            ticker: NSE ticker symbol.

        Returns:
            List of RawEvent objects for each insider trade.
        """
        url = f"https://www.nseindia.com/api/corporates-pit?symbol={ticker}&issuer=&from=&to=&type=insider"

        events = []

        try:
            await self._rate_limit()
            session = await self._get_session()

            # First request to get cookies
            async with session.get("https://www.nseindia.com", timeout=aiohttp.ClientTimeout(total=10)):
                pass

            await self._rate_limit()

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                trades = data.get("data", [])

                for trade in trades:
                    acquirer_name = trade.get("acqName", "")
                    no_of_securities = trade.get("noOfSecurities", 0)
                    type_of_sec = trade.get("secType", "")
                    transaction_type = trade.get("tdpTransactionType", "")
                    before_acq = float(trade.get("befAcqSharesNo", 0) or 0)
                    after_acq = float(trade.get("afterAcqSharesNo", 0) or 0)

                    # Compute holding change percentage
                    holding_change_pct = 0.0
                    if before_acq > 0:
                        holding_change_pct = ((after_acq - before_acq) / before_acq) * 100

                    raw_payload = {
                        "acquirerName": acquirer_name,
                        "noOfSecurities": no_of_securities,
                        "typeOfSec": type_of_sec,
                        "transactionType": transaction_type,
                        "beforeAcq": before_acq,
                        "afterAcq": after_acq,
                        "holding_change_pct": holding_change_pct,
                    }

                    # Parse timestamp
                    timestamp = datetime.now()
                    if trade.get("td"):
                        try:
                            timestamp = datetime.strptime(trade["td"], "%d-%b-%Y")
                        except ValueError:
                            pass

                    events.append(
                        RawEvent(
                            ticker=ticker,
                            exchange="NSE",
                            event_type=EventType.BULK_DEAL,  # Insider trades are similar to bulk deals
                            timestamp=timestamp,
                            raw_payload=raw_payload,
                            source="NSE_INSIDER_TRADE_API",
                            parse_confidence=1.0,
                        )
                    )

        except Exception as e:
            print(f"Error fetching insider trades for {ticker}: {e}")

        return events
