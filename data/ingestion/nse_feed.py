"""
SIGMA NSE Data Feed.
Fetches Bhavcopy, historical OHLCV, and live quotes from NSE.
"""

import asyncio
import io
from datetime import date, datetime, timedelta
from typing import Any

import aiohttp
import pandas as pd

from models.events import EventType, RawEvent

# Realistic browser headers to avoid blocking
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
}

# In-memory cache for OHLCV data
_ohlcv_cache: dict[tuple[str, int], pd.DataFrame] = {}

# Rate limiting
_last_request_time = 0.0
_rate_limit_seconds = 0.5


class NSEFeed:
    """
    NSE data feed for SIGMA.
    Provides Bhavcopy, historical OHLCV, and live quote data.
    """

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

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

    async def fetch_bhavcopy(self, target_date: date) -> list[RawEvent]:
        """
        Download NSE EOD Bhavcopy CSV.

        Args:
            target_date: Date to fetch Bhavcopy for.

        Returns:
            List of RawEvent objects for each equity row.
        """
        # Handle weekends
        target_date = self._get_last_trading_day(target_date)
        date_str = target_date.strftime("%d%m%Y")

        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"

        events = []

        try:
            await self._rate_limit()
            session = await self._get_session()

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    # Try alternate URL format
                    url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response2:
                        if response2.status != 200:
                            return []
                        content = await response2.text()
                else:
                    content = await response.text()

            # Parse CSV
            df = pd.read_csv(io.StringIO(content))

            # Filter for equity series only
            if "SERIES" in df.columns:
                df = df[df["SERIES"] == "EQ"]
            elif " SERIES" in df.columns:
                df = df[df[" SERIES"].str.strip() == "EQ"]

            # Create RawEvent for each row
            for _, row in df.iterrows():
                # Handle column name variations
                symbol = row.get("SYMBOL", row.get(" SYMBOL", "")).strip()
                if not symbol:
                    continue

                # Extract OHLCV data
                try:
                    raw_payload = {
                        "symbol": symbol,
                        "series": "EQ",
                        "open": float(row.get("OPEN", row.get(" OPEN_PRICE", 0))),
                        "high": float(row.get("HIGH", row.get(" HIGH_PRICE", 0))),
                        "low": float(row.get("LOW", row.get(" LOW_PRICE", 0))),
                        "close": float(row.get("CLOSE", row.get(" CLOSE_PRICE", 0))),
                        "volume": int(row.get("TOTTRDQTY", row.get(" TTL_TRD_QNTY", 0))),
                        "value": float(row.get("TOTTRDVAL", row.get(" TURNOVER_LACS", 0))),
                    }
                    parse_confidence = 1.0
                except (ValueError, TypeError):
                    raw_payload = {"symbol": symbol, "error": "parse_error"}
                    parse_confidence = 0.5

                events.append(
                    RawEvent(
                        ticker=symbol,
                        exchange="NSE",
                        event_type=EventType.TECHNICAL_BREAKOUT,  # Will be classified by Signal Agent
                        timestamp=datetime.combine(target_date, datetime.min.time()),
                        raw_payload=raw_payload,
                        source="NSE_BHAVCOPY",
                        parse_confidence=parse_confidence,
                    )
                )

        except Exception as e:
            # Log error but don't fail
            print(f"Error fetching Bhavcopy: {e}")

        return events

    def fetch_historical_ohlcv(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical OHLCV data using yfinance.

        Args:
            ticker: NSE ticker symbol.
            days: Number of days of history.

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume columns.
        """
        cache_key = (ticker, days)
        if cache_key in _ohlcv_cache:
            return _ohlcv_cache[cache_key].copy()

        try:
            import yfinance as yf

            # Download data
            df = yf.download(f"{ticker}.NS", period=f"{days}d", interval="1d", progress=False)

            if df.empty:
                # Try without .NS suffix
                df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)

            if df.empty:
                return pd.DataFrame()

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Ensure standard column names
            df = df.reset_index()
            df = df.rename(
                columns={
                    "Datetime": "Date",
                    "Adj Close": "Adj_Close",
                }
            )

            # Ensure required columns exist
            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns:
                    if col == "Date" and "index" in df.columns:
                        df = df.rename(columns={"index": "Date"})
                    elif col == "Volume" and "Vol" in df.columns:
                        df = df.rename(columns={"Vol": "Volume"})

            # Cache and return
            _ohlcv_cache[cache_key] = df
            return df.copy()

        except Exception as e:
            print(f"Error fetching OHLCV for {ticker}: {e}")
            return pd.DataFrame()

    async def get_live_quote(self, ticker: str) -> dict[str, Any]:
        """
        Get live quote from NSE API.

        Args:
            ticker: NSE ticker symbol.

        Returns:
            Dict with lastPrice, change, pChange, totalTradedVolume.
        """
        url = f"https://www.nseindia.com/api/quote-equity?symbol={ticker}"

        try:
            await self._rate_limit()
            session = await self._get_session()

            # First request to get cookies
            async with session.get("https://www.nseindia.com", timeout=aiohttp.ClientTimeout(total=10)):
                pass

            await self._rate_limit()

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    return {}

                data = await response.json()
                price_info = data.get("priceInfo", {})

                return {
                    "lastPrice": price_info.get("lastPrice", 0),
                    "change": price_info.get("change", 0),
                    "pChange": price_info.get("pChange", 0),
                    "totalTradedVolume": data.get("securityWiseDP", {}).get("quantityTraded", 0),
                }

        except Exception as e:
            print(f"Error fetching live quote for {ticker}: {e}")
            return {}

    @staticmethod
    def _get_last_trading_day(d: date) -> date:
        """Return the last trading day (skip weekends)."""
        while d.weekday() > 4:  # Saturday=5, Sunday=6
            d -= timedelta(days=1)
        return d

    @staticmethod
    def get_last_trading_day() -> date:
        """Return today if weekday, else last Friday."""
        return NSEFeed._get_last_trading_day(date.today())
