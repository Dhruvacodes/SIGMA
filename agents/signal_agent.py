"""
SIGMA Signal Agent.
Transforms RawEvents into typed DetectedSignal objects.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from data.ingestion.nse_feed import NSEFeed
from data.technical.indicators import (
    check_rsi_status,
    compute_all_indicators,
    detect_52w_breakout,
    detect_support_resistance,
)
from data.technical.patterns import BulkDealClassifier
from models.events import DetectedSignal, EventType, RawEvent, SignalDirection
from models.state import SigmaState


class SignalAgent:
    """
    Signal Agent for SIGMA.
    Transforms RawEvents into typed DetectedSignal objects.
    """

    def __init__(self):
        self.classifier = BulkDealClassifier()
        self.nse_feed = NSEFeed()

    async def run(self, state: SigmaState) -> dict[str, Any]:
        """
        Process all raw_events in state and produce detected_signals.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with detected_signals.
        """
        signals: list[DetectedSignal] = []
        errors: list[dict] = []

        for event in state.get("raw_events", []):
            try:
                detected = await self._process_event(event)
                if detected:
                    if isinstance(detected, list):
                        signals.extend(detected)
                    else:
                        signals.append(detected)
            except Exception as e:
                errors.append(
                    {
                        "agent": "SignalAgent",
                        "error": str(e),
                        "event_id": event.event_id,
                        "timestamp": datetime.now(),
                    }
                )

        return {"detected_signals": signals, "error_log": errors}

    async def _process_event(
        self, event: RawEvent
    ) -> DetectedSignal | list[DetectedSignal] | None:
        """Route event to appropriate detection method."""

        if event.event_type == EventType.BULK_DEAL:
            return await self._detect_bulk_deal_signal(event)
        elif event.event_type == EventType.TECHNICAL_BREAKOUT:
            return await self._detect_technical_signals(event)
        elif event.event_type == EventType.NEWS_EVENT:
            return self._detect_news_signal(event)
        return None

    async def _detect_technical_signals(self, event: RawEvent) -> list[DetectedSignal]:
        """
        Detect technical signals for a ticker.

        1. Fetch historical OHLCV (252 days min)
        2. Run compute_all_indicators()
        3. Run detect_52w_breakout() → if triggered, create DetectedSignal
        4. Run check_rsi_status() → if strength > 0.3, create DetectedSignal
        """
        signals: list[DetectedSignal] = []

        df = self.nse_feed.fetch_historical_ohlcv(event.ticker, days=300)
        if df.empty or len(df) < 50:
            return []

        df = compute_all_indicators(df)

        # Check breakout
        breakout = detect_52w_breakout(df)
        if breakout:
            signals.append(
                DetectedSignal(
                    signal_id=str(uuid4()),
                    raw_event_id=event.event_id,
                    ticker=event.ticker,
                    signal_type=EventType.TECHNICAL_BREAKOUT,
                    direction=SignalDirection.BULLISH,
                    strength=breakout["strength"],
                    evidence={**breakout, "signal_subtype": "52W_BREAKOUT"},
                    timestamp=datetime.now(),
                    requires_enrichment=True,
                )
            )

        # Check RSI
        rsi_status = check_rsi_status(df)
        if rsi_status["strength"] > 0.3:
            direction = (
                SignalDirection.BEARISH
                if rsi_status["status"] == "overbought"
                else SignalDirection.BULLISH
            )
            signals.append(
                DetectedSignal(
                    signal_id=str(uuid4()),
                    raw_event_id=event.event_id,
                    ticker=event.ticker,
                    signal_type=EventType.RSI_SIGNAL,
                    direction=direction,
                    strength=rsi_status["strength"],
                    evidence=rsi_status,
                    timestamp=datetime.now(),
                    requires_enrichment=True,
                )
            )

        # Check support/resistance proximity
        sr_levels = detect_support_resistance(df)
        if sr_levels["support_levels"] or sr_levels["resistance_levels"]:
            current_price = df["Close"].iloc[-1]

            # Check if near a confirmed support
            for support in sr_levels["support_levels"]:
                if support["strength"] == "confirmed":
                    distance_pct = abs(current_price - support["price"]) / support["price"] * 100
                    if distance_pct < 2:  # Within 2% of support
                        signals.append(
                            DetectedSignal(
                                signal_id=str(uuid4()),
                                raw_event_id=event.event_id,
                                ticker=event.ticker,
                                signal_type=EventType.TECHNICAL_BREAKOUT,
                                direction=SignalDirection.BULLISH,
                                strength=0.5,
                                evidence={
                                    "signal_subtype": "NEAR_SUPPORT",
                                    "support_price": support["price"],
                                    "current_price": current_price,
                                    "touches": support["touches"],
                                },
                                timestamp=datetime.now(),
                                requires_enrichment=True,
                            )
                        )
                        break

        return signals

    async def _detect_bulk_deal_signal(self, event: RawEvent) -> DetectedSignal | None:
        """
        Detect signals from bulk deals.
        """
        payload = event.raw_payload

        # Build deal dict for classifier
        deal = {
            "price_discount_to_market": payload.get(
                "price_vs_prev_close", payload.get("price_discount_to_prev_close_pct", 0)
            ),
            "stake_sold_pct": payload.get("quantity_pct_equity", 1.0),
            "management_commentary_sentiment": payload.get("sentiment"),
        }

        # Stub earnings and pledge data (would be fetched in production)
        earnings_data = {}
        pledge_data = {"pledged_pct": 0}

        # Classify intent
        classification = self.classifier.classify_intent(deal, earnings_data, pledge_data)

        # Determine direction based on classification
        if classification["classification"] == "likely_distress":
            direction = SignalDirection.BEARISH
            strength = classification["distress_probability"]
        elif classification["classification"] == "likely_routine":
            direction = SignalDirection.NEUTRAL
            strength = 0.3
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.4

        # Only create signal if it's promoter-flagged or significant
        if not payload.get("is_promoter_flagged") and strength < 0.5:
            return None

        return DetectedSignal(
            signal_id=str(uuid4()),
            raw_event_id=event.event_id,
            ticker=event.ticker,
            signal_type=EventType.BULK_DEAL,
            direction=direction,
            strength=strength,
            evidence={
                "classification": classification["classification"],
                "distress_probability": classification["distress_probability"],
                "feature_breakdown": classification["feature_breakdown"],
                "is_promoter_flagged": payload.get("is_promoter_flagged", False),
                "client_name": payload.get("clientName", ""),
                "deal_type": payload.get("dealType", ""),
                "quantity_pct_equity": payload.get("quantity_pct_equity", 0),
                "price_discount_to_prev_close_pct": payload.get("price_discount_to_prev_close_pct", 0),
            },
            timestamp=datetime.now(),
            requires_enrichment=True,
        )

    def _detect_news_signal(self, event: RawEvent) -> DetectedSignal | None:
        """
        Detect signals from news events.
        """
        payload = event.raw_payload
        entities = payload.get("entities", {})
        sentiment = payload.get("sentiment", "neutral")
        event_keywords = entities.get("event_keywords", [])

        # Check for significant negative events
        negative_keywords = ["default", "SEBI order", "fraud", "investigation", "penalty"]
        has_negative_event = any(kw in event_keywords for kw in negative_keywords)

        if sentiment == "negative" and has_negative_event:
            return DetectedSignal(
                signal_id=str(uuid4()),
                raw_event_id=event.event_id,
                ticker=event.ticker,
                signal_type=EventType.NEWS_EVENT,
                direction=SignalDirection.BEARISH,
                strength=0.6,
                evidence={
                    "sentiment": sentiment,
                    "event_keywords": event_keywords,
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                },
                timestamp=datetime.now(),
                requires_enrichment=True,
            )

        # Check for significant positive events
        positive_keywords = ["acquisition", "buyback", "merger", "dividend"]
        has_positive_event = any(kw in event_keywords for kw in positive_keywords)

        if sentiment == "positive" and has_positive_event:
            return DetectedSignal(
                signal_id=str(uuid4()),
                raw_event_id=event.event_id,
                ticker=event.ticker,
                signal_type=EventType.NEWS_EVENT,
                direction=SignalDirection.BULLISH,
                strength=0.5,
                evidence={
                    "sentiment": sentiment,
                    "event_keywords": event_keywords,
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                },
                timestamp=datetime.now(),
                requires_enrichment=True,
            )

        # Check for macro events (rate cuts, etc.)
        macro_keywords = ["rate cut", "RBI", "repo rate"]
        has_macro_event = any(kw.lower() in str(event_keywords).lower() for kw in macro_keywords)

        if has_macro_event:
            # Rate cuts are generally positive for NBFCs and banks
            return DetectedSignal(
                signal_id=str(uuid4()),
                raw_event_id=event.event_id,
                ticker=event.ticker,
                signal_type=EventType.NEWS_EVENT,
                direction=SignalDirection.BULLISH if "cut" in str(event_keywords).lower() else SignalDirection.BEARISH,
                strength=0.5,
                evidence={
                    "sentiment": sentiment,
                    "event_keywords": event_keywords,
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                    "is_macro_event": True,
                },
                timestamp=datetime.now(),
                requires_enrichment=True,
            )

        return None
