"""
SIGMA Retriever.
Composite retriever logic for RAG enrichment.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from config import settings
from rag.vector_store import SigmaVectorStore

if TYPE_CHECKING:
    from models.events import DetectedSignal


# Hardcoded sector impact data for fast lookup
SECTOR_IMPACT_DATA = {
    "RATE_CUT": {
        "NBFC": {"avg_impact_pct": 8.0, "recovery_timeline": "N/A", "confidence": "high"},
        "Banking": {"avg_impact_pct": 4.0, "recovery_timeline": "N/A", "confidence": "high"},
        "IT": {"avg_impact_pct": -1.2, "recovery_timeline": "N/A", "confidence": "medium"},
        "FMCG": {"avg_impact_pct": 0.0, "recovery_timeline": "N/A", "confidence": "medium"},
        "Pharma": {"avg_impact_pct": 2.0, "recovery_timeline": "N/A", "confidence": "medium"},
    },
    "RATE_HIKE": {
        "NBFC": {"avg_impact_pct": -7.0, "recovery_timeline": "3-6 months", "confidence": "high"},
        "Banking": {"avg_impact_pct": -3.0, "recovery_timeline": "2-3 months", "confidence": "high"},
        "IT": {"avg_impact_pct": 0.5, "recovery_timeline": "N/A", "confidence": "low"},
        "FMCG": {"avg_impact_pct": 0.0, "recovery_timeline": "N/A", "confidence": "low"},
    },
    "DRUG_PRICE_ORDER": {
        "Pharma": {"avg_impact_pct": -15.0, "recovery_timeline": "6 months", "confidence": "high"},
    },
    "SEBI_ORDER": {
        "All": {"avg_impact_pct": -8.0, "recovery_timeline": "variable", "confidence": "medium"},
    },
}


class SigmaRetriever:
    """
    Retriever for SIGMA's RAG pipeline.
    Retrieves historical context, management sentiment, and sector context.
    """

    def __init__(self, store: SigmaVectorStore):
        self.store = store

    def get_historical_context(self, signal: "DetectedSignal") -> dict[str, Any]:
        """
        Build a targeted query from signal metadata and retrieve relevant historical outcomes.

        Args:
            signal: The detected signal to find context for.

        Returns:
            Dict with historical context data.
        """
        # Build query based on signal type
        if signal.signal_type.value == "TECHNICAL_BREAKOUT":
            pattern = signal.evidence.get("signal_subtype", "52W breakout")
            market_cap = signal.evidence.get("market_cap", "large-cap")
            rsi = signal.evidence.get("rsi", "neutral")
            volume_ratio = signal.evidence.get("volume_ratio", "1x")

            query = (
                f"{signal.direction.value} {pattern} on {market_cap} stock. "
                f"RSI at entry: {rsi}. Volume ratio: {volume_ratio}"
            )
        elif signal.signal_type.value == "BULK_DEAL":
            discount = signal.evidence.get("discount", signal.evidence.get("price_discount_to_prev_close_pct", "market price"))
            stake_pct = signal.evidence.get("stake_pct", signal.evidence.get("quantity_pct_equity", "?"))

            query = f"Promoter bulk sale at {discount}% discount. Stake sold: {stake_pct}%"
        elif signal.signal_type.value == "RSI_SIGNAL":
            rsi = signal.evidence.get("rsi", 50)
            status = signal.evidence.get("status", "neutral")
            query = f"RSI {status} signal. RSI value: {rsi}"
        else:
            query = f"{signal.signal_type.value} signal on {signal.ticker}"

        # Query the vector store
        try:
            # Build where filter for signal type
            where_filter = None
            if signal.signal_type.value in ["TECHNICAL_BREAKOUT", "BULK_DEAL", "RSI_SIGNAL"]:
                where_filter = {"signal_type": signal.signal_type.value}

            results = self.store.query(
                "historical_patterns",
                query,
                n_results=settings.RAG_TOP_K,
                where=where_filter,
            )
        except Exception:
            results = []

        if not results:
            return {
                "query_used": query,
                "base_rate_positive": None,
                "median_30d_return": None,
                "worst_quartile_return": None,
                "sample_size": 0,
                "raw_passages": [],
            }

        # Post-process results
        outcomes = []
        positive_count = 0
        passages = []

        for r in results:
            metadata = r.get("metadata", {})
            passages.append(r.get("text", ""))

            outcome_30d = metadata.get("outcome_30d")
            outcome_positive = metadata.get("outcome_positive")

            if outcome_30d is not None:
                try:
                    outcomes.append(float(outcome_30d))
                except (ValueError, TypeError):
                    pass

            if outcome_positive is not None:
                if str(outcome_positive).lower() == "true" or outcome_positive is True:
                    positive_count += 1

        sample_size = len(results)
        base_rate_positive = positive_count / sample_size if sample_size > 0 else None

        # Compute statistics
        if outcomes:
            median_return = float(np.median(outcomes))
            worst_quartile = float(np.percentile(outcomes, 25))
        else:
            median_return = None
            worst_quartile = None

        return {
            "query_used": query,
            "base_rate_positive": base_rate_positive,
            "median_30d_return": median_return,
            "worst_quartile_return": worst_quartile,
            "sample_size": sample_size,
            "raw_passages": passages,
        }

    def get_management_sentiment(self, ticker: str) -> dict[str, Any]:
        """
        Retrieve recent management commentary for ticker.

        Args:
            ticker: Stock ticker to get sentiment for.

        Returns:
            Dict with sentiment data.
        """
        query = f"management commentary guidance outlook for {ticker}"

        try:
            # Try with ticker filter first
            results = self.store.query(
                "management_commentary",
                query,
                n_results=3,
                where={"ticker": ticker},
            )

            # If no results with filter, try without
            if not results:
                results = self.store.query(
                    "management_commentary",
                    query,
                    n_results=3,
                )
        except Exception:
            results = []

        if not results:
            return {
                "ticker": ticker,
                "sentiment": "neutral",
                "passages": [],
                "data_available": False,
            }

        # Aggregate sentiment via majority vote
        sentiments = []
        passages = []

        for r in results:
            metadata = r.get("metadata", {})
            passages.append(r.get("text", ""))

            sentiment = metadata.get("sentiment")
            if sentiment in ["positive", "negative", "neutral"]:
                sentiments.append(sentiment)

        # Majority vote
        if sentiments:
            from collections import Counter
            sentiment_counts = Counter(sentiments)
            majority_sentiment = sentiment_counts.most_common(1)[0][0]
        else:
            majority_sentiment = "neutral"

        return {
            "ticker": ticker,
            "sentiment": majority_sentiment,
            "passages": passages,
            "data_available": True,
        }

    def get_sector_context(self, sector: str, event_type: str) -> dict[str, Any]:
        """
        Returns historical sector-level impact data.

        Args:
            sector: Sector name (e.g., "IT", "Banking", "NBFC").
            event_type: Type of event (e.g., "RATE_CUT", "DRUG_PRICE_ORDER").

        Returns:
            Dict with sector impact data.
        """
        # Use hardcoded lookup for speed
        event_data = SECTOR_IMPACT_DATA.get(event_type, {})
        sector_data = event_data.get(sector, event_data.get("All", {}))

        if sector_data:
            return {
                "sector": sector,
                "event_type": event_type,
                "avg_impact_pct": sector_data.get("avg_impact_pct", 0.0),
                "recovery_timeline": sector_data.get("recovery_timeline", "unknown"),
                "confidence": sector_data.get("confidence", "low"),
            }

        # Try to get from vector store as fallback
        try:
            query = f"{sector} sector impact from {event_type}"
            results = self.store.query("sector_context", query, n_results=1)

            if results:
                metadata = results[0].get("metadata", {})
                return {
                    "sector": sector,
                    "event_type": event_type,
                    "avg_impact_pct": metadata.get("avg_impact_pct", 0.0),
                    "recovery_timeline": metadata.get("recovery_timeline", "unknown"),
                    "confidence": "medium",
                }
        except Exception:
            pass

        return {
            "sector": sector,
            "event_type": event_type,
            "avg_impact_pct": 0.0,
            "recovery_timeline": "unknown",
            "confidence": "low",
        }
