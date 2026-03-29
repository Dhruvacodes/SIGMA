"""
SIGMA Context Agent.
Enriches signals using RAG retrieval.
"""

from datetime import datetime
from typing import Any

from models.events import DetectedSignal, EnrichedSignal, SignalDirection
from models.state import SigmaState
from rag.knowledge_base import KnowledgeBaseSeeder
from rag.retriever import SigmaRetriever
from rag.vector_store import SigmaVectorStore

# Sector map for tickers
SECTOR_MAP = {
    "INFY": "IT",
    "TCS": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "KOTAKBANK": "Banking",
    "AXISBANK": "Banking",
    "INDUSINDBK": "Banking",
    "RELIANCE": "Oil & Gas",
    "ONGC": "Oil & Gas",
    "GAIL": "Oil & Gas",
    "SUNPHARMA": "Pharma",
    "DRREDDY": "Pharma",
    "CIPLA": "Pharma",
    "BAJFINANCE": "NBFC",
    "BAJAJFINSV": "NBFC",
    "MUTHOOTFIN": "NBFC",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "DABUR": "FMCG",
    "BRITANNIA": "FMCG",
    "ASIANPAINT": "Consumer",
    "TITAN": "Consumer",
    "PIDILITIND": "Consumer",
    "TATAMOTORS": "Auto",
    "MARUTI": "Auto",
    "HEROMOTOCO": "Auto",
    "EICHERMOT": "Auto",
    "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",
    "HINDALCO": "Metals",
    "COALINDIA": "Mining",
    "NTPC": "Power",
    "POWERGRID": "Power",
    "LT": "Infrastructure",
    "ULTRACEMCO": "Cement",
    "GRASIM": "Cement",
    "ADANIPORTS": "Infrastructure",
    "ADANIENT": "Conglomerate",
    "BHARTIARTL": "Telecom",
    # Synthetic tickers for demo
    "LARGECAP_IT": "IT",
    "LARGECAP_IT_1": "IT",
    "MIDCAP_FMCG": "FMCG",
    "MIDCAP_FMCG_1": "FMCG",
    "PHARMA_HOLDING": "Pharma",
    "NBFC_1": "NBFC",
    "NBFC_2": "NBFC",
}


class ContextAgent:
    """
    Context Agent for SIGMA.
    Enriches DetectedSignals with historical context and RAG data.
    """

    def __init__(self):
        self.store = SigmaVectorStore()
        self.retriever = SigmaRetriever(self.store)
        self._seeded = False

    def _ensure_seeded(self) -> None:
        """Ensure knowledge base is seeded."""
        if not self._seeded:
            if self.store.is_empty("historical_patterns"):
                seeder = KnowledgeBaseSeeder(self.store)
                seeder.seed_all()
            self._seeded = True

    async def run(self, state: SigmaState) -> dict[str, Any]:
        """
        Enrich each DetectedSignal into an EnrichedSignal.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with enriched_signals.
        """
        self._ensure_seeded()

        enriched: list[EnrichedSignal] = []
        errors: list[dict] = []

        for signal in state.get("detected_signals", []):
            try:
                enriched_signal = await self._enrich_signal(signal)
                enriched.append(enriched_signal)
            except Exception as e:
                # On enrichment failure, create minimal EnrichedSignal
                enriched.append(
                    EnrichedSignal(
                        signal=signal,
                        historical_base_rate=None,
                        historical_sample_size=None,
                        supporting_context=[],
                        contradicting_context=[],
                        sector_context={},
                        management_sentiment=None,
                    )
                )
                errors.append(
                    {
                        "agent": "ContextAgent",
                        "error": str(e),
                        "signal_id": signal.signal_id,
                        "timestamp": datetime.now(),
                    }
                )

        return {"enriched_signals": enriched, "error_log": errors}

    async def _enrich_signal(self, signal: DetectedSignal) -> EnrichedSignal:
        """
        Enrich a single signal with context.

        1. Get historical context from RAG
        2. Get management sentiment
        3. Get sector context
        4. Split passages into supporting vs contradicting
        """
        # 1. Get historical context
        historical = self.retriever.get_historical_context(signal)

        # 2. Get management sentiment
        sentiment_data = self.retriever.get_management_sentiment(signal.ticker)

        # 3. Get sector context
        sector = SECTOR_MAP.get(signal.ticker, "Other")

        # Determine event type for sector context
        event_type = "GENERAL"
        if signal.evidence.get("is_macro_event"):
            event_keywords = signal.evidence.get("event_keywords", [])
            if any("rate cut" in str(kw).lower() for kw in event_keywords):
                event_type = "RATE_CUT"
            elif any("rate hike" in str(kw).lower() for kw in event_keywords):
                event_type = "RATE_HIKE"
        elif "DPCO" in str(signal.evidence.get("event_keywords", [])) or "drug price" in str(signal.evidence.get("title", "")).lower():
            event_type = "DRUG_PRICE_ORDER"
        elif "SEBI" in str(signal.evidence.get("event_keywords", [])):
            event_type = "SEBI_ORDER"

        sector_context = self.retriever.get_sector_context(sector, event_type)

        # 4. Split passages into supporting vs contradicting
        supporting_context = []
        contradicting_context = []

        for passage in historical.get("raw_passages", []):
            # Determine if passage supports or contradicts signal direction
            # For BULLISH signals, positive outcomes support; negative contradict
            # For BEARISH signals, negative outcomes support; positive contradict

            is_positive_outcome = "positive" in passage.lower() or "+%" in passage or "sustained" in passage.lower()

            if signal.direction == SignalDirection.BULLISH:
                if is_positive_outcome:
                    supporting_context.append(passage)
                else:
                    contradicting_context.append(passage)
            elif signal.direction == SignalDirection.BEARISH:
                if not is_positive_outcome:
                    supporting_context.append(passage)
                else:
                    contradicting_context.append(passage)
            else:
                # For NEUTRAL or CONFLICTING, add to both
                supporting_context.append(passage)

        return EnrichedSignal(
            signal=signal,
            historical_base_rate=historical.get("base_rate_positive"),
            historical_sample_size=historical.get("sample_size"),
            supporting_context=supporting_context,
            contradicting_context=contradicting_context,
            sector_context=sector_context,
            management_sentiment=sentiment_data.get("sentiment") if sentiment_data.get("data_available") else None,
        )
