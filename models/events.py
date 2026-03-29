"""
SIGMA Event and Signal Pydantic schemas.
All agent inputs and outputs MUST conform to these schemas.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Types of events that can be detected."""
    BULK_DEAL = "BULK_DEAL"
    TECHNICAL_BREAKOUT = "TECHNICAL_BREAKOUT"
    RSI_SIGNAL = "RSI_SIGNAL"
    FII_FLOW_CHANGE = "FII_FLOW_CHANGE"
    NEWS_EVENT = "NEWS_EVENT"
    EARNINGS_UPDATE = "EARNINGS_UPDATE"


class SignalDirection(StrEnum):
    """Direction of a detected signal."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    CONFLICTING = "CONFLICTING"


class AlertSeverity(StrEnum):
    """Severity level of a final alert."""
    URGENT = "URGENT"
    OPPORTUNITY = "OPPORTUNITY"
    INFORMATIONAL = "INFORMATIONAL"
    WATCH = "WATCH"


class RawEvent(BaseModel):
    """Raw event ingested from data sources."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    ticker: str
    exchange: Literal["NSE", "BSE"]
    event_type: EventType
    timestamp: datetime
    raw_payload: dict  # original parsed data, kept for audit
    source: str  # "SEBI_FILING", "NSE_WEBSOCKET", "ET_NEWS", etc.
    parse_confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class DetectedSignal(BaseModel):
    """Signal detected from a raw event."""
    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_event_id: str
    ticker: str
    signal_type: EventType
    direction: SignalDirection
    strength: float = Field(ge=0.0, le=1.0)
    evidence: dict  # structured evidence dict, e.g. {"rsi": 78, "volume_ratio": 2.1}
    timestamp: datetime
    requires_enrichment: bool = True


class EnrichedSignal(BaseModel):
    """Signal enriched with historical context and RAG data."""
    signal: DetectedSignal
    historical_base_rate: float | None = None  # % of similar past signals that resolved in signal direction
    historical_sample_size: int | None = None
    supporting_context: list[str] = Field(default_factory=list)  # list of retrieved RAG passages
    contradicting_context: list[str] = Field(default_factory=list)
    sector_context: dict = Field(default_factory=dict)
    management_sentiment: str | None = None  # "positive" / "negative" / "neutral" / None


class ReasoningOutput(BaseModel):
    """Output from the reasoning agent's CoT analysis."""
    enriched_signal_id: str
    ticker: str
    reasoning_chain: list[dict]  # list of {step: int, label: str, finding: str}
    confidence_score: float = Field(ge=0.0, le=1.0)
    direction: SignalDirection
    conflict_detected: bool
    conflict_description: str | None = None
    risk_factors: list[str] = Field(default_factory=list)
    stop_loss_trigger: str | None = None  # human-readable, e.g. "CMP - 8%"


class PortfolioImpact(BaseModel):
    """Impact assessment for a user's portfolio."""
    ticker: str
    current_weight_pct: float  # 0-100
    current_value_inr: float
    estimated_pnl_delta_inr: float  # confidence-weighted
    concentration_risk: bool
    sector_overweight: bool
    tax_consideration: str | None = None
    recommended_weight_pct: float | None = None


class FinalAlert(BaseModel):
    """Final alert generated for the user."""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    ticker: str
    severity: AlertSeverity
    headline: str  # one line, max 120 chars
    signal_summary: str
    supporting_data: list[str] = Field(default_factory=list)  # cited data points with source
    context_summary: str
    conflict_analysis: str | None = None
    portfolio_impact: PortfolioImpact | None = None
    recommended_action: str
    allocation_guidance: str | None = None  # e.g., "Consider adding 3-5% of portfolio"
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning_trace: list[dict] = Field(default_factory=list)  # full CoT chain, for audit
    sources: list[str] = Field(default_factory=list)  # URLs or filing IDs
    disclaimer: str  # always populated, never empty
    generated_at: datetime
    model_used: str  # which LLM was used for reasoning
