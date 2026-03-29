"""
SIGMA LangGraph State definition.
This is the shared mutable state object passed between all nodes.
"""

import operator
from typing import Annotated, TypedDict

from models.events import (
    DetectedSignal,
    EnrichedSignal,
    FinalAlert,
    PortfolioImpact,
    RawEvent,
    ReasoningOutput,
)
from models.portfolio import UserPortfolio


class SigmaState(TypedDict):
    """
    Shared state object for the SIGMA pipeline.
    Uses Annotated with operator.add for list fields so LangGraph can merge partial updates.
    """
    raw_events: Annotated[list[RawEvent], operator.add]
    detected_signals: Annotated[list[DetectedSignal], operator.add]
    enriched_signals: Annotated[list[EnrichedSignal], operator.add]
    reasoning_outputs: Annotated[list[ReasoningOutput], operator.add]
    portfolio: UserPortfolio | None
    portfolio_impacts: Annotated[list[PortfolioImpact], operator.add]
    final_alerts: Annotated[list[FinalAlert], operator.add]
    error_log: Annotated[list[dict], operator.add]  # {agent: str, error: str, timestamp: datetime}
    audit_trail: Annotated[list[dict], operator.add]  # append-only log of every agent transition
