"""
SIGMA API Routes.
FastAPI route definitions for the SIGMA system.
"""

import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from audit.logger import audit_logger
from models.events import AlertSeverity, FinalAlert
from models.portfolio import UserPortfolio
from orchestrator import run_pipeline
from portfolio.store import portfolio_store
from rag.knowledge_base import KnowledgeBaseSeeder
from rag.vector_store import SigmaVectorStore

router = APIRouter(prefix="/api", tags=["api"])


class PipelineRequest(BaseModel):
    """Request body for running the pipeline."""
    portfolio: UserPortfolio | None = None


class PipelineResponse(BaseModel):
    """Response from pipeline execution."""
    alerts: list[dict[str, Any]]
    pipeline_duration_ms: float
    signal_count: int


class PortfolioResponse(BaseModel):
    """Response from portfolio creation."""
    user_id: str
    holdings_count: int
    total_value: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    agents: list[str]
    rag_initialized: bool
    timestamp: str


# Severity ordering for sorting
SEVERITY_ORDER = {
    AlertSeverity.URGENT: 0,
    AlertSeverity.OPPORTUNITY: 1,
    AlertSeverity.WATCH: 2,
    AlertSeverity.INFORMATIONAL: 3,
}


@router.post("/run-pipeline", response_model=PipelineResponse)
async def run_pipeline_endpoint(request: PipelineRequest) -> PipelineResponse:
    """
    Run the SIGMA pipeline.

    Args:
        request: Pipeline request with optional portfolio.

    Returns:
        Pipeline response with alerts and metrics.
    """
    start_time = time.time()

    try:
        alerts = await run_pipeline(request.portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    duration_ms = (time.time() - start_time) * 1000

    # Sort alerts by severity
    sorted_alerts = sorted(
        alerts,
        key=lambda a: SEVERITY_ORDER.get(a.severity, 99),
    )

    return PipelineResponse(
        alerts=[a.model_dump(mode="json") for a in sorted_alerts],
        pipeline_duration_ms=duration_ms,
        signal_count=len(alerts),
    )


@router.get("/alerts/{alert_id}")
async def get_alert(alert_id: str) -> dict[str, Any]:
    """
    Retrieve a specific alert from the audit log.

    Args:
        alert_id: Alert ID to retrieve.

    Returns:
        Full alert with reasoning trace.
    """
    alert = audit_logger.get_alerts_from_log(alert_id)

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return alert


@router.post("/portfolio", response_model=PortfolioResponse)
async def create_portfolio(portfolio: UserPortfolio) -> PortfolioResponse:
    """
    Store a user portfolio.

    Args:
        portfolio: Portfolio to store.

    Returns:
        Confirmation with portfolio summary.
    """
    portfolio_store.save(portfolio)

    return PortfolioResponse(
        user_id=portfolio.user_id,
        holdings_count=len(portfolio.holdings),
        total_value=portfolio.total_value,
    )


@router.get("/portfolio/{user_id}")
async def get_portfolio(user_id: str) -> dict[str, Any]:
    """
    Retrieve a stored portfolio.

    Args:
        user_id: User ID to retrieve.

    Returns:
        Portfolio data.
    """
    portfolio = portfolio_store.get(user_id)

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return portfolio.model_dump(mode="json")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Quick health check for the SIGMA system.

    Returns:
        Health status with system info.
    """
    # Check if RAG is initialized
    rag_initialized = False
    try:
        store = SigmaVectorStore()
        rag_initialized = not store.is_empty("historical_patterns")
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        agents=[
            "DataAgent",
            "SignalAgent",
            "ContextAgent",
            "ReasoningAgent",
            "PortfolioAgent",
            "ActionAgent",
        ],
        rag_initialized=rag_initialized,
        timestamp=datetime.now().isoformat(),
    )


@router.post("/seed-knowledge-base")
async def seed_knowledge_base() -> dict[str, str]:
    """
    Seed the RAG knowledge base with synthetic data.

    Returns:
        Confirmation message.
    """
    try:
        store = SigmaVectorStore()
        seeder = KnowledgeBaseSeeder(store)
        seeder.seed_all()
        return {"status": "Knowledge base seeded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seeding error: {str(e)}")
