"""
SIGMA API Routes.
FastAPI route definitions - with graceful error handling for serverless.
"""

import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["api"])


class PipelineRequest(BaseModel):
    """Request body for running the pipeline."""
    portfolio: dict | None = None


class PipelineResponse(BaseModel):
    """Response from pipeline execution."""
    alerts: list[dict[str, Any]]
    pipeline_duration_ms: float
    signal_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    agents: list[str]
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
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
        timestamp=datetime.now().isoformat(),
    )


@router.post("/run-pipeline", response_model=PipelineResponse)
async def run_pipeline_endpoint(request: PipelineRequest) -> PipelineResponse:
    """
    Run the SIGMA pipeline.
    """
    start_time = time.time()

    try:
        # Lazy import to reduce cold start
        from orchestrator import run_pipeline
        from models.portfolio import UserPortfolio

        portfolio = None
        if request.portfolio:
            portfolio = UserPortfolio(**request.portfolio)

        alerts = await run_pipeline(portfolio)
        duration_ms = (time.time() - start_time) * 1000

        return PipelineResponse(
            alerts=[a.model_dump(mode="json") for a in alerts],
            pipeline_duration_ms=duration_ms,
            signal_count=len(alerts),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.get("/alerts/{alert_id}")
async def get_alert(alert_id: str) -> dict[str, Any]:
    """Retrieve a specific alert from the audit log."""
    try:
        from audit.logger import audit_logger
        alert = audit_logger.get_alerts_from_log(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        return alert
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio")
async def create_portfolio(portfolio: dict) -> dict[str, Any]:
    """Store a user portfolio."""
    try:
        from models.portfolio import UserPortfolio
        from portfolio.store import portfolio_store

        user_portfolio = UserPortfolio(**portfolio)
        portfolio_store.save(user_portfolio)

        return {
            "user_id": user_portfolio.user_id,
            "holdings_count": len(user_portfolio.holdings),
            "total_value": user_portfolio.total_value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/{user_id}")
async def get_portfolio(user_id: str) -> dict[str, Any]:
    """Retrieve a stored portfolio."""
    try:
        from portfolio.store import portfolio_store
        portfolio = portfolio_store.get(user_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        return portfolio.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seed-knowledge-base")
async def seed_knowledge_base() -> dict[str, str]:
    """Seed the RAG knowledge base with synthetic data."""
    try:
        from rag.vector_store import SigmaVectorStore
        from rag.knowledge_base import KnowledgeBaseSeeder

        store = SigmaVectorStore()
        seeder = KnowledgeBaseSeeder(store)
        seeder.seed_all()
        return {"status": "Knowledge base seeded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seeding error: {str(e)}")
