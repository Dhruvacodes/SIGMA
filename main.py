"""
SIGMA Main Application.
FastAPI app entry point - optimized for Vercel serverless.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="SIGMA",
    description="Signal Intelligence & Guided Market Advisor - AI-powered financial signal intelligence for Indian retail investors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with system info."""
    return {
        "name": "SIGMA",
        "description": "Signal Intelligence & Guided Market Advisor",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/health",
    }


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    """API health check with system info."""
    from datetime import datetime
    return {
        "status": "ok",
        "agents": [
            "DataAgent",
            "SignalAgent",
            "ContextAgent",
            "ReasoningAgent",
            "PortfolioAgent",
            "ActionAgent",
        ],
        "timestamp": datetime.now().isoformat(),
    }


# Lazy load routes to reduce cold start time
_routes_loaded = False

@app.on_event("startup")
async def load_routes():
    """Load full API routes on first request."""
    global _routes_loaded
    if not _routes_loaded:
        try:
            from api.routes import router as api_router
            app.include_router(api_router)
            _routes_loaded = True
        except Exception as e:
            print(f"Warning: Could not load full API routes: {e}")


# Export for Vercel
handler = app
