"""
SIGMA Main Application.
FastAPI app entry point with dashboard UI.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Create FastAPI app
app = FastAPI(
    title="SIGMA",
    description="Signal Intelligence & Guided Market Advisor",
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

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>SIGMA</h1><p>Dashboard not found. Visit <a href='/docs'>/docs</a> for API.</p>")


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    """API health check."""
    from datetime import datetime
    return {
        "status": "ok",
        "agents": ["DataAgent", "SignalAgent", "ContextAgent", "ReasoningAgent", "PortfolioAgent", "ActionAgent"],
        "timestamp": datetime.now().isoformat(),
    }


# Load API routes
try:
    from api.routes import router as api_router
    app.include_router(api_router)
except Exception as e:
    print(f"Warning: Could not load API routes: {e}")


# Mount static files (for any additional assets)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
