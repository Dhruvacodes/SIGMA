"""
SIGMA Main Application.
FastAPI app entry point.
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from api.websocket import alert_websocket_handler

# Create FastAPI app
app = FastAPI(
    title="SIGMA",
    description="Signal Intelligence & Guided Market Advisor - AI-powered financial signal intelligence for Indian retail investors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration (allow all origins for hackathon demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint with system info."""
    return {
        "name": "SIGMA",
        "description": "Signal Intelligence & Guided Market Advisor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket, portfolio_id: str | None = None):
    """
    WebSocket endpoint for real-time alert streaming.

    Query params:
        portfolio_id: Optional portfolio ID to use for personalized alerts.
    """
    await alert_websocket_handler(websocket, portfolio_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
