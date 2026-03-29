"""
SIGMA WebSocket Handler.
Real-time alert streaming over WebSocket.
"""

import asyncio
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from orchestrator import run_pipeline
from portfolio.store import portfolio_store


async def alert_websocket_handler(websocket: WebSocket, portfolio_id: str | None = None):
    """
    WebSocket handler for streaming alerts.

    Args:
        websocket: WebSocket connection.
        portfolio_id: Optional portfolio ID to use.
    """
    await websocket.accept()

    try:
        # Fetch portfolio if provided
        portfolio = None
        if portfolio_id:
            portfolio = portfolio_store.get(portfolio_id)

        # Notify client that pipeline is starting
        await websocket.send_json(
            {
                "type": "pipeline_started",
                "timestamp": datetime.now().isoformat(),
                "portfolio_loaded": portfolio is not None,
            }
        )

        # Run the pipeline
        try:
            alerts = await run_pipeline(portfolio)
        except Exception as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return

        # Stream each alert with a slight delay for visual effect
        for alert in alerts:
            await websocket.send_json(
                {
                    "type": "alert",
                    "data": alert.model_dump(mode="json"),
                }
            )
            await asyncio.sleep(0.1)  # Slight delay for streaming feel

        # Notify client that pipeline is complete
        await websocket.send_json(
            {
                "type": "pipeline_complete",
                "alert_count": len(alerts),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            pass
