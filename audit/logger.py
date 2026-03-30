"""
SIGMA Audit Logger.
In-memory audit log for serverless deployment (Vercel has read-only filesystem).
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.events import FinalAlert
    from models.state import SigmaState


class SigmaAuditLogger:
    """
    Singleton audit logger that logs all agent outputs.
    Uses in-memory storage for serverless compatibility.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._logs: list[dict] = []
        self._alerts: dict[str, dict] = {}

    def log_agent_run(self, agent_name: str, **kwargs) -> None:
        """Log an agent execution event."""
        record = {
            "type": "agent_run",
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self._logs.append(record)

    def log_pipeline_run(self, final_state: "SigmaState") -> None:
        """Log complete pipeline execution with full state snapshot."""
        alerts = final_state.get("final_alerts", [])
        alerts_serialized = []
        for a in alerts:
            if hasattr(a, "model_dump"):
                alert_dict = a.model_dump(mode="json")
                alerts_serialized.append(alert_dict)
                # Store for retrieval
                self._alerts[alert_dict.get("alert_id", "")] = alert_dict
            else:
                alerts_serialized.append(a)
                if isinstance(a, dict) and "alert_id" in a:
                    self._alerts[a["alert_id"]] = a

        errors = final_state.get("error_log", [])
        errors_serialized = []
        for e in errors:
            if isinstance(e, dict):
                serialized = {}
                for k, v in e.items():
                    if isinstance(v, datetime):
                        serialized[k] = v.isoformat()
                    else:
                        serialized[k] = v
                errors_serialized.append(serialized)
            else:
                errors_serialized.append(e)

        record = {
            "type": "pipeline_run",
            "timestamp": datetime.now().isoformat(),
            "signal_count": len(final_state.get("detected_signals", [])),
            "alert_count": len(alerts),
            "error_count": len(errors),
            "errors": errors_serialized,
            "alerts": alerts_serialized,
            "audit_trail": final_state.get("audit_trail", []),
        }
        self._logs.append(record)

    def log_alert(self, alert: "FinalAlert") -> None:
        """Log individual alert with full reasoning trace."""
        record = {
            "type": "alert",
            "alert_id": alert.alert_id,
            "ticker": alert.ticker,
            "severity": alert.severity.value if hasattr(alert.severity, 'value') else alert.severity,
            "confidence": alert.confidence_score,
            "reasoning_trace": alert.reasoning_trace,
            "sources": alert.sources,
            "generated_at": alert.generated_at.isoformat() if hasattr(alert.generated_at, 'isoformat') else str(alert.generated_at),
            "model_used": alert.model_used,
        }
        self._logs.append(record)
        self._alerts[alert.alert_id] = record

    def get_alerts_from_log(self, alert_id: str) -> dict | None:
        """Retrieve a specific alert from the audit log by alert_id."""
        return self._alerts.get(alert_id)

    def get_recent_logs(self, limit: int = 100) -> list[dict]:
        """Get recent log entries."""
        return self._logs[-limit:]


# Module-level singleton
audit_logger = SigmaAuditLogger()
