"""
SIGMA Audit Logger.
Append-only audit log for full reproducibility.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.events import FinalAlert
    from models.state import SigmaState


class SigmaAuditLogger:
    """
    Singleton audit logger that logs all agent outputs.

    All agent outputs are logged here before being returned.
    This is what makes the system "enterprise ready" — every alert is fully reproducible.
    """

    _instance = None

    def __new__(cls, log_dir: str = "./audit_logs"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir: str = "./audit_logs"):
        if self._initialized:
            return
        self._initialized = True

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._update_log_file()

    def _update_log_file(self) -> None:
        """Update log file path for current day."""
        self.log_file = self.log_dir / f"sigma_audit_{date.today().isoformat()}.jsonl"

    def _ensure_current_day(self) -> None:
        """Ensure we're writing to today's log file."""
        expected_file = self.log_dir / f"sigma_audit_{date.today().isoformat()}.jsonl"
        if self.log_file != expected_file:
            self._update_log_file()

    def log_agent_run(self, agent_name: str, **kwargs) -> None:
        """Log an agent execution event."""
        self._ensure_current_day()
        record = {
            "type": "agent_run",
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self._append(record)

    def log_pipeline_run(self, final_state: "SigmaState") -> None:
        """Log complete pipeline execution with full state snapshot."""
        self._ensure_current_day()

        # Handle potential list of FinalAlert objects
        alerts = final_state.get("final_alerts", [])
        alerts_serialized = []
        for a in alerts:
            if hasattr(a, "model_dump"):
                alerts_serialized.append(a.model_dump(mode="json"))
            else:
                alerts_serialized.append(a)

        # Handle error log serialization
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
        self._append(record)

    def log_alert(self, alert: "FinalAlert") -> None:
        """Log individual alert with full reasoning trace."""
        self._ensure_current_day()
        record = {
            "type": "alert",
            "alert_id": alert.alert_id,
            "ticker": alert.ticker,
            "severity": alert.severity,
            "confidence": alert.confidence_score,
            "reasoning_trace": alert.reasoning_trace,
            "sources": alert.sources,
            "generated_at": alert.generated_at.isoformat(),
            "model_used": alert.model_used,
        }
        self._append(record)

    def _append(self, record: dict) -> None:
        """Append a record to the log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def get_alerts_from_log(self, alert_id: str) -> dict | None:
        """Retrieve a specific alert from the audit log by alert_id."""
        self._ensure_current_day()
        if not self.log_file.exists():
            return None

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("type") == "alert" and record.get("alert_id") == alert_id:
                        return record
                    # Also check in pipeline_run alerts
                    if record.get("type") == "pipeline_run":
                        for alert in record.get("alerts", []):
                            if alert.get("alert_id") == alert_id:
                                return alert
                except json.JSONDecodeError:
                    continue
        return None


# Module-level singleton
audit_logger = SigmaAuditLogger()
