"""
SIGMA Action Agent.
Generates final alerts with recommendations and guardrails.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from config import settings
from guardrails.disclaimer import DISCLAIMER_TEXT, validate_alert_guardrails
from models.events import (
    AlertSeverity,
    FinalAlert,
    PortfolioImpact,
    ReasoningOutput,
    SignalDirection,
)
from models.state import SigmaState


class ActionAgent:
    """
    Action Agent for SIGMA.
    Generates final alerts with prioritisation and guardrails.
    """

    def __init__(self):
        self.client = None

    async def run(self, state: SigmaState) -> dict[str, Any]:
        """
        Generate final alerts from reasoning outputs.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with final_alerts.
        """
        alerts: list[FinalAlert] = []
        errors: list[dict] = []

        # Match reasoning outputs to portfolio impacts by ticker
        impact_map = {imp.ticker: imp for imp in state.get("portfolio_impacts", [])}

        # PRIORITISATION: Sort by urgency × impact before generating alerts
        prioritised = self._prioritise(state.get("reasoning_outputs", []), impact_map)

        # Get enriched signals for source lookup
        enriched_map = {
            e.signal.signal_id: e for e in state.get("enriched_signals", [])
        }

        for reasoning in prioritised:
            try:
                impact = impact_map.get(reasoning.ticker)
                enriched = enriched_map.get(reasoning.enriched_signal_id)

                alert = await self._generate_alert(reasoning, impact, enriched, state)

                # Validate guardrails
                validation_errors = validate_alert_guardrails(alert)
                if validation_errors:
                    # Fix common issues
                    if not alert.sources:
                        alert.sources = ["Internal analysis"]
                    if len(alert.headline) > 120:
                        alert.headline = alert.headline[:117] + "..."

                alerts.append(alert)
            except Exception as e:
                errors.append(
                    {
                        "agent": "ActionAgent",
                        "error": str(e),
                        "ticker": reasoning.ticker,
                        "timestamp": datetime.now(),
                    }
                )

        return {"final_alerts": alerts, "error_log": errors}

    def _prioritise(
        self, outputs: list[ReasoningOutput], impact_map: dict[str, PortfolioImpact]
    ) -> list[ReasoningOutput]:
        """
        Score and sort reasoning outputs for prioritisation.

        Priority = urgency_score * abs(pnl_impact)
        """

        def score(r: ReasoningOutput) -> float:
            urgency = {
                "BEARISH": 1.0,
                "CONFLICTING": 0.7,
                "BULLISH": 0.5,
                "NEUTRAL": 0.3,
            }.get(r.direction.value, 0.3)

            imp = impact_map.get(r.ticker)
            pnl = abs(imp.estimated_pnl_delta_inr) if imp else 1.0
            return urgency * (pnl if pnl > 0 else 1.0)

        return sorted(outputs, key=score, reverse=True)

    async def _generate_alert(
        self,
        reasoning: ReasoningOutput,
        impact: PortfolioImpact | None,
        enriched: Any,
        state: SigmaState,
    ) -> FinalAlert:
        """
        Generate a final alert from reasoning output.
        """
        # Determine severity
        severity = self._determine_severity(reasoning, impact)

        # Build recommended action
        recommended_action = self._build_recommended_action(reasoning, impact)

        # Build allocation guidance
        allocation_guidance = None
        if impact and impact.recommended_weight_pct is not None:
            if reasoning.direction == SignalDirection.BULLISH:
                allocation_guidance = f"Consider adding to reach {impact.recommended_weight_pct:.1f}% of portfolio"
            elif reasoning.direction == SignalDirection.BEARISH:
                allocation_guidance = f"Consider reducing to {impact.recommended_weight_pct:.1f}% of portfolio"

        # Build headline
        headline = self._build_headline(reasoning, severity)

        # Build signal summary
        signal_summary = self._build_signal_summary(reasoning)

        # Build context summary
        context_summary = self._build_context_summary(reasoning, enriched)

        # Build conflict analysis
        conflict_analysis = None
        if reasoning.conflict_detected:
            conflict_analysis = reasoning.conflict_description or "Conflicting signals detected. Multiple indicators point in opposite directions."

        # Collect sources
        sources = self._collect_sources(enriched, state)

        # Build supporting data
        supporting_data = []
        if enriched and enriched.supporting_context:
            supporting_data = [ctx[:100] for ctx in enriched.supporting_context[:3]]

        return FinalAlert(
            alert_id=str(uuid4()),
            ticker=reasoning.ticker,
            severity=severity,
            headline=headline[:120],  # Ensure max 120 chars
            signal_summary=signal_summary,
            supporting_data=supporting_data,
            context_summary=context_summary,
            conflict_analysis=conflict_analysis,
            portfolio_impact=impact,
            recommended_action=recommended_action,
            allocation_guidance=allocation_guidance,
            confidence_score=reasoning.confidence_score,
            reasoning_trace=reasoning.reasoning_chain,
            sources=sources if sources else ["Internal analysis"],
            disclaimer=DISCLAIMER_TEXT,
            generated_at=datetime.now(),
            model_used=settings.REASONING_MODEL,
        )

    def _determine_severity(
        self, reasoning: ReasoningOutput, impact: PortfolioImpact | None
    ) -> AlertSeverity:
        """Determine alert severity based on direction, confidence, and portfolio."""
        in_portfolio = impact and impact.current_weight_pct > 0
        high_confidence = reasoning.confidence_score >= settings.HIGH_CONFIDENCE_THRESHOLD
        medium_confidence = reasoning.confidence_score >= settings.LOW_CONFIDENCE_THRESHOLD

        if reasoning.direction == SignalDirection.BEARISH and high_confidence and in_portfolio:
            return AlertSeverity.URGENT

        if reasoning.direction == SignalDirection.BULLISH and high_confidence:
            if not in_portfolio or (impact and impact.current_weight_pct < 5):
                return AlertSeverity.OPPORTUNITY

        if reasoning.conflict_detected or (medium_confidence and not high_confidence):
            return AlertSeverity.WATCH

        return AlertSeverity.INFORMATIONAL

    def _build_recommended_action(
        self, reasoning: ReasoningOutput, impact: PortfolioImpact | None
    ) -> str:
        """Build recommended action based on decision table."""
        in_portfolio = impact and impact.current_weight_pct > 0
        confidence = reasoning.confidence_score
        direction = reasoning.direction

        # Handle conflicting signals
        if direction == SignalDirection.CONFLICTING:
            return "Conflicting signals detected. Watch for resolution. Do not enter/exit based on this signal alone."

        # Low confidence
        if confidence < settings.LOW_CONFIDENCE_THRESHOLD:
            return "Monitor only. No action recommended due to low signal confidence."

        # Medium confidence
        if confidence < settings.HIGH_CONFIDENCE_THRESHOLD:
            trigger = reasoning.stop_loss_trigger or "further confirmation"
            return f"Watch. Entry trigger: {trigger}"

        # High confidence BULLISH
        if direction == SignalDirection.BULLISH:
            if not in_portfolio:
                rec_weight = impact.recommended_weight_pct if impact and impact.recommended_weight_pct else 3
                return f"Consider adding {rec_weight:.0f}% of portfolio"
            elif impact and impact.current_weight_pct < 5:
                rec_weight = impact.recommended_weight_pct if impact.recommended_weight_pct else 5
                return f"Consider adding to reach {rec_weight:.0f}% allocation"
            else:
                return "Hold. Position size already adequate."

        # High confidence BEARISH
        if direction == SignalDirection.BEARISH:
            if in_portfolio:
                rec_weight = impact.recommended_weight_pct if impact and impact.recommended_weight_pct else 0
                stop_loss = reasoning.stop_loss_trigger or "CMP - 8%"
                return f"Consider reducing to <{rec_weight:.0f}%. Stop-loss: {stop_loss}"
            else:
                return "Avoid entry. Signal indicates potential downside."

        return "Monitor. No specific action recommended."

    def _build_headline(self, reasoning: ReasoningOutput, severity: AlertSeverity) -> str:
        """Build concise headline."""
        direction_word = {
            SignalDirection.BULLISH: "bullish",
            SignalDirection.BEARISH: "bearish",
            SignalDirection.CONFLICTING: "conflicting",
            SignalDirection.NEUTRAL: "neutral",
        }.get(reasoning.direction, "neutral")

        severity_prefix = {
            AlertSeverity.URGENT: "URGENT:",
            AlertSeverity.OPPORTUNITY: "OPPORTUNITY:",
            AlertSeverity.WATCH: "WATCH:",
            AlertSeverity.INFORMATIONAL: "",
        }.get(severity, "")

        return f"{severity_prefix} {reasoning.ticker} shows {direction_word} signal (confidence: {reasoning.confidence_score:.0%})"

    def _build_signal_summary(self, reasoning: ReasoningOutput) -> str:
        """Build signal summary from reasoning chain."""
        if reasoning.reasoning_chain:
            first_step = reasoning.reasoning_chain[0]
            return first_step.get("finding", "Signal detected")
        return "Signal detected"

    def _build_context_summary(self, reasoning: ReasoningOutput, enriched: Any) -> str:
        """Build context summary."""
        parts = []

        if enriched and enriched.historical_base_rate is not None:
            rate_pct = enriched.historical_base_rate * 100
            parts.append(f"Historical success rate: {rate_pct:.0f}%")

        if enriched and enriched.management_sentiment:
            parts.append(f"Management sentiment: {enriched.management_sentiment}")

        if reasoning.conflict_detected:
            parts.append("Note: Conflicting signals present")

        if parts:
            return ". ".join(parts) + "."
        return "Limited historical context available."

    def _collect_sources(self, enriched: Any, state: SigmaState) -> list[str]:
        """Collect source references."""
        sources = set()

        # Add source from enriched signal
        if enriched and hasattr(enriched, "signal"):
            # Look up raw event source
            for event in state.get("raw_events", []):
                if event.event_id == enriched.signal.raw_event_id:
                    sources.add(event.source)
                    if event.raw_payload.get("url"):
                        sources.add(event.raw_payload["url"])
                    break

        # Add RAG sources
        sources.add("SIGMA Historical Pattern Database")

        return list(sources) if sources else ["Internal analysis"]

    def _validate_guardrails(self, alert: FinalAlert) -> None:
        """
        Validate alert against guardrails.

        Raises:
            ValueError: If guardrails fail.
        """
        errors = validate_alert_guardrails(alert)
        if errors:
            raise ValueError(f"Guardrail violations: {', '.join(errors)}")
