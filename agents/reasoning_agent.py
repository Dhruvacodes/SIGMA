"""
SIGMA Reasoning Agent.
Core intelligence using LLM with structured Chain-of-Thought prompting.
Supports both Groq and Anthropic as LLM providers.
"""

import json
from datetime import datetime
from typing import Any

from config import settings
from models.events import (
    EnrichedSignal,
    EventType,
    ReasoningOutput,
    SignalDirection,
)
from models.state import SigmaState

REASONING_SYSTEM_PROMPT = """
You are SIGMA's Reasoning Agent — a financial signal analyst for Indian retail investors.

Your job: Given a detected signal and its enriched context, produce a structured 6-step reasoning chain.
You MUST follow this exact structure. Do NOT skip steps. Do NOT produce only a conclusion.

Output ONLY valid JSON matching this schema:
{
  "reasoning_chain": [
    {"step": 1, "label": "Signal assessment", "finding": "<what is the primary signal and its measured strength?>"},
    {"step": 2, "label": "Confirming evidence", "finding": "<what context supports this signal? cite specific data points>"},
    {"step": 3, "label": "Contradicting evidence", "finding": "<what context CONTRADICTS this signal? be specific. If none, say 'No significant contradicting evidence found.'>"},
    {"step": 4, "label": "Historical base rate", "finding": "<in similar past conditions, what % resolved in the signal direction? cite sample size>"},
    {"step": 5, "label": "Risk factors", "finding": "<what specific conditions would invalidate this signal?>"},
    {"step": 6, "label": "Confidence assessment", "finding": "<explain your confidence score calculation>"}
  ],
  "confidence_score": <float 0.0-1.0>,
  "direction": "<BULLISH|BEARISH|NEUTRAL|CONFLICTING>",
  "conflict_detected": <true|false>,
  "conflict_description": "<describe the conflict if present, else null>",
  "stop_loss_trigger": "<specific trigger, e.g. 'Close below 20-day SMA' or 'CMP - 8%', or null>"
}

CONFLICT RULE: If confirming_evidence_weight > 0.4 AND contradicting_evidence_weight > 0.4,
set conflict_detected=true and direction=CONFLICTING.
Reduce confidence_score by 20% from what you would otherwise assign.
NEVER output a binary bullish/bearish call for CONFLICTING signals.

CONFIDENCE FORMULA (apply this explicitly in step 6):
base_confidence = signal_strength * 0.3
             + historical_base_rate_alignment * 0.3  (0 if no data)
             + context_alignment * 0.2
             + volume_confirmation * 0.2  (0 if N/A)
If conflict_detected: final_confidence = base_confidence * 0.8

REGULATORY COMPLIANCE: Never guarantee returns. Never say "will go up/down".
Use: "historical data suggests", "signal indicates", "consider", "may".
"""


class ReasoningAgent:
    """
    Reasoning Agent for SIGMA.
    Uses LLM with structured CoT prompting for signal analysis.
    Supports both Groq and Anthropic as LLM providers.
    """

    def __init__(self):
        self._groq_client = None
        self._anthropic_client = None
        self.provider = settings.LLM_PROVIDER  # "groq" or "anthropic"

        # Set models based on provider
        if self.provider == "groq":
            self.small_model = settings.GROQ_FAST_MODEL
            self.reasoning_model = settings.GROQ_REASONING_MODEL
        else:
            self.small_model = settings.FAST_MODEL
            self.reasoning_model = settings.REASONING_MODEL

    def _get_groq_client(self):
        """Lazy load Groq client."""
        if self._groq_client is None:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=settings.GROQ_API_KEY)
            except ImportError:
                raise ImportError("groq package required. Install with: pip install groq")
        return self._groq_client

    def _get_anthropic_client(self):
        """Lazy load Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._anthropic_client

    async def run(self, state: SigmaState) -> dict[str, Any]:
        """
        Process enriched signals and produce reasoning outputs.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with reasoning_outputs.
        """
        reasoning_outputs: list[ReasoningOutput] = []
        errors: list[dict] = []

        for enriched in state.get("enriched_signals", []):
            try:
                output = await self._reason_about_signal(enriched)
                reasoning_outputs.append(output)
            except Exception as e:
                # Create minimal output on failure
                reasoning_outputs.append(
                    ReasoningOutput(
                        enriched_signal_id=enriched.signal.signal_id,
                        ticker=enriched.signal.ticker,
                        reasoning_chain=[
                            {"step": 1, "label": "Error", "finding": f"Reasoning failed: {str(e)}"}
                        ],
                        confidence_score=0.0,
                        direction=SignalDirection.NEUTRAL,
                        conflict_detected=False,
                        conflict_description=None,
                        risk_factors=["Reasoning agent error"],
                        stop_loss_trigger=None,
                    )
                )
                errors.append(
                    {
                        "agent": "ReasoningAgent",
                        "error": str(e),
                        "signal_id": enriched.signal.signal_id,
                        "timestamp": datetime.now(),
                    }
                )

        return {"reasoning_outputs": reasoning_outputs, "error_log": errors}

    async def _reason_about_signal(self, enriched: EnrichedSignal) -> ReasoningOutput:
        """
        Build user message from enriched signal data, call LLM, parse response.
        """
        # Build context message
        user_message = self._build_reasoning_prompt(enriched)

        # Determine model based on signal complexity
        use_heavy = (
            enriched.signal.strength >= 0.4
            or enriched.signal.signal_type == EventType.BULK_DEAL
            or len(enriched.contradicting_context) > 0
        )
        model = self.reasoning_model if use_heavy else self.small_model

        # Call LLM based on provider
        if self.provider == "groq":
            raw_json = self._call_groq(model, user_message)
            model_used = f"groq/{model}"
        else:
            raw_json = self._call_anthropic(model, user_message)
            model_used = f"anthropic/{model}"

        # Strip markdown code fences if present
        raw_json = raw_json.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        elif raw_json.startswith("```"):
            raw_json = raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
        raw_json = raw_json.strip()

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_json)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError(f"Failed to parse LLM response as JSON: {e}")

        # Extract risk factors from step 5
        risk_factors = []
        for step in parsed.get("reasoning_chain", []):
            if step.get("step") == 5:
                risk_factors.append(step.get("finding", ""))

        return ReasoningOutput(
            enriched_signal_id=enriched.signal.signal_id,
            ticker=enriched.signal.ticker,
            reasoning_chain=parsed.get("reasoning_chain", []),
            confidence_score=float(parsed.get("confidence_score", 0.0)),
            direction=SignalDirection(parsed.get("direction", "NEUTRAL")),
            conflict_detected=bool(parsed.get("conflict_detected", False)),
            conflict_description=parsed.get("conflict_description"),
            risk_factors=risk_factors,
            stop_loss_trigger=parsed.get("stop_loss_trigger"),
        )

    def _call_groq(self, model: str, user_message: str) -> str:
        """Call Groq API and return response text."""
        client = self._get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1500,
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, model: str, user_message: str) -> str:
        """Call Anthropic API and return response text."""
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            system=REASONING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _build_reasoning_prompt(self, enriched: EnrichedSignal) -> str:
        """
        Build a structured prompt from enriched signal data.
        """
        lines = [
            f"SIGNAL: {enriched.signal.signal_type.value} on {enriched.signal.ticker}",
            f"DIRECTION: {enriched.signal.direction.value}",
            f"STRENGTH: {enriched.signal.strength:.2f}",
            "",
            "EVIDENCE:",
        ]

        for k, v in enriched.signal.evidence.items():
            lines.append(f"- {k}: {v}")

        lines += ["", "HISTORICAL CONTEXT:"]
        if enriched.historical_base_rate is not None:
            lines.append(
                f"- Base rate positive: {enriched.historical_base_rate*100:.0f}% (n={enriched.historical_sample_size})"
            )
        else:
            lines.append("- No historical data available")

        lines += ["", "SUPPORTING CONTEXT:"]
        if enriched.supporting_context:
            for p in enriched.supporting_context[:3]:
                lines.append(f"- {p[:200]}")
        else:
            lines.append("- None")

        lines += ["", "CONTRADICTING CONTEXT:"]
        if enriched.contradicting_context:
            for p in enriched.contradicting_context[:3]:
                lines.append(f"- {p[:200]}")
        else:
            lines.append("- None")

        if enriched.management_sentiment:
            lines += ["", f"MANAGEMENT SENTIMENT: {enriched.management_sentiment}"]

        if enriched.sector_context:
            lines += ["", "SECTOR CONTEXT:"]
            for k, v in enriched.sector_context.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)
