"""
SIGMA Portfolio Agent.
Computes portfolio-specific impact for each signal.
"""

from datetime import datetime
from typing import Any

from config import settings
from models.events import PortfolioImpact, ReasoningOutput, SignalDirection
from models.portfolio import UserPortfolio
from models.state import SigmaState
from portfolio.impact import SECTOR_MAP, compute_portfolio_impact, get_sector


class PortfolioAgent:
    """
    Portfolio Agent for SIGMA.
    Computes portfolio-specific impact for each reasoning output.
    """

    async def run(self, state: SigmaState) -> dict[str, Any]:
        """
        For each reasoning output, compute portfolio-specific impact.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with portfolio_impacts.
        """
        portfolio = state.get("portfolio")

        if not portfolio:
            # No portfolio → create informational impacts with no personal data
            impacts = [
                self._no_portfolio_impact(r) for r in state.get("reasoning_outputs", [])
            ]
            return {"portfolio_impacts": impacts}

        impacts: list[PortfolioImpact] = []
        errors: list[dict] = []

        # Get enriched signals for historical data lookup
        enriched_map = {
            e.signal.signal_id: e for e in state.get("enriched_signals", [])
        }

        for reasoning in state.get("reasoning_outputs", []):
            try:
                # Get historical median return if available
                enriched = enriched_map.get(reasoning.enriched_signal_id)
                historical_median = None
                if enriched and enriched.historical_base_rate is not None:
                    # Estimate median return based on base rate
                    # This is a simplification; in production, would use actual median from RAG
                    if enriched.historical_base_rate > 0.5:
                        historical_median = 4.0  # Positive bias
                    else:
                        historical_median = -6.0  # Negative bias

                impact = self._compute_impact(reasoning, portfolio, historical_median)
                impacts.append(impact)
            except Exception as e:
                impacts.append(self._no_portfolio_impact(reasoning))
                errors.append(
                    {
                        "agent": "PortfolioAgent",
                        "error": str(e),
                        "ticker": reasoning.ticker,
                        "timestamp": datetime.now(),
                    }
                )

        return {"portfolio_impacts": impacts, "error_log": errors}

    def _compute_impact(
        self,
        reasoning: ReasoningOutput,
        portfolio: UserPortfolio,
        historical_median_return: float | None = None,
    ) -> PortfolioImpact:
        """
        Compute full portfolio impact for a reasoning output.
        """
        ticker = reasoning.ticker

        # 1. Get current weight
        current_weight_pct = portfolio.get_weight(ticker)

        # 2. Get current value
        holding = portfolio.get_holding(ticker)
        current_value_inr = holding.current_value if holding else 0.0

        # 3. Compute expected price move and P&L delta
        expected_move = self._get_expected_move(
            reasoning.direction,
            reasoning.confidence_score,
            historical_median_return,
        )
        pnl_delta = current_value_inr * (expected_move / 100) * reasoning.confidence_score

        # 4. Check concentration risk
        concentration_risk = (
            current_weight_pct > settings.MAX_SINGLE_STOCK_WEIGHT_PCT
            or (current_weight_pct + 3) > settings.MAX_SINGLE_STOCK_WEIGHT_PCT
        )

        # 5. Check sector overweight
        sector = get_sector(ticker)
        sector_weight = portfolio.get_sector_weight(sector)
        sector_overweight = sector_weight > settings.MAX_SECTOR_WEIGHT_PCT

        # 6. Tax consideration
        tax_consideration = None
        if holding and holding.is_stcg_eligible and reasoning.direction == SignalDirection.BEARISH:
            tax_consideration = (
                "Note: Selling now would trigger Short-Term Capital Gains tax (15%). "
                "Consider if gain is >₹1L."
            )

        # 7. Recommended weight
        recommended_weight = self._compute_recommended_weight(
            reasoning.direction,
            reasoning.confidence_score,
            current_weight_pct,
            concentration_risk,
            sector_overweight,
            portfolio.risk_profile.value,
        )

        return PortfolioImpact(
            ticker=ticker,
            current_weight_pct=current_weight_pct,
            current_value_inr=current_value_inr,
            estimated_pnl_delta_inr=pnl_delta,
            concentration_risk=concentration_risk,
            sector_overweight=sector_overweight,
            tax_consideration=tax_consideration,
            recommended_weight_pct=recommended_weight,
        )

    def _get_expected_move(
        self,
        direction: SignalDirection,
        confidence: float,
        historical_median: float | None,
    ) -> float:
        """Compute expected price move based on direction and historical data."""
        if direction == SignalDirection.CONFLICTING:
            return 0.0

        # Use historical median if available
        if historical_median is not None:
            if direction == SignalDirection.BEARISH:
                return -abs(historical_median)
            return historical_median

        # Fallback defaults
        if direction == SignalDirection.BULLISH:
            return 3.0
        elif direction == SignalDirection.BEARISH:
            return -6.0
        else:
            return 0.0

    def _compute_recommended_weight(
        self,
        direction: SignalDirection,
        confidence: float,
        current_weight: float,
        concentration_risk: bool,
        sector_overweight: bool,
        risk_profile: str,
    ) -> float | None:
        """Compute recommended portfolio weight."""
        # No recommendation for conflicting or low confidence
        if direction == SignalDirection.CONFLICTING:
            return None
        if confidence < settings.LOW_CONFIDENCE_THRESHOLD:
            return None

        # Risk profile adjustment
        profile_adj = 0.0
        if risk_profile == "CONSERVATIVE":
            profile_adj = -1.0
        elif risk_profile == "AGGRESSIVE":
            profile_adj = 1.0

        # High confidence recommendations
        if confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
            if direction == SignalDirection.BULLISH:
                if concentration_risk or sector_overweight:
                    # Don't recommend adding to overweight position
                    return None
                target = min(current_weight + 3 + profile_adj, 8)
                return max(0.0, target)

            elif direction == SignalDirection.BEARISH:
                target = max(current_weight - 3 + profile_adj, 0)
                return target

        return None

    def _no_portfolio_impact(self, reasoning: ReasoningOutput) -> PortfolioImpact:
        """Create minimal impact for when no portfolio is provided."""
        return PortfolioImpact(
            ticker=reasoning.ticker,
            current_weight_pct=0.0,
            current_value_inr=0.0,
            estimated_pnl_delta_inr=0.0,
            concentration_risk=False,
            sector_overweight=False,
            tax_consideration=None,
            recommended_weight_pct=None,
        )
