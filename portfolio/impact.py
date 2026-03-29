"""
SIGMA Portfolio Impact Calculations.
Computes P&L impact, concentration risk, and tax considerations.
"""

from typing import TYPE_CHECKING

from config import settings
from models.events import PortfolioImpact, SignalDirection

if TYPE_CHECKING:
    from models.events import ReasoningOutput
    from models.portfolio import UserPortfolio


# Sector mapping for stocks
SECTOR_MAP = {
    "INFY": "IT",
    "TCS": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "KOTAKBANK": "Banking",
    "AXISBANK": "Banking",
    "INDUSINDBK": "Banking",
    "RELIANCE": "Oil & Gas",
    "ONGC": "Oil & Gas",
    "GAIL": "Oil & Gas",
    "SUNPHARMA": "Pharma",
    "DRREDDY": "Pharma",
    "CIPLA": "Pharma",
    "BAJFINANCE": "NBFC",
    "BAJAJFINSV": "NBFC",
    "MUTHOOTFIN": "NBFC",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "DABUR": "FMCG",
    "BRITANNIA": "FMCG",
    "ASIANPAINT": "Consumer",
    "TITAN": "Consumer",
    "PIDILITIND": "Consumer",
    "TATAMOTORS": "Auto",
    "MARUTI": "Auto",
    "HEROMOTOCO": "Auto",
    "EICHERMOT": "Auto",
    "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",
    "HINDALCO": "Metals",
    "COALINDIA": "Mining",
    "NTPC": "Power",
    "POWERGRID": "Power",
    "LT": "Infrastructure",
    "ULTRACEMCO": "Cement",
    "GRASIM": "Cement",
    "ADANIPORTS": "Infrastructure",
    "ADANIENT": "Conglomerate",
    "BHARTIARTL": "Telecom",
}


def get_sector(ticker: str) -> str:
    """Get sector for a ticker."""
    return SECTOR_MAP.get(ticker, "Other")


def compute_portfolio_impact(
    reasoning: "ReasoningOutput",
    portfolio: "UserPortfolio",
    historical_median_return: float | None = None,
) -> PortfolioImpact:
    """
    Compute portfolio impact for a reasoning output.

    Args:
        reasoning: ReasoningOutput from the reasoning agent.
        portfolio: User's portfolio.
        historical_median_return: Median historical return for similar signals.

    Returns:
        PortfolioImpact with all calculations.
    """
    ticker = reasoning.ticker

    # 1. Get current weight
    current_weight_pct = portfolio.get_weight(ticker)

    # 2. Get current value
    holding = portfolio.get_holding(ticker)
    current_value_inr = holding.current_value if holding else 0.0

    # 3. Compute estimated P&L delta
    expected_move = _get_expected_move(
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
    recommended_weight = _compute_recommended_weight(
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
