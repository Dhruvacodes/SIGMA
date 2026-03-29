"""
SIGMA Portfolio Pydantic schemas.
"""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, computed_field


class RiskProfile(StrEnum):
    """User's risk profile."""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class Holding(BaseModel):
    """A single stock holding in a portfolio."""
    ticker: str
    exchange: str
    quantity: int
    avg_buy_price: float
    current_price: float
    purchase_date: date
    sector: str

    @computed_field
    @property
    def current_value(self) -> float:
        """Current market value of the holding."""
        return self.quantity * self.current_price

    @computed_field
    @property
    def unrealised_pnl(self) -> float:
        """Unrealised profit/loss on the holding."""
        return self.quantity * (self.current_price - self.avg_buy_price)

    @computed_field
    @property
    def holding_period_days(self) -> int:
        """Number of days the holding has been held."""
        return (date.today() - self.purchase_date).days

    @computed_field
    @property
    def is_stcg_eligible(self) -> bool:
        """Whether selling would trigger Short-Term Capital Gains tax (held < 365 days)."""
        return self.holding_period_days < 365


class UserPortfolio(BaseModel):
    """A user's complete portfolio."""
    user_id: str
    holdings: list[Holding]
    risk_profile: RiskProfile
    last_updated: datetime

    @computed_field
    @property
    def total_value(self) -> float:
        """Total market value of all holdings."""
        return sum(h.current_value for h in self.holdings)

    def get_weight(self, ticker: str) -> float:
        """
        Get the weight (percentage) of a ticker in the portfolio.

        Returns:
            Weight as a percentage (0-100), or 0.0 if not held.
        """
        if self.total_value == 0:
            return 0.0
        for holding in self.holdings:
            if holding.ticker == ticker:
                return (holding.current_value / self.total_value) * 100
        return 0.0

    def get_holding(self, ticker: str) -> Holding | None:
        """Get a specific holding by ticker."""
        for holding in self.holdings:
            if holding.ticker == ticker:
                return holding
        return None

    def get_sector_weight(self, sector: str) -> float:
        """Get the total weight of a sector in the portfolio."""
        if self.total_value == 0:
            return 0.0
        sector_value = sum(h.current_value for h in self.holdings if h.sector == sector)
        return (sector_value / self.total_value) * 100
