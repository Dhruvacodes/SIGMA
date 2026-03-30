"""
SIGMA Portfolio Pydantic schemas.
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class RiskProfile(StrEnum):
    """User's risk profile."""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class Holding(BaseModel):
    """A single stock holding in a portfolio."""
    ticker: str
    quantity: int
    exchange: str = "NSE"
    avg_buy_price: float = 0.0
    avg_cost: float = 0.0  # Alias for avg_buy_price
    current_price: float = 0.0
    purchase_date: date = Field(default_factory=date.today)
    sector: str = "Unknown"

    def model_post_init(self, __context):
        # Use avg_cost as fallback for avg_buy_price
        if self.avg_buy_price == 0.0 and self.avg_cost > 0:
            object.__setattr__(self, 'avg_buy_price', self.avg_cost)

    @computed_field
    @property
    def current_value(self) -> float:
        """Current market value of the holding."""
        price = self.current_price if self.current_price > 0 else self.avg_buy_price
        return self.quantity * price

    @computed_field
    @property
    def unrealised_pnl(self) -> float:
        """Unrealised profit/loss on the holding."""
        if self.avg_buy_price == 0:
            return 0.0
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
    risk_profile: RiskProfile = RiskProfile.MODERATE
    last_updated: datetime = Field(default_factory=datetime.now)
    total_value: float = 0.0  # Can be provided or computed
    tax_regime: str = "new"

    @computed_field
    @property
    def computed_total_value(self) -> float:
        """Total market value of all holdings."""
        return sum(h.current_value for h in self.holdings)

    def get_weight(self, ticker: str) -> float:
        """
        Get the weight (percentage) of a ticker in the portfolio.

        Returns:
            Weight as a percentage (0-100), or 0.0 if not held.
        """
        total = self.computed_total_value or self.total_value
        if total == 0:
            return 0.0
        for holding in self.holdings:
            if holding.ticker == ticker:
                return (holding.current_value / total) * 100
        return 0.0

    def get_holding(self, ticker: str) -> Holding | None:
        """Get a specific holding by ticker."""
        for holding in self.holdings:
            if holding.ticker == ticker:
                return holding
        return None

    def get_sector_weight(self, sector: str) -> float:
        """Get the total weight of a sector in the portfolio."""
        total = self.computed_total_value or self.total_value
        if total == 0:
            return 0.0
        sector_value = sum(h.current_value for h in self.holdings if h.sector == sector)
        return (sector_value / total) * 100
