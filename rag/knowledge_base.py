"""
SIGMA Knowledge Base Seeder.
Seeds the vector store with synthetic historical data for hackathon demo.
"""

import random
from datetime import datetime, timedelta
from uuid import uuid4

from rag.vector_store import SigmaVectorStore


class KnowledgeBaseSeeder:
    """
    Seeds the vector store with synthetic historical data for hackathon demo.
    In production, this would ingest real historical NSE data.
    """

    def __init__(self, store: SigmaVectorStore):
        self.store = store

    def seed_all(self) -> None:
        """Seed all collections if they are empty."""
        if self.store.is_empty("historical_patterns"):
            self.seed_historical_patterns()
        if self.store.is_empty("management_commentary"):
            self.seed_management_commentary()
        if self.store.is_empty("sector_context"):
            self.seed_sector_context()

    def seed_historical_patterns(self) -> None:
        """
        Create 50 synthetic historical pattern records and upsert them.
        """
        patterns = []

        # Synthetic tickers
        it_tickers = [f"LARGECAP_IT_{i}" for i in range(1, 11)]
        fmcg_tickers = [f"MIDCAP_FMCG_{i}" for i in range(1, 6)]
        bank_tickers = [f"LARGECAP_BANK_{i}" for i in range(1, 6)]
        pharma_tickers = [f"MIDCAP_PHARMA_{i}" for i in range(1, 6)]

        # 52W breakout (bullish volume confirmation): 60% positive, median +4.2%, worst quartile -8%
        for i in range(12):
            ticker = random.choice(it_tickers + bank_tickers)
            outcome_positive = random.random() < 0.60
            outcome_30d = random.gauss(4.2, 6) if outcome_positive else random.gauss(-8, 4)
            volume_ratio = round(random.uniform(1.6, 3.0), 1)
            rsi = random.randint(55, 68)
            date_str = (datetime.now() - timedelta(days=random.randint(30, 500))).strftime("%Y-%m-%d")

            text = (
                f"52-week breakout on large-cap stock ({ticker}) on {date_str}. "
                f"Volume ratio: {volume_ratio}x. RSI at entry: {rsi} (neutral zone). "
                f"Outcome: 30-day return was {outcome_30d:+.1f}%. "
                f"{'Breakout sustained for ' + str(random.randint(10, 30)) + ' sessions.' if outcome_positive else 'Breakout failed after ' + str(random.randint(3, 8)) + ' sessions.'}"
            )
            patterns.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "signal_type": "TECHNICAL_BREAKOUT",
                    "subtype": "52W_BREAKOUT_VOLUME_CONFIRMED",
                    "market_cap_bucket": "large-cap",
                    "rsi_at_entry": rsi,
                    "volume_ratio": volume_ratio,
                    "outcome_30d": round(outcome_30d, 2),
                    "outcome_positive": outcome_positive,
                    "date": date_str,
                    "source": "historical_analysis",
                },
            })

        # 52W breakout (overbought RSI > 75): 54% positive, median +2.8%, worst quartile -12%
        for i in range(10):
            ticker = random.choice(it_tickers + fmcg_tickers)
            outcome_positive = random.random() < 0.54
            outcome_30d = random.gauss(2.8, 5) if outcome_positive else random.gauss(-12, 5)
            volume_ratio = round(random.uniform(1.5, 2.5), 1)
            rsi = random.randint(75, 85)
            date_str = (datetime.now() - timedelta(days=random.randint(30, 500))).strftime("%Y-%m-%d")

            text = (
                f"52-week breakout on stock ({ticker}) on {date_str}. "
                f"Volume ratio: {volume_ratio}x. RSI at entry: {rsi} (overbought). "
                f"Outcome: 30-day return was {outcome_30d:+.1f}%. "
                f"Elevated RSI at entry increased volatility risk."
            )
            patterns.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "signal_type": "TECHNICAL_BREAKOUT",
                    "subtype": "52W_BREAKOUT_OVERBOUGHT",
                    "market_cap_bucket": random.choice(["large-cap", "mid-cap"]),
                    "rsi_at_entry": rsi,
                    "volume_ratio": volume_ratio,
                    "outcome_30d": round(outcome_30d, 2),
                    "outcome_positive": outcome_positive,
                    "date": date_str,
                    "source": "historical_analysis",
                },
            })

        # 52W breakout (FII selling simultaneously): 48% positive, median +1.1%, worst quartile -15%
        for i in range(8):
            ticker = random.choice(it_tickers + bank_tickers)
            outcome_positive = random.random() < 0.48
            outcome_30d = random.gauss(1.1, 4) if outcome_positive else random.gauss(-15, 5)
            volume_ratio = round(random.uniform(1.5, 2.2), 1)
            fii_change = round(random.uniform(-3.5, -1.0), 1)
            date_str = (datetime.now() - timedelta(days=random.randint(30, 500))).strftime("%Y-%m-%d")

            text = (
                f"52-week breakout on stock ({ticker}) on {date_str}. "
                f"Volume ratio: {volume_ratio}x. FII holding change: {fii_change}% QoQ. "
                f"CONFLICTING SIGNAL: Technical breakout with institutional selling. "
                f"Outcome: 30-day return was {outcome_30d:+.1f}%."
            )
            patterns.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "signal_type": "TECHNICAL_BREAKOUT",
                    "subtype": "52W_BREAKOUT_FII_SELLING",
                    "market_cap_bucket": "large-cap",
                    "fii_qoq_change": fii_change,
                    "volume_ratio": volume_ratio,
                    "outcome_30d": round(outcome_30d, 2),
                    "outcome_positive": outcome_positive,
                    "date": date_str,
                    "source": "historical_analysis",
                },
            })

        # Bulk deal promoter selling (distress signals): 34% positive within 3 months
        for i in range(8):
            ticker = random.choice(fmcg_tickers + pharma_tickers)
            outcome_positive = random.random() < 0.34
            outcome_30d = random.gauss(-8, 6) if not outcome_positive else random.gauss(5, 4)
            discount = round(random.uniform(4, 10), 1)
            stake_sold = round(random.uniform(2.5, 6), 1)
            date_str = (datetime.now() - timedelta(days=random.randint(30, 500))).strftime("%Y-%m-%d")

            text = (
                f"Promoter bulk deal (SELL) on {ticker} on {date_str}. "
                f"Stake sold: {stake_sold}% at {discount}% discount to market. "
                f"DISTRESS INDICATORS: High pledge, margin contraction in recent quarters. "
                f"Outcome: 90-day return was {outcome_30d:+.1f}%."
            )
            patterns.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "signal_type": "BULK_DEAL",
                    "subtype": "PROMOTER_DISTRESS",
                    "market_cap_bucket": "mid-cap",
                    "discount_pct": discount,
                    "stake_sold_pct": stake_sold,
                    "outcome_30d": round(outcome_30d, 2),
                    "outcome_positive": outcome_positive,
                    "date": date_str,
                    "source": "historical_analysis",
                },
            })

        # Bulk deal promoter selling (routine): 71% positive within 3 months
        for i in range(6):
            ticker = random.choice(it_tickers + bank_tickers)
            outcome_positive = random.random() < 0.71
            outcome_30d = random.gauss(6, 5) if outcome_positive else random.gauss(-4, 3)
            discount = round(random.uniform(0, 2), 1)
            stake_sold = round(random.uniform(0.5, 1.5), 1)
            date_str = (datetime.now() - timedelta(days=random.randint(30, 500))).strftime("%Y-%m-%d")

            text = (
                f"Promoter bulk deal (SELL) on {ticker} on {date_str}. "
                f"Stake sold: {stake_sold}% at {discount}% discount (near market price). "
                f"ROUTINE BLOCK: Low pledge, stable margins, no red flags. "
                f"Outcome: 90-day return was {outcome_30d:+.1f}%."
            )
            patterns.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "signal_type": "BULK_DEAL",
                    "subtype": "PROMOTER_ROUTINE",
                    "market_cap_bucket": "large-cap",
                    "discount_pct": discount,
                    "stake_sold_pct": stake_sold,
                    "outcome_30d": round(outcome_30d, 2),
                    "outcome_positive": outcome_positive,
                    "date": date_str,
                    "source": "historical_analysis",
                },
            })

        # RSI oversold bounce: 67% positive 14-day return, median +5.1%
        for i in range(6):
            ticker = random.choice(bank_tickers + pharma_tickers)
            outcome_positive = random.random() < 0.67
            outcome_30d = random.gauss(5.1, 4) if outcome_positive else random.gauss(-6, 3)
            rsi = random.randint(18, 28)
            date_str = (datetime.now() - timedelta(days=random.randint(30, 500))).strftime("%Y-%m-%d")

            text = (
                f"RSI oversold bounce on {ticker} on {date_str}. "
                f"RSI at entry: {rsi} (deeply oversold). "
                f"Outcome: 14-day return was {outcome_30d:+.1f}%."
            )
            patterns.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "signal_type": "RSI_SIGNAL",
                    "subtype": "OVERSOLD_BOUNCE",
                    "market_cap_bucket": random.choice(["large-cap", "mid-cap"]),
                    "rsi_at_entry": rsi,
                    "outcome_30d": round(outcome_30d, 2),
                    "outcome_positive": outcome_positive,
                    "date": date_str,
                    "source": "historical_analysis",
                },
            })

        # Upsert all patterns
        for pattern in patterns:
            doc_id = str(uuid4())
            self.store.upsert_document(
                "historical_patterns",
                doc_id,
                pattern["text"],
                pattern["metadata"],
            )

    def seed_management_commentary(self) -> None:
        """
        Seed 30 synthetic management commentary excerpts.
        """
        commentaries = []
        quarters = ["Q1FY24", "Q2FY24", "Q3FY24", "Q4FY24", "Q1FY25"]
        tickers = [
            "LARGECAP_IT_1", "LARGECAP_IT_2", "MIDCAP_FMCG_1", "MIDCAP_FMCG_2",
            "LARGECAP_BANK_1", "LARGECAP_BANK_2", "MIDCAP_PHARMA_1", "MIDCAP_PHARMA_2",
        ]

        # Positive commentaries (10)
        positive_templates = [
            "Management expressed strong confidence in FY25 guidance. Deal pipeline remains robust with significant large deal wins expected in {quarter}.",
            "Margins expanded by 150bps driven by operational efficiency. Management guided for 22-24% EBITDA margins for FY25.",
            "Strong volume growth of 12% YoY. Rural demand recovery evident. Management expects acceleration in H2.",
            "Record order book of ₹50,000 Cr. Management confident of 15%+ revenue growth guidance.",
            "Successful product launches driving market share gains. Management raised full-year guidance by 200bps.",
        ]

        for i in range(10):
            ticker = random.choice(tickers)
            quarter = random.choice(quarters)
            template = random.choice(positive_templates)
            text = f"{quarter} earnings call - {ticker}: {template.format(quarter=quarter)} Analyst reaction: positive."
            commentaries.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "quarter": quarter,
                    "sentiment": "positive",
                    "topics": "margins,growth,guidance",
                    "signal_type": "MANAGEMENT_COMMENTARY",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "earnings_call",
                },
            })

        # Negative commentaries (10)
        negative_templates = [
            "Management cited cautious outlook on volume growth in H2 due to rural demand softness. Margins expected to remain under pressure at 18-19%.",
            "Deal ramp-ups slower than expected. Management revised down FY25 guidance by 100-150bps due to client budget cuts.",
            "Margin compression of 200bps due to wage hikes and lower utilisation. Management expects pressure to continue in {quarter}.",
            "Elevated pledge levels discussed. Working capital cycle stretched. Management acknowledged challenges in debt repayment schedule.",
            "Regulatory headwinds impacting pricing power. Management guided for flat to negative growth in the pricing-controlled segment.",
        ]

        for i in range(10):
            ticker = random.choice(tickers)
            quarter = random.choice(quarters)
            template = random.choice(negative_templates)
            text = f"{quarter} earnings call - {ticker}: {template.format(quarter=quarter)} Analyst reaction: negative."
            commentaries.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "quarter": quarter,
                    "sentiment": "negative",
                    "topics": "margins,guidance,challenges",
                    "signal_type": "MANAGEMENT_COMMENTARY",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "earnings_call",
                },
            })

        # Neutral commentaries (10)
        neutral_templates = [
            "Management maintained FY25 guidance unchanged. No significant updates to margin outlook. Watchful on macro environment.",
            "Steady state performance in line with expectations. Management reiterated existing guidance without changes.",
            "Mixed quarter with some segments outperforming, others underperforming. Management expects normalisation in H2.",
            "Volume growth in line with industry. Margins within guided range. No surprises in management commentary.",
            "Business as usual quarter. Management focused on execution of existing strategy without major changes.",
        ]

        for i in range(10):
            ticker = random.choice(tickers)
            quarter = random.choice(quarters)
            template = random.choice(neutral_templates)
            text = f"{quarter} earnings call - {ticker}: {template.format(quarter=quarter)} Analyst reaction: muted."
            commentaries.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "quarter": quarter,
                    "sentiment": "neutral",
                    "topics": "guidance,steady",
                    "signal_type": "MANAGEMENT_COMMENTARY",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "earnings_call",
                },
            })

        # Upsert all commentaries
        for commentary in commentaries:
            doc_id = str(uuid4())
            self.store.upsert_document(
                "management_commentary",
                doc_id,
                commentary["text"],
                commentary["metadata"],
            )

    def seed_sector_context(self) -> None:
        """Seed sector-level context data."""
        sector_data = [
            {
                "text": "NBFC sector historical response to RBI rate cuts: Average +8% in 30 days post announcement. NBFCs benefit from lower funding costs and improved NIM spreads.",
                "metadata": {
                    "ticker": "SECTOR_NBFC",
                    "signal_type": "MACRO_EVENT",
                    "event_type": "RATE_CUT",
                    "sector": "NBFC",
                    "avg_impact_pct": 8.0,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "sector_analysis",
                },
            },
            {
                "text": "Banking sector historical response to RBI rate cuts: Average +4% in 30 days. Banks benefit from treasury gains and improved loan demand.",
                "metadata": {
                    "ticker": "SECTOR_BANKING",
                    "signal_type": "MACRO_EVENT",
                    "event_type": "RATE_CUT",
                    "sector": "Banking",
                    "avg_impact_pct": 4.0,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "sector_analysis",
                },
            },
            {
                "text": "IT sector typically neutral to mildly negative on rate cuts: Average -1.2% as rate cuts often signal economic weakness affecting client spending.",
                "metadata": {
                    "ticker": "SECTOR_IT",
                    "signal_type": "MACRO_EVENT",
                    "event_type": "RATE_CUT",
                    "sector": "IT",
                    "avg_impact_pct": -1.2,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "sector_analysis",
                },
            },
            {
                "text": "Pharma sector response to DPCO drug price orders: Initial impact -15% on affected companies. Recovery typically takes 6 months as companies adjust product mix.",
                "metadata": {
                    "ticker": "SECTOR_PHARMA",
                    "signal_type": "REGULATORY_EVENT",
                    "event_type": "DRUG_PRICE_ORDER",
                    "sector": "Pharma",
                    "avg_impact_pct": -15.0,
                    "recovery_timeline": "6 months",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "sector_analysis",
                },
            },
            {
                "text": "SEBI enforcement orders typically cause -8% initial impact across affected companies. Recovery timeline is variable depending on severity.",
                "metadata": {
                    "ticker": "SECTOR_ALL",
                    "signal_type": "REGULATORY_EVENT",
                    "event_type": "SEBI_ORDER",
                    "sector": "All",
                    "avg_impact_pct": -8.0,
                    "recovery_timeline": "variable",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "sector_analysis",
                },
            },
        ]

        for data in sector_data:
            doc_id = str(uuid4())
            self.store.upsert_document(
                "sector_context",
                doc_id,
                data["text"],
                data["metadata"],
            )
