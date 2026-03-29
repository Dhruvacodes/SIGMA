"""
SIGMA Scenario Test Runner.
CLI test runner for the 3 mandatory hackathon scenarios.

Run with: python tests/scenario_runner.py --scenario [1|2|3|all]
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.events import (
    AlertSeverity,
    DetectedSignal,
    EventType,
    FinalAlert,
    RawEvent,
    SignalDirection,
)
from models.portfolio import Holding, RiskProfile, UserPortfolio
from orchestrator import run_agents_from_signals


# Output directory for scenario results
OUTPUT_DIR = Path("./scenario_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class ScenarioResult:
    """Result from running a scenario."""

    def __init__(self, scenario_id: int, name: str):
        self.scenario_id = scenario_id
        self.name = name
        self.alerts: list[FinalAlert] = []
        self.checks: list[tuple[str, bool, str]] = []  # (name, passed, details)
        self.execution_time: float = 0.0
        self.errors: list[str] = []

    def add_check(self, name: str, passed: bool, details: str = ""):
        self.checks.append((name, passed, details))

    @property
    def all_passed(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    @property
    def top_alert(self) -> FinalAlert | None:
        return self.alerts[0] if self.alerts else None


def create_scenario_1_data():
    """
    SCENARIO 1: Bulk Deal / Distress Sell

    Synthetic data:
    - RawEvent: ticker="MIDCAP_FMCG", event_type=BULK_DEAL,
      raw_payload with promoter selling at 6% discount

    Synthetic portfolio:
    - Holdings: MIDCAP_FMCG at 43.5% weight (intentionally high)
    """
    # Create detected signal directly (bypassing data and signal agents)
    signal = DetectedSignal(
        signal_id=str(uuid4()),
        raw_event_id=str(uuid4()),
        ticker="MIDCAP_FMCG",
        signal_type=EventType.BULK_DEAL,
        direction=SignalDirection.BEARISH,
        strength=0.72,
        evidence={
            "classification": "likely_distress",
            "distress_probability": 0.72,
            "is_promoter_flagged": True,
            "client_name": "Promoter Holdings Pvt Ltd",
            "deal_type": "Sell",
            "quantity_pct_equity": 4.2,
            "price_discount_to_prev_close_pct": 6.0,
            "remarks": "Off-market transaction",
            "feature_breakdown": {
                "price_discount": {"contribution": "distress +0.3", "value": 6.0},
                "stake_sold": {"contribution": "distress +0.2", "value": 4.2},
            },
        },
        timestamp=datetime.now(),
        requires_enrichment=True,
    )

    # Create portfolio with high concentration
    portfolio = UserPortfolio(
        user_id="test_user_1",
        holdings=[
            Holding(
                ticker="MIDCAP_FMCG",
                exchange="NSE",
                quantity=1000,
                avg_buy_price=850.0,
                current_price=870.0,
                purchase_date=date(2024, 3, 1),
                sector="FMCG",
            ),
            Holding(
                ticker="OTHER_STOCK",
                exchange="NSE",
                quantity=500,
                avg_buy_price=200.0,
                current_price=220.0,
                purchase_date=date(2023, 6, 1),
                sector="IT",
            ),
        ],
        risk_profile=RiskProfile.MODERATE,
        last_updated=datetime.now(),
    )

    return [signal], portfolio


def create_scenario_2_data():
    """
    SCENARIO 2: Breakout with Conflicting Signals

    Synthetic DetectedSignals:
    - Signal 1: 52W breakout (BULLISH, strength 0.82)
    - Signal 2: RSI overbought (BEARISH, strength 0.53)
    - Signal 3: FII selling (BEARISH, strength 0.45)

    Synthetic portfolio: does NOT hold LARGECAP_IT
    """
    signals = [
        DetectedSignal(
            signal_id=str(uuid4()),
            raw_event_id=str(uuid4()),
            ticker="LARGECAP_IT",
            signal_type=EventType.TECHNICAL_BREAKOUT,
            direction=SignalDirection.BULLISH,
            strength=0.82,
            evidence={
                "breakout_pct": 1.8,
                "volume_ratio": 2.1,
                "signal_subtype": "52W_BREAKOUT",
            },
            timestamp=datetime.now(),
            requires_enrichment=True,
        ),
        DetectedSignal(
            signal_id=str(uuid4()),
            raw_event_id=str(uuid4()),
            ticker="LARGECAP_IT",
            signal_type=EventType.RSI_SIGNAL,
            direction=SignalDirection.BEARISH,
            strength=0.53,
            evidence={
                "rsi": 78,
                "status": "overbought",
            },
            timestamp=datetime.now(),
            requires_enrichment=True,
        ),
        DetectedSignal(
            signal_id=str(uuid4()),
            raw_event_id=str(uuid4()),
            ticker="LARGECAP_IT",
            signal_type=EventType.FII_FLOW_CHANGE,
            direction=SignalDirection.BEARISH,
            strength=0.45,
            evidence={
                "fii_qoq_change_pct": -1.8,
                "z_score": -1.9,
            },
            timestamp=datetime.now(),
            requires_enrichment=True,
        ),
    ]

    # Empty portfolio (does not hold LARGECAP_IT)
    portfolio = UserPortfolio(
        user_id="test_user_2",
        holdings=[
            Holding(
                ticker="SOME_OTHER_STOCK",
                exchange="NSE",
                quantity=100,
                avg_buy_price=500.0,
                current_price=520.0,
                purchase_date=date(2023, 1, 1),
                sector="Banking",
            ),
        ],
        risk_profile=RiskProfile.MODERATE,
        last_updated=datetime.now(),
    )

    return signals, portfolio


def create_scenario_3_data():
    """
    SCENARIO 3: Portfolio-Aware Prioritisation

    Simultaneous events:
    - Event A (macro): RBI rate cut → affects NBFC holdings
    - Event B (regulatory): DPCO drug price order → affects PHARMA_HOLDING

    Synthetic portfolio:
    - PHARMA_HOLDING: 9% weight
    - NBFC_1: 12% weight
    - NBFC_2: 10% weight
    - 5 other stocks at ~14% each
    """
    signals = [
        # Macro event (rate cut) - positive for NBFCs
        DetectedSignal(
            signal_id=str(uuid4()),
            raw_event_id=str(uuid4()),
            ticker="NBFC_1",
            signal_type=EventType.NEWS_EVENT,
            direction=SignalDirection.BULLISH,
            strength=0.55,
            evidence={
                "event_keywords": ["RBI rate cut", "repo rate", "-25bps"],
                "sentiment": "positive",
                "is_macro_event": True,
                "title": "RBI cuts repo rate by 25bps",
            },
            timestamp=datetime.now(),
            requires_enrichment=True,
        ),
        # Regulatory event (drug price order) - negative for pharma
        DetectedSignal(
            signal_id=str(uuid4()),
            raw_event_id=str(uuid4()),
            ticker="PHARMA_HOLDING",
            signal_type=EventType.NEWS_EVENT,
            direction=SignalDirection.BEARISH,
            strength=0.65,
            evidence={
                "event_keywords": ["DPCO", "drug price order", "pricing cap"],
                "sentiment": "negative",
                "title": "DPCO brings more drugs under price control",
            },
            timestamp=datetime.now(),
            requires_enrichment=True,
        ),
    ]

    # Portfolio with pharma and NBFC holdings
    portfolio = UserPortfolio(
        user_id="test_user_3",
        holdings=[
            Holding(
                ticker="PHARMA_HOLDING",
                exchange="NSE",
                quantity=200,
                avg_buy_price=900.0,
                current_price=900.0,
                purchase_date=date(2024, 1, 1),
                sector="Pharma",
            ),
            Holding(
                ticker="NBFC_1",
                exchange="NSE",
                quantity=400,
                avg_buy_price=600.0,
                current_price=600.0,
                purchase_date=date(2023, 6, 1),
                sector="NBFC",
            ),
            Holding(
                ticker="NBFC_2",
                exchange="NSE",
                quantity=400,
                avg_buy_price=500.0,
                current_price=500.0,
                purchase_date=date(2023, 6, 1),
                sector="NBFC",
            ),
            Holding(
                ticker="OTHER_1",
                exchange="NSE",
                quantity=280,
                avg_buy_price=1000.0,
                current_price=1000.0,
                purchase_date=date(2023, 1, 1),
                sector="IT",
            ),
            Holding(
                ticker="OTHER_2",
                exchange="NSE",
                quantity=280,
                avg_buy_price=1000.0,
                current_price=1000.0,
                purchase_date=date(2023, 1, 1),
                sector="IT",
            ),
        ],
        risk_profile=RiskProfile.MODERATE,
        last_updated=datetime.now(),
    )

    return signals, portfolio


async def run_scenario(scenario_id: int) -> ScenarioResult:
    """
    Run a specific scenario.

    Args:
        scenario_id: Scenario number (1, 2, or 3).

    Returns:
        ScenarioResult with alerts and check results.
    """
    scenarios = {
        1: ("Bulk Deal / Distress Sell", create_scenario_1_data),
        2: ("Breakout with Conflicting Signals", create_scenario_2_data),
        3: ("Portfolio-Aware Prioritisation", create_scenario_3_data),
    }

    if scenario_id not in scenarios:
        raise ValueError(f"Invalid scenario ID: {scenario_id}")

    name, data_fn = scenarios[scenario_id]
    result = ScenarioResult(scenario_id, name)

    print(f"\n{'='*60}")
    print(f"SCENARIO {scenario_id}: {name}")
    print("=" * 60)

    try:
        start_time = time.time()

        # Create scenario data
        signals, portfolio = data_fn()
        print(f"Created {len(signals)} signal(s)")
        print(f"Portfolio: {len(portfolio.holdings)} holdings, ₹{portfolio.total_value:,.0f} total value")

        # Run agents from signals (bypassing data/signal agents)
        alerts = await run_agents_from_signals(signals, portfolio)

        result.execution_time = time.time() - start_time
        result.alerts = alerts

        print(f"\nGenerated {len(alerts)} alert(s) in {result.execution_time:.2f}s")

        # Run scenario-specific checks
        if scenario_id == 1:
            run_scenario_1_checks(result, portfolio)
        elif scenario_id == 2:
            run_scenario_2_checks(result)
        elif scenario_id == 3:
            run_scenario_3_checks(result)

        # Print check results
        print("\nCHECKS:")
        for check_name, passed, details in result.checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {check_name}")
            if details and not passed:
                print(f"         {details}")

        # Save results to file
        save_scenario_results(result)

    except Exception as e:
        result.errors.append(str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    return result


def run_scenario_1_checks(result: ScenarioResult, portfolio: UserPortfolio):
    """Run checks for Scenario 1: Bulk Deal / Distress Sell."""
    if not result.alerts:
        result.add_check("Has alerts", False, "No alerts generated")
        return

    alert = result.alerts[0]

    # Check severity is URGENT
    result.add_check(
        "Severity is URGENT",
        alert.severity == AlertSeverity.URGENT,
        f"Got: {alert.severity}",
    )

    # Check confidence >= 0.60
    result.add_check(
        "Confidence >= 0.60",
        alert.confidence_score >= 0.60,
        f"Got: {alert.confidence_score:.2f}",
    )

    # Check "promoter" in signal_summary
    result.add_check(
        "'promoter' in signal_summary",
        "promoter" in alert.signal_summary.lower(),
        f"Signal summary: {alert.signal_summary[:50]}...",
    )

    # Check recommended_action contains "reduce" or "SEBI"
    has_action_keyword = "reduce" in alert.recommended_action.lower() or "sebi" in alert.recommended_action.lower()
    result.add_check(
        "'reduce' or 'SEBI' in recommended_action",
        has_action_keyword,
        f"Got: {alert.recommended_action[:50]}...",
    )

    # Check concentration_risk
    if alert.portfolio_impact:
        result.add_check(
            "concentration_risk is True",
            alert.portfolio_impact.concentration_risk,
            f"Got: {alert.portfolio_impact.concentration_risk}",
        )
    else:
        result.add_check("concentration_risk is True", False, "No portfolio impact")

    # Check disclaimer
    result.add_check(
        "Disclaimer present and >100 chars",
        alert.disclaimer is not None and len(alert.disclaimer) > 100,
        f"Length: {len(alert.disclaimer) if alert.disclaimer else 0}",
    )


def run_scenario_2_checks(result: ScenarioResult):
    """Run checks for Scenario 2: Breakout with Conflicting Signals."""
    if not result.alerts:
        result.add_check("Has alerts", False, "No alerts generated")
        return

    alert = result.alerts[0]

    # Check severity is WATCH or INFORMATIONAL (not URGENT or OPPORTUNITY)
    result.add_check(
        "Severity is WATCH or INFORMATIONAL",
        alert.severity in [AlertSeverity.WATCH, AlertSeverity.INFORMATIONAL],
        f"Got: {alert.severity}",
    )

    # Check conflict_analysis is present
    result.add_check(
        "conflict_analysis is not None",
        alert.conflict_analysis is not None,
        f"Got: {alert.conflict_analysis}",
    )

    # Check confidence < 0.60 (penalized for conflict)
    result.add_check(
        "Confidence < 0.60",
        alert.confidence_score < 0.60,
        f"Got: {alert.confidence_score:.2f}",
    )

    # Check "watch" or "conflicting" in recommended_action
    has_watch = "watch" in alert.recommended_action.lower() or "conflicting" in alert.recommended_action.lower()
    result.add_check(
        "'watch' or 'conflicting' in recommended_action",
        has_watch,
        f"Got: {alert.recommended_action[:50]}...",
    )

    # Check reasoning trace has 6 steps
    result.add_check(
        "reasoning_trace has 6 steps",
        len(alert.reasoning_trace) == 6,
        f"Got: {len(alert.reasoning_trace)} steps",
    )


def run_scenario_3_checks(result: ScenarioResult):
    """Run checks for Scenario 3: Portfolio-Aware Prioritisation."""
    if not result.alerts:
        result.add_check("Has alerts", False, "No alerts generated")
        return

    # Check first alert is PHARMA_HOLDING (prioritized as URGENT)
    result.add_check(
        "First alert is PHARMA_HOLDING",
        result.alerts[0].ticker == "PHARMA_HOLDING",
        f"Got: {result.alerts[0].ticker}",
    )

    # Check first alert severity is URGENT
    result.add_check(
        "First alert severity is URGENT",
        result.alerts[0].severity == AlertSeverity.URGENT,
        f"Got: {result.alerts[0].severity}",
    )

    # Check PHARMA_HOLDING has negative P&L delta
    pharma_alert = result.alerts[0]
    if pharma_alert.portfolio_impact:
        result.add_check(
            "PHARMA P&L delta is negative",
            pharma_alert.portfolio_impact.estimated_pnl_delta_inr < 0,
            f"Got: ₹{pharma_alert.portfolio_impact.estimated_pnl_delta_inr:,.0f}",
        )
    else:
        result.add_check("PHARMA P&L delta is negative", False, "No portfolio impact")

    # Print prioritisation notice
    print("\nPRIORITISATION: Alerts ordered by urgency × P&L impact")


def save_scenario_results(result: ScenarioResult):
    """Save scenario results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"scenario_{result.scenario_id}_{timestamp}.json"

    output = {
        "scenario_id": result.scenario_id,
        "scenario_name": result.name,
        "execution_time_seconds": result.execution_time,
        "alert_count": len(result.alerts),
        "all_checks_passed": result.all_passed,
        "checks": [
            {"name": name, "passed": passed, "details": details}
            for name, passed, details in result.checks
        ],
        "alerts": [alert.model_dump(mode="json") for alert in result.alerts],
        "errors": result.errors,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {filename}")


async def run_all_scenarios() -> list[ScenarioResult]:
    """Run all 3 scenarios."""
    results = []
    for scenario_id in [1, 2, 3]:
        result = await run_scenario(scenario_id)
        results.append(result)
    return results


def print_summary(results: list[ScenarioResult]):
    """Print summary table of all scenario results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Scenario':<30} {'Alerts':<8} {'Top Alert':<20} {'Conf':<8} {'Time':<8}")
    print("-" * 74)

    for result in results:
        top_alert = result.top_alert
        if top_alert:
            top_alert_str = f"{top_alert.severity.value} ({top_alert.ticker[:10]})"
            conf = f"{top_alert.confidence_score:.2f}"
        else:
            top_alert_str = "None"
            conf = "N/A"

        print(
            f"{result.name:<30} {len(result.alerts):<8} {top_alert_str:<20} {conf:<8} {result.execution_time:.1f}s"
        )

    # Overall status
    all_passed = all(r.all_passed for r in results)
    status = "ALL SCENARIOS PASSED" if all_passed else "SOME CHECKS FAILED"
    print(f"\n{status}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SIGMA Scenario Test Runner")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario to run: 1, 2, 3, or 'all'",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        results = asyncio.run(run_all_scenarios())
        print_summary(results)
    else:
        try:
            scenario_id = int(args.scenario)
            asyncio.run(run_scenario(scenario_id))
        except ValueError:
            print(f"Invalid scenario: {args.scenario}")
            print("Use: 1, 2, 3, or 'all'")
            sys.exit(1)


if __name__ == "__main__":
    main()
