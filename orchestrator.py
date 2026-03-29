"""
SIGMA Orchestrator.
LangGraph graph definition for the multi-agent pipeline.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.action_agent import ActionAgent
from agents.context_agent import ContextAgent
from agents.data_agent import DataAgent
from agents.portfolio_agent import PortfolioAgent
from agents.reasoning_agent import ReasoningAgent
from agents.signal_agent import SignalAgent
from audit.logger import audit_logger
from guardrails.disclaimer import DISCLAIMER_TEXT
from models.events import AlertSeverity, FinalAlert
from models.portfolio import UserPortfolio
from models.state import SigmaState


def build_sigma_graph():
    """
    Build and compile the SIGMA agent graph.

    GRAPH STRUCTURE:
    START → data_agent → signal_agent → context_agent → reasoning_agent
         → [conditional] → portfolio_agent → action_agent → END
                        → summary_node → END

    Returns:
        Compiled LangGraph workflow.
    """
    workflow = StateGraph(SigmaState)

    # Instantiate agents
    data_agent = DataAgent()
    signal_agent = SignalAgent()
    context_agent = ContextAgent()
    reasoning_agent = ReasoningAgent()
    portfolio_agent = PortfolioAgent()
    action_agent = ActionAgent()

    # Add nodes (async functions)
    workflow.add_node("data_agent", data_agent.run)
    workflow.add_node("signal_agent", signal_agent.run)
    workflow.add_node("context_agent", context_agent.run)
    workflow.add_node("reasoning_agent", reasoning_agent.run)
    workflow.add_node("portfolio_agent", portfolio_agent.run)
    workflow.add_node("action_agent", action_agent.run)
    workflow.add_node("summary_node", generate_summary_node)

    # Add edges
    workflow.set_entry_point("data_agent")
    workflow.add_edge("data_agent", "signal_agent")
    workflow.add_edge("signal_agent", "context_agent")
    workflow.add_edge("context_agent", "reasoning_agent")

    # Conditional edge: skip to summary if all low confidence
    workflow.add_conditional_edges(
        "reasoning_agent",
        should_skip_to_summary,
        {"skip": "summary_node", "continue": "portfolio_agent"},
    )

    workflow.add_edge("portfolio_agent", "action_agent")
    workflow.add_edge("action_agent", END)
    workflow.add_edge("summary_node", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def should_skip_to_summary(state: SigmaState) -> Literal["skip", "continue"]:
    """
    Determine whether to skip to summary node.

    Returns "skip" if:
    - No reasoning outputs, OR
    - All confidence scores < 0.3 AND no portfolio holdings are impacted

    Returns "continue" otherwise.
    """
    outputs = state.get("reasoning_outputs", [])

    if not outputs:
        return "skip"

    portfolio = state.get("portfolio")
    if not portfolio:
        tickers_in_portfolio = set()
    else:
        tickers_in_portfolio = {h.ticker for h in portfolio.holdings}

    all_low_conf = all(r.confidence_score < 0.3 for r in outputs)
    none_in_portfolio = not any(r.ticker in tickers_in_portfolio for r in outputs)

    return "skip" if (all_low_conf and none_in_portfolio) else "continue"


async def generate_summary_node(state: SigmaState) -> dict[str, Any]:
    """
    Minimal fallback: create INFORMATIONAL alerts for low-confidence signals.
    """
    alerts = []

    for r in state.get("reasoning_outputs", []):
        alerts.append(
            FinalAlert(
                alert_id=str(uuid4()),
                ticker=r.ticker,
                severity=AlertSeverity.INFORMATIONAL,
                headline=f"Low-confidence signal detected for {r.ticker} — monitor only",
                signal_summary=r.reasoning_chain[0]["finding"] if r.reasoning_chain else "Signal detected",
                supporting_data=[],
                context_summary="Insufficient confidence for actionable recommendation.",
                conflict_analysis=None,
                portfolio_impact=None,
                recommended_action="Monitor only. No action recommended.",
                allocation_guidance=None,
                confidence_score=r.confidence_score,
                reasoning_trace=r.reasoning_chain,
                sources=["Internal analysis"],
                disclaimer=DISCLAIMER_TEXT,
                generated_at=datetime.now(),
                model_used="none",
            )
        )

    return {"final_alerts": alerts}


async def run_pipeline(portfolio: UserPortfolio | None = None) -> list[FinalAlert]:
    """
    Entry point to run the full pipeline once.

    Args:
        portfolio: Optional user portfolio for personalized alerts.

    Returns:
        List of FinalAlert objects.
    """
    graph = build_sigma_graph()

    initial_state: SigmaState = {
        "raw_events": [],
        "detected_signals": [],
        "enriched_signals": [],
        "reasoning_outputs": [],
        "portfolio": portfolio,
        "portfolio_impacts": [],
        "final_alerts": [],
        "error_log": [],
        "audit_trail": [],
    }

    config = {"configurable": {"thread_id": str(uuid4())}}
    final_state = await graph.ainvoke(initial_state, config=config)

    # Log full state to audit log
    audit_logger.log_pipeline_run(final_state)

    return final_state.get("final_alerts", [])


async def run_pipeline_from_state(
    initial_state: SigmaState,
    start_node: str = "data_agent",
) -> SigmaState:
    """
    Run pipeline from a given state, optionally starting from a specific node.

    This is useful for testing scenarios where we want to bypass early agents.

    Args:
        initial_state: Pre-populated state.
        start_node: Node to start from (for testing).

    Returns:
        Final state after pipeline completion.
    """
    graph = build_sigma_graph()

    config = {"configurable": {"thread_id": str(uuid4())}}

    # For testing, we may want to start from a later node
    # This requires building a custom graph or manipulating state
    final_state = await graph.ainvoke(initial_state, config=config)

    return final_state


# Convenience function for running specific agents in isolation (for testing)
async def run_agents_from_signals(
    signals,
    portfolio: UserPortfolio | None = None,
) -> list[FinalAlert]:
    """
    Run pipeline from detected signals (bypassing data and signal agents).

    Args:
        signals: List of DetectedSignal objects.
        portfolio: Optional user portfolio.

    Returns:
        List of FinalAlert objects.
    """
    # Create context agent and run
    context_agent = ContextAgent()
    context_state = {
        "detected_signals": signals,
        "raw_events": [],
        "enriched_signals": [],
        "reasoning_outputs": [],
        "portfolio": portfolio,
        "portfolio_impacts": [],
        "final_alerts": [],
        "error_log": [],
        "audit_trail": [],
    }

    context_result = await context_agent.run(context_state)
    context_state["enriched_signals"] = context_result.get("enriched_signals", [])

    # Run reasoning agent
    reasoning_agent = ReasoningAgent()
    reasoning_result = await reasoning_agent.run(context_state)
    context_state["reasoning_outputs"] = reasoning_result.get("reasoning_outputs", [])

    # Run portfolio agent
    portfolio_agent = PortfolioAgent()
    portfolio_result = await portfolio_agent.run(context_state)
    context_state["portfolio_impacts"] = portfolio_result.get("portfolio_impacts", [])

    # Run action agent
    action_agent = ActionAgent()
    action_result = await action_agent.run(context_state)

    return action_result.get("final_alerts", [])
