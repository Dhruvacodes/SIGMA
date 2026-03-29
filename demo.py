"""
SIGMA Demo Script.
Standalone demo for hackathon presentation.

Run with: python demo.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import scenario runner functions
from tests.scenario_runner import (
    ScenarioResult,
    run_scenario,
)

console = Console()


def print_header():
    """Print the demo header."""
    header_text = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███████╗██╗ ██████╗ ███╗   ███╗ █████╗                    ║
║   ██╔════╝██║██╔════╝ ████╗ ████║██╔══██╗                   ║
║   ███████╗██║██║  ███╗██╔████╔██║███████║                   ║
║   ╚════██║██║██║   ██║██║╚██╔╝██║██╔══██║                   ║
║   ███████║██║╚██████╔╝██║ ╚═╝ ██║██║  ██║                   ║
║   ╚══════╝╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝                   ║
║                                                              ║
║   Signal Intelligence & Guided Market Advisor                ║
║                                                              ║
║   ET AI Hackathon 2026 — Track 6: AI for the Indian Investor ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    console.print(header_text, style="bold blue")


def print_disclaimer():
    """Print the final disclaimer."""
    disclaimer = Panel(
        "[bold]DISCLAIMER:[/bold] For informational purposes only.\n"
        "Not licensed financial advice. Consult SEBI-registered\n"
        "investment advisor before acting on any recommendation.",
        title="⚠️  Important Notice",
        border_style="yellow",
        padding=(1, 2),
    )
    console.print(disclaimer)


async def seed_knowledge_base():
    """Seed the knowledge base if empty."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Seeding knowledge base...", total=None)

        try:
            from rag.knowledge_base import KnowledgeBaseSeeder
            from rag.vector_store import SigmaVectorStore

            store = SigmaVectorStore()
            if store.is_empty("historical_patterns"):
                seeder = KnowledgeBaseSeeder(store)
                seeder.seed_all()
                progress.update(task, description="Knowledge base seeded ✓")
            else:
                progress.update(task, description="Knowledge base already initialized ✓")
        except Exception as e:
            progress.update(task, description=f"Knowledge base seeding skipped: {e}")

        await asyncio.sleep(0.5)


async def run_demo():
    """Run the full demo."""
    print_header()
    console.print()

    # Seed knowledge base
    await seed_knowledge_base()
    console.print()

    # Run all 3 scenarios
    results: list[ScenarioResult] = []
    scenario_names = [
        "Bulk Deal / Distress Sell",
        "Breakout + Conflicting Signals",
        "Portfolio-Aware Prioritisation",
    ]

    for scenario_id in [1, 2, 3]:
        console.print(f"\n[bold cyan]Running Scenario {scenario_id}:[/bold cyan] {scenario_names[scenario_id-1]}")
        console.print("-" * 60)

        try:
            result = await run_scenario(scenario_id)
            results.append(result)

            # Print quick summary
            if result.alerts:
                alert = result.alerts[0]
                console.print(
                    f"  → Generated [bold]{len(result.alerts)}[/bold] alert(s) "
                    f"| Top: [bold]{alert.severity.value}[/bold] ({alert.ticker}) "
                    f"| Confidence: [bold]{alert.confidence_score:.0%}[/bold]"
                )
            else:
                console.print("  → No alerts generated")

            checks_passed = sum(1 for _, passed, _ in result.checks if passed)
            total_checks = len(result.checks)
            status = "✓" if result.all_passed else "✗"
            console.print(f"  → Checks: {checks_passed}/{total_checks} passed {status}")

        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            results.append(ScenarioResult(scenario_id, scenario_names[scenario_id-1]))

        await asyncio.sleep(2)  # Pause between scenarios

    # Print summary table
    console.print("\n" + "=" * 60)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 60 + "\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="dim", width=30)
    table.add_column("Alerts", justify="center")
    table.add_column("Top Alert", justify="left")
    table.add_column("Confidence", justify="center")
    table.add_column("Time", justify="right")

    for result in results:
        top_alert = result.top_alert
        if top_alert:
            top_alert_str = f"{top_alert.severity.value} ({top_alert.ticker[:8]})"
            conf = f"{top_alert.confidence_score:.2f}"
        else:
            top_alert_str = "None"
            conf = "N/A"

        table.add_row(
            result.name,
            str(len(result.alerts)),
            top_alert_str,
            conf,
            f"{result.execution_time:.1f}s",
        )

    console.print(table)

    # Print file locations
    console.print("\n[dim]Full audit logs written to:[/dim] ./audit_logs/")
    console.print("[dim]Alert JSON files written to:[/dim] ./scenario_outputs/")

    # Print disclaimer
    console.print()
    print_disclaimer()


def main():
    """Main entry point."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
