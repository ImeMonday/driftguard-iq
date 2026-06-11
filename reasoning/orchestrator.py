import asyncio
import logging
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from detection.psi_calculator import PSIResult
from reasoning.agent_isolation import run_isolation, IsolationFinding
from reasoning.agent_rootcause import run_rootcause, RootCauseFinding
from reasoning.agent_impact import run_impact, RevenueImpactFinding

logger = logging.getLogger(__name__)
console = Console()


async def run_with_retry(agent_func, *args, retries: int = 3) -> object:
    """Run an agent function with exponential backoff retry logic.

    Args:
        agent_func: The async agent function to execute.
        *args: Arguments to pass to the agent function.
        retries: Number of retry attempts before raising.

    Returns:
        The result of the agent function on success.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    for attempt in range(retries):
        try:
            return await agent_func(*args)
        except Exception as error:
            logger.error(
                f"Agent {agent_func.__name__} attempt {attempt + 1} failed: {error}"
            )
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(
        f"Agent {agent_func.__name__} failed after {retries} attempts"
    )


async def run_investigation(
    ticker: str,
    psi_results: list[PSIResult],
    baseline: dict,
) -> dict:
    """Orchestrate the full three-agent investigation chain for a drift event.

    Args:
        ticker: The stock ticker symbol where drift was detected.
        psi_results: List of PSIResult objects from the detection layer.
        baseline: The baseline cache dictionary for this ticker.

    Returns:
        A dictionary containing the full investigation findings and incident report.
    """
    console.print(
        Panel(
            f"[bold yellow]Investigation Started[/bold yellow]\n"
            f"Ticker: [cyan]{ticker}[/cyan]\n"
            f"Drifted features: {sum(1 for r in psi_results if r.psi_score >= 0.1)}\n"
            f"Timestamp: {datetime.utcnow().isoformat()}",
            title="DriftGuard IQ — Reasoning Layer",
            border_style="yellow",
        )
    )

    try:
        console.print("\n[bold white]Agent 1 — Feature Isolation[/bold white]")
        isolation_finding: IsolationFinding = await run_with_retry(
            run_isolation, ticker, psi_results
        )
        console.print(
            f"[green]Isolation complete.[/green] "
            f"Top drifted feature: [cyan]{isolation_finding.top_feature}[/cyan] "
            f"PSI: {isolation_finding.top_psi_score:.6f}"
        )

    except RuntimeError as error:
        logger.error(f"Isolation agent failed for {ticker}: {error}")
        console.print(f"[red]Isolation agent failed: {error}[/red]")
        return {"status": "failed", "stage": "isolation", "ticker": ticker}

    try:
        console.print("\n[bold white]Agent 2 — Root Cause Analysis[/bold white]")
        rootcause_finding: RootCauseFinding = await run_with_retry(
            run_rootcause, ticker, isolation_finding, baseline
        )
        console.print(
            f"[green]Root cause identified.[/green] "
            f"Confidence: [cyan]{rootcause_finding.confidence_score:.2f}[/cyan]\n"
            f"Hypothesis: {rootcause_finding.hypothesis}"
        )

    except RuntimeError as error:
        logger.error(f"Root cause agent failed for {ticker}: {error}")
        console.print(f"[red]Root cause agent failed: {error}[/red]")
        return {"status": "failed", "stage": "rootcause", "ticker": ticker}

    try:
        console.print("\n[bold white]Agent 3 — Revenue Impact Scoring[/bold white]")
        impact_finding: RevenueImpactFinding = await run_with_retry(
            run_impact, ticker, isolation_finding, rootcause_finding
        )
        console.print(
            f"[green]Impact scored.[/green] "
            f"Revenue at risk: [red]${impact_finding.revenue_at_risk_usd:,.2f}[/red] "
            f"({impact_finding.confidence_interval})"
        )

    except RuntimeError as error:
        logger.error(f"Impact agent failed for {ticker}: {error}")
        console.print(f"[red]Impact agent failed: {error}[/red]")
        return {"status": "failed", "stage": "impact", "ticker": ticker}

    from action.incident_reporter import assemble_and_report
    incident = await assemble_and_report(
        ticker=ticker,
        psi_results=psi_results,
        isolation_finding=isolation_finding,
        rootcause_finding=rootcause_finding,
        impact_finding=impact_finding,
    )

    return {
        "status": "success",
        "ticker": ticker,
        "incident": incident,
    }