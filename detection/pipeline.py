import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from rich.console import Console

from config import PIPELINE_INTERVAL_SECONDS, PSI_WARNING_THRESHOLD
from detection.baseline import get_or_build_baselines
from detection.data_fetcher import fetch_all_tickers
from detection.psi_calculator import PSIResult, Severity, evaluate_all_features

logger = logging.getLogger("DriftGuardIQ.Pipeline")
console = Console()

PIPELINE_LOG_PATH = "logs/pipeline.log"
INVESTIGATION_COOLDOWN_SECONDS = 3600


def log_pipeline_cycle(ticker: str, results: List[PSIResult]) -> None:
    """Persists structured analytical drift dimensions to local disk logs.

    Args:
        ticker: The stock ticker symbol evaluated in this cycle.
        results: List of PSIResult objects from the current evaluation.
    """
    os.makedirs("logs", exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "results": [
            {
                "feature": r.feature,
                "psi_score": r.psi_score,
                "severity": r.severity.value,
                "baseline_mean": r.baseline_mean,
                "current_mean": r.current_mean,
                "deviation_pct": r.deviation_pct,
            }
            for r in results
        ],
    }

    with open(PIPELINE_LOG_PATH, "a") as log_file:
        log_file.write(json.dumps(entry) + "\n")


def print_cycle_summary(ticker: str, results: List[PSIResult]) -> None:
    """Renders highly scannable terminal reports using Rich formatting.

    Args:
        ticker: The stock ticker symbol evaluated.
        results: List of PSIResult objects from the current evaluation.
    """
    severity_colors = {
        Severity.LOW: "green",
        Severity.MEDIUM: "yellow",
        Severity.HIGH: "orange3",
        Severity.CRITICAL: "red",
    }

    timestamp = datetime.utcnow().strftime("%H:%M:%S UTC")
    console.print(f"\n[bold cyan]DriftGuard IQ[/bold cyan] — {ticker} @ {timestamp}")

    for result in results:
        color = severity_colors.get(result.severity, "white")
        console.print(
            f"  [{color}]{result.severity.value:<10}[/{color}] "
            f"{result.feature:<25} "
            f"PSI: {result.psi_score:.6f}  "
            f"Drift: {result.deviation_pct:+.2f}%"
        )


def has_drift(results: List[PSIResult]) -> bool:
    """Evaluates if any metric exceeds the active alert threshold.

    Args:
        results: List of PSIResult objects to evaluate.

    Returns:
        True if any result has a severity above the warning threshold.
    """
    return any(r.psi_score >= PSI_WARNING_THRESHOLD for r in results)


async def run_ticker_evaluation(
    ticker: str,
    baselines: Dict,
    current_data: Dict,
) -> List[PSIResult]:
    """Executes asynchronous drift analytics for an individual security.

    Args:
        ticker: The stock ticker symbol to evaluate.
        baselines: The full baseline cache dictionary.
        current_data: Dictionary of current DataFrames per ticker.

    Returns:
        A list of PSIResult objects for the evaluated ticker.
    """
    if ticker not in current_data:
        logger.error(f"Real-time feature matrix unavailable for: {ticker}")
        return []

    if ticker not in baselines:
        logger.error(f"Missing expected structural baseline cache for: {ticker}")
        return []

    results = evaluate_all_features(
        baseline=baselines[ticker],
        current_df=current_data[ticker],
        ticker=ticker,
    )

    log_pipeline_cycle(ticker, results)
    print_cycle_summary(ticker, results)

    return results


async def run_pipeline_cycle(baselines: Dict) -> Dict[str, List[PSIResult]]:
    """Coordinates parallel historical ingestion and comparative calculations.

    Args:
        baselines: The full baseline cache dictionary.

    Returns:
        A dictionary mapping each ticker to its list of PSIResult objects.
    """
    console.print("\n[bold white]Running pipeline cycle...[/bold white]")

    current_data = fetch_all_tickers()

    tasks = [
        run_ticker_evaluation(ticker, baselines, current_data)
        for ticker in baselines.keys()
    ]

    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    cycle_results: Dict[str, List[PSIResult]] = {}

    for ticker, result in zip(baselines.keys(), all_results):
        if isinstance(result, Exception):
            logger.error(f"Concurrent evaluation tree failed for {ticker}: {result}")
            continue
        cycle_results[ticker] = result

    return cycle_results


async def start_pipeline() -> None:
    """Main execution loop for continuous market tracking and orchestration.

    Loads or builds baselines on startup then runs continuous evaluation
    cycles at the configured interval. Triggers the reasoning layer when
    drift is detected in any ticker with a per ticker cooldown to prevent
    duplicate incident generation.
    """
    console.print("\n[bold green]DriftGuard IQ — Pipeline Starting[/bold green]")
    console.print(
        f"Interval: {PIPELINE_INTERVAL_SECONDS}s | "
        f"Warning threshold: {PSI_WARNING_THRESHOLD}\n"
    )

    baselines = get_or_build_baselines()
    console.print(f"[green]Baselines loaded for {len(baselines)} tickers.[/green]")

    last_investigated: Dict[str, float] = {}

    while True:
        try:
            cycle_results = await run_pipeline_cycle(baselines)

            for ticker, results in cycle_results.items():
                if has_drift(results):
                    last_time = last_investigated.get(ticker, 0.0)
                    current_time = asyncio.get_event_loop().time()
                    time_since_last = current_time - last_time

                    if time_since_last > INVESTIGATION_COOLDOWN_SECONDS:
                        console.print(
                            f"\n[bold red]DRIFT DETECTED — {ticker}[/bold red] "
                            "Triggering reasoning layer..."
                        )
                        from reasoning.orchestrator import run_investigation
                        await run_investigation(ticker, results, baselines[ticker])
                        last_investigated[ticker] = current_time
                    else:
                        remaining = int(INVESTIGATION_COOLDOWN_SECONDS - time_since_last)
                        console.print(
                            f"\n[yellow]DRIFT DETECTED — {ticker}[/yellow] "
                            f"Cooldown active — next investigation in {remaining}s"
                        )

        except Exception as error:
            logger.error(f"Pipeline root loop execution anomaly: {error}")
            console.print(f"[red]Pipeline error: {error}[/red]")

        await asyncio.sleep(PIPELINE_INTERVAL_SECONDS)