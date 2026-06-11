import asyncio
import logging
import os
import signal
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel

from config import (
    PIPELINE_INTERVAL_SECONDS,
    PSI_WARNING_THRESHOLD,
    PSI_CRITICAL_THRESHOLD,
    TRACKED_TICKERS,
    USE_FOUNDRY,
    FOUNDRY_IQ_DEPLOYMENT,
    OLLAMA_MODEL,
)

console = Console()

logging.basicConfig(
    filename="logs/errors.log",
    level=logging.ERROR,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)

logger = logging.getLogger(__name__)

shutdown_event = asyncio.Event()


def handle_shutdown(signum: int, frame: object) -> None:
    """Handle SIGINT and SIGTERM signals for graceful shutdown.

    Args:
        signum: The signal number received.
        frame: The current stack frame at signal receipt.
    """
    console.print("\n[yellow]Shutdown signal received. Stopping DriftGuard IQ...[/yellow]")
    shutdown_event.set()


def print_startup_banner() -> None:
    """Print the DriftGuard IQ startup banner to the terminal."""
    reasoning_engine = (
        f"Azure Foundry IQ ({FOUNDRY_IQ_DEPLOYMENT})"
        if USE_FOUNDRY
        else f"Ollama ({OLLAMA_MODEL})"
    )

    banner = (
        f"[bold cyan]DriftGuard IQ[/bold cyan] — Autonomous Financial Drift Detection\n\n"
        f"[bold]Tickers monitored:[/bold]    {', '.join(TRACKED_TICKERS)}\n"
        f"[bold]Warning threshold:[/bold]    PSI {PSI_WARNING_THRESHOLD}\n"
        f"[bold]Critical threshold:[/bold]   PSI {PSI_CRITICAL_THRESHOLD}\n"
        f"[bold]Pipeline interval:[/bold]    {PIPELINE_INTERVAL_SECONDS}s\n"
        f"[bold]Reasoning engine:[/bold]     {reasoning_engine}\n"
    )

    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))


def start_dashboard() -> subprocess.Popen:
    """Launch the Streamlit dashboard as a background subprocess.

    Returns:
        The Popen process handle for the dashboard subprocess.

    Raises:
        OSError: If the Streamlit process cannot be started.
    """
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard/app.py",
                "--server.headless",
                "true",
                "--server.port",
                "8501",
                "--server.address",
                "localhost",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print("[green]Dashboard started at http://localhost:8501[/green]")
        return process

    except OSError as error:
        logger.error(f"Failed to start dashboard: {error}")
        console.print(f"[yellow]Dashboard could not be started: {error}[/yellow]")
        raise


async def run_pipeline_until_shutdown() -> None:
    """Run the detection pipeline loop until a shutdown signal is received."""
    from detection.pipeline import start_pipeline

    console.print("[cyan]Building baselines and starting pipeline...[/cyan]")

    pipeline_task = asyncio.create_task(start_pipeline())

    while not shutdown_event.is_set():
        if pipeline_task.done():
            exception = pipeline_task.exception()
            if exception:
                logger.error(f"Pipeline task failed: {exception}")
                console.print(f"[red]Pipeline error: {exception}[/red]")
            break
        await asyncio.sleep(1)

    pipeline_task.cancel()
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass


def flush_logs() -> None:
    """Flush all log handlers to ensure pending log entries are written."""
    for handler in logging.root.handlers:
        handler.flush()


async def main() -> None:
    """Main async entry point for DriftGuard IQ.

    Starts the Streamlit dashboard subprocess and the async detection
    pipeline loop concurrently. Handles graceful shutdown on SIGINT
    or SIGTERM by cancelling all tasks and flushing logs.
    """
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs/mitigations", exist_ok=True)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    print_startup_banner()

    dashboard_process = None

    try:
        dashboard_process = start_dashboard()
    except OSError:
        console.print("[yellow]Continuing without dashboard.[/yellow]")

    try:
        await run_pipeline_until_shutdown()

    except Exception as error:
        logger.error(f"Fatal pipeline error: {error}")
        console.print(f"[red]Fatal error: {error}[/red]")

    finally:
        console.print("[yellow]Shutting down pipeline...[/yellow]")
        flush_logs()

        if dashboard_process and dashboard_process.poll() is None:
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()

        console.print("[green]DriftGuard IQ stopped cleanly.[/green]")


if __name__ == "__main__":
    asyncio.run(main())