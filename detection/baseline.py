import json
import logging
import os
import tempfile
from datetime import datetime

import numpy as np
from rich.console import Console

from config import TRACKED_TICKERS, TRACKED_FEATURES, BASELINE_WINDOW_DAYS
from detection.data_fetcher import fetch_historical_data

logger = logging.getLogger(__name__)
console = Console()

BASELINE_CACHE_PATH = "data/baseline_cache.json"


def build_baseline(ticker: str) -> dict:
    """Build a baseline distribution for all tracked features for a given ticker.

    Args:
        ticker: The stock ticker symbol to build a baseline for.

    Returns:
        A dictionary containing bin edges and statistics per feature.
    """
    df = fetch_historical_data(ticker, days=BASELINE_WINDOW_DAYS)

    baseline = {
        "ticker": ticker,
        "created_at": datetime.utcnow().isoformat(),
        "features": {},
    }

    for feature in TRACKED_FEATURES:
        values = df[feature].dropna().tolist()

        if len(values) < 10:
            logger.error(
                f"Insufficient data for baseline feature {feature} on {ticker}"
            )
            continue

        arr = np.array(values)
        bin_edges = np.percentile(arr, np.linspace(0, 100, 11)).tolist()

        baseline["features"][feature] = {
            "bin_edges": bin_edges,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sample_size": len(values),
        }

    return baseline


def save_baseline(baselines: dict) -> None:
    """Atomically save all ticker baselines to the cache file.

    Args:
        baselines: Dictionary mapping ticker symbols to their baseline data.
    """
    dir_name = os.path.dirname(BASELINE_CACHE_PATH)
    os.makedirs(dir_name, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name)

    try:
        with os.fdopen(tmp_fd, "w") as tmp_file:
            json.dump(baselines, tmp_file, indent=2)

        os.replace(tmp_path, BASELINE_CACHE_PATH)

    except Exception as error:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"Failed to save baseline cache: {error}")
        raise


def load_baseline() -> dict:
    """Load the baseline cache from disk.

    Returns:
        A dictionary of all cached ticker baselines.

    Raises:
        FileNotFoundError: If the baseline cache does not exist.
        ValueError: If the baseline cache is empty or malformed.
    """
    if not os.path.exists(BASELINE_CACHE_PATH):
        raise FileNotFoundError(
            f"Baseline cache not found at {BASELINE_CACHE_PATH}."
        )

    if os.path.getsize(BASELINE_CACHE_PATH) == 0:
        raise ValueError("Baseline cache file is empty.")

    with open(BASELINE_CACHE_PATH, "r") as cache_file:
        data = json.load(cache_file)

    if not data:
        raise ValueError("Baseline cache contains no data.")

    return data


def build_all_baselines() -> dict:
    """Build and cache baselines for all tracked tickers.

    Returns:
        A dictionary mapping all ticker symbols to their baseline data.
    """
    baselines = {}

    for ticker in TRACKED_TICKERS:
        try:
            console.print(f"[cyan]Building baseline for {ticker}...[/cyan]")
            baselines[ticker] = build_baseline(ticker)
            console.print(f"[green]Baseline complete for {ticker}.[/green]")
        except Exception as error:
            logger.error(f"Failed to build baseline for {ticker}: {error}")
            console.print(f"[red]Failed to build baseline for {ticker}: {error}[/red]")

    save_baseline(baselines)
    console.print(
        f"[bold green]All baselines saved for {len(baselines)} tickers.[/bold green]"
    )

    return baselines


def get_or_build_baselines() -> dict:
    """Load baseline from cache or build it if cache does not exist or is empty.

    Returns:
        A dictionary of all ticker baselines.
    """
    try:
        data = load_baseline()
        if not data:
            raise ValueError("Empty baseline cache.")
        console.print(
            f"[green]Baselines loaded from cache for {len(data)} tickers.[/green]"
        )
        return data
    except (FileNotFoundError, ValueError):
        console.print(
            "[yellow]No valid baseline cache found. Building from scratch...[/yellow]"
        )
        return build_all_baselines()