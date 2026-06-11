import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import TRACKED_TICKERS, TRACKED_FEATURES, BASELINE_WINDOW_DAYS

logger = logging.getLogger("DriftGuardIQ.DataFetcher")


def fetch_historical_data(
    ticker: str,
    days: int = BASELINE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Fetch historical OHLCV data to establish baseline reference states.

    Args:
        ticker: The stock ticker symbol to fetch data for.
        days: Number of historical days to fetch.

    Returns:
        A DataFrame with computed reference financial features.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    last_error = None

    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            raw = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                raise_errors=False,
            )

            if raw is None or raw.empty:
                raise ValueError("No data returned (DataFrame empty or None).")

            if len(raw) < 10:
                raise ValueError(f"Insufficient rows returned ({len(raw)}).")

            raw.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in raw.columns
            ]

            return compute_features(raw, ticker)

        except Exception as error:
            last_error = error
            logger.error(f"Attempt {attempt + 1} failed for {ticker}: {error}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"All retry attempts failed for ticker {ticker}. Reason: {last_error}"
    )


def fetch_latest_data(ticker: str) -> pd.DataFrame:
    """Fetch recent data for live evaluation using explicit date boundaries.
    
    This bypasses the unstable Yahoo Finance endpoint triggered by string periods.

    Args:
        ticker: The stock ticker symbol to fetch data for.

    Returns:
        A DataFrame with the latest computed financial features.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    end_date = datetime.today()
    # Using a 90-day explicit window to guarantee plenty of baseline context
    start_date = end_date - timedelta(days=90)
    last_error = None

    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            # Route request via explicit calendar filters to avoid cookie errors
            raw = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                raise_errors=False,
            )

            if raw is None or raw.empty:
                raise ValueError("No recent data returned (DataFrame empty or None).")

            if len(raw) < 25:
                raise ValueError(
                    f"Insufficient rows for rolling calculations ({len(raw)})."
                )

            raw.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in raw.columns
            ]

            return compute_features(raw, ticker)

        except Exception as error:
            last_error = error
            logger.error(
                f"Attempt {attempt + 1} failed fetching latest for {ticker}: {error}"
            )
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"All retry attempts failed fetching latest data for {ticker}. Reason: {last_error}"
    )


def compute_features(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute all tracked financial features from raw OHLCV matrix columns."""
    df = pd.DataFrame()

    df["close_price"] = raw["Close"].squeeze()
    df["volume"] = raw["Volume"].squeeze()
    df["price_change_pct"] = df["close_price"].pct_change(fill_method=None) * 100
    df["volatility_7d"] = df["price_change_pct"].rolling(window=7).std()
    df["relative_volume"] = (
        df["volume"] / df["volume"].rolling(window=20).mean()
    )

    # Clean out rolling initialization rows safely
    df.dropna(inplace=True)
    df["ticker"] = ticker

    return df[TRACKED_FEATURES + ["ticker"]]


def fetch_all_tickers() -> dict[str, pd.DataFrame]:
    """Fetch latest data matrices across all active configuration assets."""
    results = {}

    for ticker in TRACKED_TICKERS:
        try:
            results[ticker] = fetch_latest_data(ticker)
        except RuntimeError as error:
            logger.error(
                f"Skipping {ticker} due to repeated fetch failure: {error}"
            )

    return results