import logging
import time
from dataclasses import dataclass

import ollama
import yfinance as yf
from openai import AzureOpenAI

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    FOUNDRY_IQ_ENDPOINT,
    FOUNDRY_IQ_API_KEY,
    FOUNDRY_IQ_DEPLOYMENT,
    FOUNDRY_IQ_API_VERSION,
    USE_FOUNDRY,
)
from reasoning.agent_isolation import IsolationFinding

logger = logging.getLogger(__name__)


@dataclass
class RootCauseFinding:
    """Holds the result of the root cause analysis agent.

    Attributes:
        ticker: The stock ticker symbol investigated.
        hypothesis: Natural language root cause hypothesis.
        confidence_score: Confidence level between 0.0 and 1.0.
        price_anomaly_detected: Whether an unusual price move was found.
        volume_anomaly_detected: Whether an unusual volume spike was found.
        recent_price_change_pct: Most recent single day price change percentage.
        avg_volume_ratio: Ratio of recent volume to 20 day average volume.
        supporting_evidence: List of evidence points supporting the hypothesis.
    """

    ticker: str
    hypothesis: str
    confidence_score: float
    price_anomaly_detected: bool
    volume_anomaly_detected: bool
    recent_price_change_pct: float
    avg_volume_ratio: float
    supporting_evidence: list[str]


def fetch_recent_market_context(ticker: str) -> dict:
    """Fetch recent price action and volume data to support root cause analysis.

    Args:
        ticker: The stock ticker symbol to fetch context for.

    Returns:
        A dictionary containing recent market statistics and anomaly flags.

    Raises:
        RuntimeError: If market data cannot be fetched after retries.
    """
    from datetime import datetime, timedelta

    end_date = datetime.today()
    start_date = end_date - timedelta(days=15)

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
                raise ValueError(f"No market context data returned for {ticker}")

            close = raw["Close"].squeeze()
            volume = raw["Volume"].squeeze()

            price_changes = close.pct_change(fill_method=None).dropna() * 100
            recent_price_change = float(price_changes.iloc[-1])

            avg_volume = float(volume.rolling(window=5).mean().iloc[-1])
            latest_volume = float(volume.iloc[-1])
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0

            price_anomaly = abs(recent_price_change) > 3.0
            volume_anomaly = volume_ratio > 2.0

            return {
                "recent_price_change_pct": round(recent_price_change, 4),
                "volume_ratio": round(volume_ratio, 4),
                "price_anomaly_detected": price_anomaly,
                "volume_anomaly_detected": volume_anomaly,
            }

        except Exception as error:
            logger.error(
                f"Attempt {attempt + 1} failed fetching market context for {ticker}: {error}"
            )
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch market context for {ticker} after 3 attempts")


def build_rootcause_prompt(
    ticker: str,
    isolation_finding: IsolationFinding,
    baseline: dict,
    market_context: dict,
) -> str:
    """Build the prompt for the root cause analysis agent.

    Args:
        ticker: The stock ticker symbol being investigated.
        isolation_finding: The findings from the isolation agent.
        baseline: The baseline cache dictionary for this ticker.
        market_context: Recent market statistics for the ticker.

    Returns:
        A formatted prompt string for the model.
    """
    baseline_mean = baseline.get("features", {}).get(
        isolation_finding.top_feature, {}
    ).get("mean", "unknown")

    return (
        f"You are a senior quantitative analyst investigating a data drift incident "
        f"for the financial stock {ticker}.\n\n"
        f"Drift Summary:\n"
        f"Top drifted feature: {isolation_finding.top_feature}\n"
        f"PSI score: {isolation_finding.top_psi_score:.6f}\n"
        f"Deviation from baseline: "
        f"{isolation_finding.deviation_map.get(isolation_finding.top_feature, 0):+.2f}%\n"
        f"Baseline mean: {baseline_mean}\n\n"
        f"Recent Market Context:\n"
        f"Latest price change: {market_context['recent_price_change_pct']:+.2f}%\n"
        f"Volume ratio vs average: {market_context['volume_ratio']:.2f}x\n"
        f"Price anomaly detected: {market_context['price_anomaly_detected']}\n"
        f"Volume anomaly detected: {market_context['volume_anomaly_detected']}\n\n"
        f"Agent summary from isolation: {isolation_finding.summary}\n\n"
        f"Based on this evidence, provide a concise root cause hypothesis in two "
        f"sentences. State the most likely cause of the drift and rate your "
        f"confidence as a decimal between 0.0 and 1.0 on the last line in the "
        f"format: CONFIDENCE: 0.00"
    )


def parse_confidence(response_text: str) -> float:
    """Extract the confidence score from the model response text.

    Args:
        response_text: The raw text response from the model.

    Returns:
        A float confidence score between 0.0 and 1.0.
    """
    for line in reversed(response_text.strip().split("\n")):
        if "CONFIDENCE:" in line.upper():
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                pass

    return 0.5


def build_supporting_evidence(
    isolation_finding: IsolationFinding,
    market_context: dict,
) -> list[str]:
    """Assemble a list of evidence points from isolation and market context.

    Args:
        isolation_finding: The findings from the isolation agent.
        market_context: Recent market statistics for the ticker.

    Returns:
        A list of human readable evidence strings.
    """
    evidence = []

    for feature in isolation_finding.drifted_features:
        psi = isolation_finding.psi_scores[feature]
        deviation = isolation_finding.deviation_map[feature]
        evidence.append(
            f"{feature} drifted with PSI {psi:.4f} and {deviation:+.2f}% deviation from baseline"
        )

    if market_context["price_anomaly_detected"]:
        evidence.append(
            f"Unusual price movement detected: "
            f"{market_context['recent_price_change_pct']:+.2f}% in latest session"
        )

    if market_context["volume_anomaly_detected"]:
        evidence.append(
            f"Volume spike detected: {market_context['volume_ratio']:.2f}x above recent average"
        )

    return evidence


def call_model(prompt: str) -> str:
    """Call the configured AI model with a prompt.

    Args:
        prompt: The prompt string to send to the model.

    Returns:
        The model response text.

    Raises:
        RuntimeError: If the model call fails.
    """
    if USE_FOUNDRY:
        client = AzureOpenAI(
            azure_endpoint=FOUNDRY_IQ_ENDPOINT,
            api_key=FOUNDRY_IQ_API_KEY,
            api_version=FOUNDRY_IQ_API_VERSION,
        )
        response = client.chat.completions.create(
            model=FOUNDRY_IQ_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    else:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()


async def run_rootcause(
    ticker: str,
    isolation_finding: IsolationFinding,
    baseline: dict,
) -> RootCauseFinding:
    """Run the root cause analysis agent to identify why drift occurred.

    Args:
        ticker: The stock ticker symbol being investigated.
        isolation_finding: The findings from the isolation agent.
        baseline: The baseline cache dictionary for this ticker.

    Returns:
        A RootCauseFinding dataclass with hypothesis and supporting evidence.

    Raises:
        RuntimeError: If the market context fetch or model call fails.
    """
    market_context = fetch_recent_market_context(ticker)

    prompt = build_rootcause_prompt(
        ticker=ticker,
        isolation_finding=isolation_finding,
        baseline=baseline,
        market_context=market_context,
    )

    try:
        response_text = call_model(prompt)
    except Exception as error:
        logger.error(f"Root cause agent failed for {ticker}: {error}")
        raise RuntimeError(f"Root cause agent model call failed: {error}")

    confidence = parse_confidence(response_text)
    hypothesis = "\n".join([
        line for line in response_text.split("\n")
        if "CONFIDENCE:" not in line.upper()
    ]).strip()

    evidence = build_supporting_evidence(isolation_finding, market_context)

    return RootCauseFinding(
        ticker=ticker,
        hypothesis=hypothesis,
        confidence_score=min(max(confidence, 0.0), 1.0),
        price_anomaly_detected=market_context["price_anomaly_detected"],
        volume_anomaly_detected=market_context["volume_anomaly_detected"],
        recent_price_change_pct=market_context["recent_price_change_pct"],
        avg_volume_ratio=market_context["volume_ratio"],
        supporting_evidence=evidence,
    )