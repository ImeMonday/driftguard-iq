import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np

from config import SEVERITY_LEVELS

logger = logging.getLogger("DriftGuardIQ.Calculator")

EPSILON = 0.0001


class Severity(str, Enum):
    """Structural severity classifications for numerical data drift."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class PSIResult:
    """Encapsulates the complete statistical evaluation state of a feature."""

    ticker: str
    feature: str
    psi_score: float
    severity: Severity
    bin_breakdown: List[float]
    baseline_mean: float
    current_mean: float
    deviation_pct: float


def resolve_severity(psi_score: float) -> Severity:
    """Maps a calculated PSI score to its corresponding Severity band."""
    for level, (lower, upper) in SEVERITY_LEVELS.items():
        if lower <= psi_score < upper:
            return Severity[level]

    return Severity.CRITICAL


def calculate_psi(
    baseline_data: Dict,
    current_values: List[float],
    ticker: str,
    feature: str,
) -> PSIResult:
    """Computes the Population Stability Index against baseline quantiles.

    Formula used:
    $$PSI = \\sum \\left( P_{current} - P_{baseline} \\right) \\times 
    \\ln\\left(\\frac{P_{current}}{P_{baseline}}\\right)$$
    """
    if not current_values:
        raise ValueError(f"Zero observations provided for feature: {feature}")

    bin_edges = baseline_data.get("bin_edges")
    baseline_mean = baseline_data.get("mean")

    if not bin_edges or baseline_mean is None:
        raise ValueError(f"Malformed baseline profile definition for: {feature}")

    bin_edges = np.array(bin_edges, dtype=float)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    n_bins = len(bin_edges) - 1
    # Equi-frequency deciles guarantee uniform 10% distribution per reference bin
    baseline_proportions = np.full(n_bins, 1.0 / n_bins)

    current_arr = np.array(current_values, dtype=float)
    current_mean = float(np.mean(current_arr))

    current_counts = np.histogram(current_arr, bins=bin_edges)[0]
    total_current = len(current_arr)

    if total_current == 0:
        raise ValueError(f"No current valid observations for {feature}")

    current_proportions = current_counts / total_current

    # Clip distributions to mitigate division-by-zero or log-of-zero operations
    baseline_proportions = np.clip(baseline_proportions, EPSILON, None)
    current_proportions = np.clip(current_proportions, EPSILON, None)

    bin_psi_values = (current_proportions - baseline_proportions) * np.log(
        current_proportions / baseline_proportions
    )

    psi_score = float(np.sum(bin_psi_values))
    abs_psi = abs(psi_score)

    if baseline_mean != 0:
        deviation_pct = ((current_mean - baseline_mean) / abs(baseline_mean)) * 100
    else:
        deviation_pct = 0.0

    return PSIResult(
        ticker=ticker,
        feature=feature,
        psi_score=round(abs_psi, 6),
        severity=resolve_severity(abs_psi),
        bin_breakdown=bin_psi_values.tolist(),
        baseline_mean=round(baseline_mean, 6),
        current_mean=round(current_mean, 6),
        deviation_pct=round(deviation_pct, 4),
    )


def evaluate_all_features(
    baseline: Dict,
    current_df,
    ticker: str,
) -> List[PSIResult]:
    """Evaluates stability metrics across all active tracking features."""
    results: List[PSIResult] = []

    for feature, feature_baseline in baseline.get("features", {}).items():
        if feature not in current_df.columns:
            logger.error(f"Target column missing in real-time matrix: {feature}")
            continue

        current_values = current_df[feature].dropna().tolist()

        if not current_values:
            logger.error(f"No valid observations present for metric: {feature}")
            continue

        try:
            result = calculate_psi(
                baseline_data=feature_baseline,
                current_values=current_values,
                ticker=ticker,
                feature=feature,
            )
            results.append(result)
        except ValueError as error:
            logger.error(f"PSI boundary constraint execution failed: {error}")

    return results