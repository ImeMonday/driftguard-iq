import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class SeverityLevel(str, Enum):
    """Severity levels for drift incidents."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DriftedFeature(BaseModel):
    """Represents a single drifted feature and its metrics.

    Attributes:
        feature_name: The name of the drifted feature.
        psi_score: The Population Stability Index score.
        severity: The severity level of the drift.
        deviation_pct: Percentage deviation from the baseline mean.
        baseline_mean: The mean of the baseline distribution.
        current_mean: The mean of the current distribution.
    """

    model_config = {"strict": True}

    feature_name: str
    psi_score: float
    severity: SeverityLevel
    deviation_pct: float
    baseline_mean: float
    current_mean: float

    @field_validator("psi_score")
    @classmethod
    def psi_must_be_positive(cls, value: float) -> float:
        """Validate that PSI score is a positive number.

        Args:
            value: The PSI score to validate.

        Returns:
            The validated PSI score.

        Raises:
            ValueError: If the PSI score is negative.
        """
        if value < 0:
            raise ValueError(f"PSI score must be non-negative, got {value}")
        return value


class IncidentReport(BaseModel):
    """Strict Pydantic v2 schema for a DriftGuard IQ incident report.

    Attributes:
        incident_id: Unique identifier for the incident.
        timestamp: UTC timestamp when the incident was detected.
        ticker: The stock ticker symbol where drift was detected.
        drifted_features: List of features that have drifted.
        top_feature: The feature with the highest PSI score.
        top_psi_score: The PSI score of the top drifted feature.
        severity: Overall incident severity based on top feature.
        root_cause_hypothesis: Natural language root cause from agent two.
        root_cause_confidence: Confidence score of the root cause hypothesis.
        price_anomaly_detected: Whether an unusual price move was found.
        volume_anomaly_detected: Whether an unusual volume spike was found.
        revenue_at_risk_usd: Estimated revenue at risk in US dollars.
        confidence_interval: Human readable revenue confidence interval.
        degradation_coefficient: Model degradation coefficient applied.
        recommended_action: Actionable mitigation recommendation.
        supporting_evidence: List of evidence points for the incident.
        impact_summary: Executive summary of financial impact.
        mitigation_script_path: Path to the auto-generated mitigation script.
        reasoning_engine: The AI engine used for reasoning.
    """

    model_config = {"strict": True}

    incident_id: str = Field(
        default_factory=lambda: f"DG-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    ticker: str
    drifted_features: list[DriftedFeature]
    top_feature: str
    top_psi_score: float
    severity: SeverityLevel
    root_cause_hypothesis: str
    root_cause_confidence: float
    price_anomaly_detected: bool
    volume_anomaly_detected: bool
    revenue_at_risk_usd: float
    confidence_interval: str
    degradation_coefficient: float
    recommended_action: str
    supporting_evidence: list[str]
    impact_summary: str
    mitigation_script_path: str | None = None
    reasoning_engine: str

    @field_validator("root_cause_confidence")
    @classmethod
    def confidence_must_be_valid(cls, value: float) -> float:
        """Validate that confidence score is between 0.0 and 1.0.

        Args:
            value: The confidence score to validate.

        Returns:
            The validated confidence score.

        Raises:
            ValueError: If the confidence score is outside valid range.
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Confidence score must be between 0.0 and 1.0, got {value}"
            )
        return value

    @field_validator("revenue_at_risk_usd")
    @classmethod
    def revenue_must_be_positive(cls, value: float) -> float:
        """Validate that revenue at risk is a positive number.

        Args:
            value: The revenue at risk value to validate.

        Returns:
            The validated revenue at risk value.

        Raises:
            ValueError: If the revenue at risk is negative.
        """
        if value < 0:
            raise ValueError(
                f"Revenue at risk must be non-negative, got {value}"
            )
        return value

    @model_validator(mode="after")
    def top_feature_must_exist_in_drifted(self) -> "IncidentReport":
        """Validate that top feature exists in the drifted features list.

        Returns:
            The validated IncidentReport instance.

        Raises:
            ValueError: If the top feature is not in the drifted features list.
        """
        feature_names = [f.feature_name for f in self.drifted_features]
        if self.top_feature not in feature_names:
            raise ValueError(
                f"Top feature {self.top_feature} not found in drifted features list"
            )
        return self


def build_recommended_action(
    severity: SeverityLevel,
    top_feature: str,
    ticker: str,
) -> str:
    """Generate a recommended action string based on severity and context.

    Args:
        severity: The overall incident severity level.
        top_feature: The feature with the highest PSI score.
        ticker: The stock ticker symbol affected.

    Returns:
        A human readable recommended action string.
    """
    actions = {
        SeverityLevel.LOW: (
            f"Monitor {top_feature} for {ticker} over the next 24 hours. "
            f"No immediate action required."
        ),
        SeverityLevel.MEDIUM: (
            f"Investigate {top_feature} distribution shift for {ticker}. "
            f"Consider revalidating model inputs within 48 hours."
        ),
        SeverityLevel.HIGH: (
            f"Immediate review of {top_feature} pipeline for {ticker} required. "
            f"Retrain or recalibrate affected models within 24 hours."
        ),
        SeverityLevel.CRITICAL: (
            f"CRITICAL: Halt automated decisions dependent on {top_feature} for {ticker}. "
            f"Escalate to engineering and risk teams immediately."
        ),
    }

    return actions.get(severity, "Review drift report and take appropriate action.")