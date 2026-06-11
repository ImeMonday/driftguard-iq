import json
import logging
import os
import tempfile
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from action.schema import IncidentReport, SeverityLevel, DriftedFeature, build_recommended_action
from detection.psi_calculator import PSIResult
from reasoning.agent_isolation import IsolationFinding
from reasoning.agent_rootcause import RootCauseFinding
from reasoning.agent_impact import RevenueImpactFinding

logger = logging.getLogger(__name__)
console = Console()

INCIDENTS_LOG_PATH = "logs/incidents.json"

SEVERITY_COLORS = {
    SeverityLevel.LOW: "green",
    SeverityLevel.MEDIUM: "yellow",
    SeverityLevel.HIGH: "orange3",
    SeverityLevel.CRITICAL: "red",
}


def build_drifted_features(
    psi_results: list[PSIResult],
    isolation_finding: IsolationFinding,
) -> list[DriftedFeature]:
    """Build a list of DriftedFeature objects from PSI results.

    Args:
        psi_results: Full list of PSIResult objects from the detection layer.
        isolation_finding: The findings from the isolation agent.

    Returns:
        A list of DriftedFeature objects for all drifted features.
    """
    drifted = []

    for result in psi_results:
        if result.feature in isolation_finding.drifted_features:
            drifted.append(
                DriftedFeature(
                    feature_name=result.feature,
                    psi_score=result.psi_score,
                    severity=SeverityLevel(result.severity.value),
                    deviation_pct=result.deviation_pct,
                    baseline_mean=result.baseline_mean,
                    current_mean=result.current_mean,
                )
            )

    return drifted


def append_incident_to_log(incident: IncidentReport) -> None:
    """Atomically append an incident report to the incidents log file.

    Args:
        incident: The validated IncidentReport to persist.

    Raises:
        OSError: If the file cannot be written after atomic replace.
    """
    os.makedirs("logs", exist_ok=True)

    existing_incidents = []

    if os.path.exists(INCIDENTS_LOG_PATH) and os.path.getsize(INCIDENTS_LOG_PATH) > 0:
        try:
            with open(INCIDENTS_LOG_PATH, "r") as log_file:
                existing_incidents = json.load(log_file)
        except (json.JSONDecodeError, ValueError) as error:
            logger.error(f"Failed to parse existing incidents log: {error}")
            existing_incidents = []

    existing_incidents.append(incident.model_dump())

    dir_name = os.path.dirname(INCIDENTS_LOG_PATH)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name)

    try:
        with os.fdopen(tmp_fd, "w") as tmp_file:
            json.dump(existing_incidents, tmp_file, indent=2)

        os.replace(tmp_path, INCIDENTS_LOG_PATH)

    except OSError as error:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"Failed to write incident log atomically: {error}")
        raise


def render_incident_terminal(incident: IncidentReport) -> None:
    """Render a formatted rich terminal summary for an incident report.

    Args:
        incident: The validated IncidentReport to render.
    """
    severity_color = SEVERITY_COLORS.get(incident.severity, "white")

    header = (
        f"[bold {severity_color}]{incident.severity.value}[/bold {severity_color}] "
        f"Incident — {incident.ticker} — {incident.incident_id}"
    )

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Field", style="dim", width=28)
    table.add_column("Value")

    table.add_row("Timestamp", incident.timestamp)
    table.add_row("Ticker", incident.ticker)
    table.add_row("Top Drifted Feature", incident.top_feature)
    table.add_row("Top PSI Score", f"{incident.top_psi_score:.6f}")
    table.add_row(
        "Severity",
        f"[{severity_color}]{incident.severity.value}[/{severity_color}]",
    )
    table.add_row(
        "Revenue at Risk",
        f"[bold red]${incident.revenue_at_risk_usd:,.2f}[/bold red]",
    )
    table.add_row("Confidence Interval", incident.confidence_interval)
    table.add_row(
        "Root Cause Confidence",
        f"{incident.root_cause_confidence:.2f}",
    )
    table.add_row("Price Anomaly", str(incident.price_anomaly_detected))
    table.add_row("Volume Anomaly", str(incident.volume_anomaly_detected))
    table.add_row("Reasoning Engine", incident.reasoning_engine)

    console.print(Panel(header, border_style=severity_color))
    console.print(table)

    console.print(
        Panel(
            incident.root_cause_hypothesis,
            title="Root Cause Hypothesis",
            border_style="cyan",
        )
    )

    console.print(
        Panel(
            incident.impact_summary,
            title="Executive Impact Summary",
            border_style="magenta",
        )
    )

    console.print(
        Panel(
            f"[bold]{incident.recommended_action}[/bold]",
            title="Recommended Action",
            border_style=severity_color,
        )
    )

    evidence_text = "\n".join(
        f"  {i + 1}. {e}" for i, e in enumerate(incident.supporting_evidence)
    )
    console.print(
        Panel(
            evidence_text,
            title="Supporting Evidence",
            border_style="white",
        )
    )

    if incident.mitigation_script_path:
        console.print(
            f"\n[green]Mitigation script generated:[/green] {incident.mitigation_script_path}"
        )


async def assemble_and_report(
    ticker: str,
    psi_results: list[PSIResult],
    isolation_finding: IsolationFinding,
    rootcause_finding: RootCauseFinding,
    impact_finding: RevenueImpactFinding,
) -> IncidentReport:
    """Assemble all agent findings into a validated incident report and persist it.

    Args:
        ticker: The stock ticker symbol where drift was detected.
        psi_results: Full list of PSIResult objects from the detection layer.
        isolation_finding: The findings from the isolation agent.
        rootcause_finding: The findings from the root cause agent.
        impact_finding: The findings from the revenue impact agent.

    Returns:
        A validated IncidentReport instance.

    Raises:
        ValueError: If the incident report fails Pydantic validation.
    """
    from action.mitigation_generator import generate_mitigation_script
    from config import USE_FOUNDRY, FOUNDRY_IQ_DEPLOYMENT, OLLAMA_MODEL

    drifted_features = build_drifted_features(psi_results, isolation_finding)

    recommended_action = build_recommended_action(
        severity=SeverityLevel(
            isolation_finding.severity_map.get(isolation_finding.top_feature, "LOW")
        ),
        top_feature=isolation_finding.top_feature,
        ticker=ticker,
    )

    reasoning_engine = (
        f"Azure Foundry IQ ({FOUNDRY_IQ_DEPLOYMENT})"
        if USE_FOUNDRY
        else f"Ollama ({OLLAMA_MODEL})"
    )

    incident = IncidentReport(
        ticker=ticker,
        drifted_features=drifted_features,
        top_feature=isolation_finding.top_feature,
        top_psi_score=isolation_finding.top_psi_score,
        severity=SeverityLevel(
            isolation_finding.severity_map.get(isolation_finding.top_feature, "LOW")
        ),
        root_cause_hypothesis=rootcause_finding.hypothesis,
        root_cause_confidence=rootcause_finding.confidence_score,
        price_anomaly_detected=rootcause_finding.price_anomaly_detected,
        volume_anomaly_detected=rootcause_finding.volume_anomaly_detected,
        revenue_at_risk_usd=impact_finding.revenue_at_risk_usd,
        confidence_interval=impact_finding.confidence_interval,
        degradation_coefficient=impact_finding.degradation_coefficient,
        recommended_action=recommended_action,
        supporting_evidence=rootcause_finding.supporting_evidence,
        impact_summary=impact_finding.impact_summary,
        reasoning_engine=reasoning_engine,
    )

    mitigation_path = generate_mitigation_script(incident, isolation_finding)
    incident = incident.model_copy(update={"mitigation_script_path": mitigation_path})

    append_incident_to_log(incident)
    render_incident_terminal(incident)

    return incident