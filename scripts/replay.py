import argparse
import asyncio
import json
import logging
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()
logger = logging.getLogger(__name__)

INCIDENTS_LOG_PATH = "logs/incidents.json"


def load_incidents() -> list[dict]:
    """Load all incidents from the incidents log file.

    Returns:
        A list of incident dictionaries loaded from disk.

    Raises:
        FileNotFoundError: If the incidents log does not exist.
        ValueError: If the incidents log cannot be parsed.
    """
    if not os.path.exists(INCIDENTS_LOG_PATH):
        raise FileNotFoundError(
            f"Incidents log not found at {INCIDENTS_LOG_PATH}. "
            f"Run the pipeline first to generate incidents."
        )

    if os.path.getsize(INCIDENTS_LOG_PATH) == 0:
        raise ValueError("Incidents log is empty. No incidents to replay.")

    with open(INCIDENTS_LOG_PATH, "r") as f:
        return json.load(f)


def select_incident_interactive(incidents: list[dict]) -> dict:
    """Present an interactive numbered menu for incident selection.

    Args:
        incidents: List of incident dictionaries to choose from.

    Returns:
        The selected incident dictionary.
    """
    console.print(
        Panel(
            "[bold cyan]DriftGuard IQ — Incident Replay[/bold cyan]\n"
            "Select an incident to replay the full agent reasoning chain.",
            border_style="cyan",
        )
    )

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("No.", width=5)
    table.add_column("Incident ID", width=30)
    table.add_column("Ticker", width=8)
    table.add_column("Severity", width=10)
    table.add_column("Top Feature", width=25)
    table.add_column("Revenue at Risk", width=18)
    table.add_column("Timestamp", width=22)

    severity_colors = {
        "CRITICAL": "red",
        "HIGH": "orange3",
        "MEDIUM": "yellow",
        "LOW": "green",
    }

    for i, incident in enumerate(incidents, 1):
        severity = incident.get("severity", "LOW")
        color = severity_colors.get(severity, "white")

        table.add_row(
            str(i),
            incident.get("incident_id", ""),
            incident.get("ticker", ""),
            f"[{color}]{severity}[/{color}]",
            incident.get("top_feature", ""),
            f"${incident.get('revenue_at_risk_usd', 0):,.2f}",
            incident.get("timestamp", "")[:19].replace("T", " "),
        )

    console.print(table)

    while True:
        try:
            choice = int(console.input(
                f"\n[bold cyan]Enter incident number (1-{len(incidents)}):[/bold cyan] "
            ))
            if 1 <= choice <= len(incidents):
                return incidents[choice - 1]
            console.print(f"[red]Please enter a number between 1 and {len(incidents)}[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")


def find_incident_by_id(incidents: list[dict], incident_id: str) -> dict:
    """Find a specific incident by its ID.

    Args:
        incidents: List of all incident dictionaries.
        incident_id: The incident ID to search for.

    Returns:
        The matching incident dictionary.

    Raises:
        ValueError: If no incident with the given ID is found.
    """
    for incident in incidents:
        if incident.get("incident_id") == incident_id:
            return incident

    raise ValueError(f"Incident {incident_id} not found in log.")


def render_agent_panel(
    agent_number: int,
    agent_name: str,
    content: str,
    border_color: str,
) -> None:
    """Render a rich panel for a single agent replay step.

    Args:
        agent_number: The sequential number of the agent in the chain.
        agent_name: The display name of the agent.
        content: The content to display inside the panel.
        border_color: The rich color string for the panel border.
    """
    console.print(
        Panel(
            content,
            title=f"[bold]Agent {agent_number} — {agent_name}[/bold]",
            border_style=border_color,
            padding=(1, 2),
        )
    )


def replay_isolation_agent(incident: dict) -> None:
    """Replay the feature isolation agent step from stored incident data.

    Args:
        incident: The incident dictionary containing isolation findings.
    """
    drifted_features = incident.get("drifted_features", [])

    feature_lines = "\n".join([
        f"  {f['feature_name']:<25} PSI: {f['psi_score']:.6f}  "
        f"Severity: {f['severity']}  Deviation: {f['deviation_pct']:+.2f}%"
        for f in sorted(drifted_features, key=lambda x: x["psi_score"], reverse=True)
    ])

    content = (
        f"[bold]Ticker:[/bold] {incident.get('ticker')}\n"
        f"[bold]Drifted Features:[/bold]\n{feature_lines}\n\n"
        f"[bold]Top Feature:[/bold] [cyan]{incident.get('top_feature')}[/cyan]\n"
        f"[bold]Top PSI Score:[/bold] {incident.get('top_psi_score'):.6f}"
    )

    render_agent_panel(1, "Feature Isolation", content, "cyan")


def replay_rootcause_agent(incident: dict) -> None:
    """Replay the root cause analysis agent step from stored incident data.

    Args:
        incident: The incident dictionary containing root cause findings.
    """
    evidence_lines = "\n".join([
        f"  {i + 1}. {e}"
        for i, e in enumerate(incident.get("supporting_evidence", []))
    ])

    content = (
        f"[bold]Hypothesis:[/bold]\n{incident.get('root_cause_hypothesis')}\n\n"
        f"[bold]Confidence Score:[/bold] {incident.get('root_cause_confidence'):.2f}\n"
        f"[bold]Price Anomaly Detected:[/bold] {incident.get('price_anomaly_detected')}\n"
        f"[bold]Volume Anomaly Detected:[/bold] {incident.get('volume_anomaly_detected')}\n\n"
        f"[bold]Supporting Evidence:[/bold]\n{evidence_lines}"
    )

    render_agent_panel(2, "Root Cause Analysis", content, "yellow")


def replay_impact_agent(incident: dict) -> None:
    """Replay the revenue impact scoring agent step from stored incident data.

    Args:
        incident: The incident dictionary containing impact findings.
    """
    content = (
        f"[bold]Revenue at Risk:[/bold] [red]${incident.get('revenue_at_risk_usd', 0):,.2f}[/red]\n"
        f"[bold]Confidence Interval:[/bold] {incident.get('confidence_interval')}\n"
        f"[bold]Degradation Coefficient:[/bold] {incident.get('degradation_coefficient'):.2f}\n"
        f"[bold]Severity:[/bold] {incident.get('severity')}\n\n"
        f"[bold]Executive Impact Summary:[/bold]\n{incident.get('impact_summary')}"
    )

    render_agent_panel(3, "Revenue Impact Scoring", content, "magenta")


def replay_action_layer(incident: dict) -> None:
    """Replay the action layer output from stored incident data.

    Args:
        incident: The incident dictionary containing action layer findings.
    """
    content = (
        f"[bold]Incident ID:[/bold] {incident.get('incident_id')}\n"
        f"[bold]Recommended Action:[/bold]\n{incident.get('recommended_action')}\n\n"
        f"[bold]Reasoning Engine:[/bold] {incident.get('reasoning_engine')}\n"
        f"[bold]Mitigation Script:[/bold] {incident.get('mitigation_script_path', 'Not generated')}"
    )

    render_agent_panel(4, "Action Layer — Incident Report", content, "green")


def run_replay(incident: dict) -> None:
    """Run the full agent reasoning chain replay for a given incident.

    Args:
        incident: The incident dictionary to replay.
    """
    console.print(
        Panel(
            f"[bold yellow]Replaying Investigation[/bold yellow]\n"
            f"Incident: [cyan]{incident.get('incident_id')}[/cyan]\n"
            f"Ticker: [cyan]{incident.get('ticker')}[/cyan]\n"
            f"Timestamp: {incident.get('timestamp', '')[:19].replace('T', ' ')} UTC",
            title="DriftGuard IQ — Replay Mode",
            border_style="yellow",
        )
    )

    console.print("\n[bold white]Replaying agent chain step by step...[/bold white]\n")

    replay_isolation_agent(incident)
    console.input("\n[dim]Press Enter to continue to Agent 2...[/dim]")

    replay_rootcause_agent(incident)
    console.input("\n[dim]Press Enter to continue to Agent 3...[/dim]")

    replay_impact_agent(incident)
    console.input("\n[dim]Press Enter to continue to Action Layer...[/dim]")

    replay_action_layer(incident)

    console.print(
        Panel(
            "[bold green]Replay complete.[/bold green]\n"
            "All agent steps have been replayed successfully.",
            border_style="green",
        )
    )


def main() -> None:
    """Main entry point for the DriftGuard IQ replay script."""
    parser = argparse.ArgumentParser(
        description="Replay a DriftGuard IQ incident investigation."
    )
    parser.add_argument(
        "--incident-id",
        type=str,
        default=None,
        help="Specific incident ID to replay. If omitted, shows interactive menu.",
    )

    args = parser.parse_args()

    try:
        incidents = load_incidents()
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        sys.exit(1)

    if args.incident_id:
        try:
            incident = find_incident_by_id(incidents, args.incident_id)
        except ValueError as error:
            console.print(f"[red]{error}[/red]")
            sys.exit(1)
    else:
        incident = select_incident_interactive(incidents)

    run_replay(incident)


if __name__ == "__main__":
    main()