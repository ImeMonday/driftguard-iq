import logging
import os
from datetime import datetime

from action.schema import IncidentReport, SeverityLevel
from reasoning.agent_isolation import IsolationFinding

logger = logging.getLogger(__name__)

MITIGATIONS_DIR = "logs/mitigations"


def resolve_mitigation_strategy(
    severity: SeverityLevel,
    top_feature: str,
) -> str:
    """Resolve the appropriate mitigation strategy based on severity and feature.

    Args:
        severity: The overall incident severity level.
        top_feature: The feature with the highest PSI score.

    Returns:
        A string identifier for the mitigation strategy to apply.
    """
    if severity == SeverityLevel.CRITICAL:
        return "halt_and_escalate"
    if severity == SeverityLevel.HIGH:
        return "retrain_and_recalibrate"
    if severity == SeverityLevel.MEDIUM:
        return "revalidate_inputs"
    return "monitor_and_log"


def generate_halt_and_escalate(
    incident: IncidentReport,
    isolation_finding: IsolationFinding,
) -> str:
    """Generate a mitigation script for CRITICAL severity incidents.

    Args:
        incident: The validated IncidentReport for this incident.
        isolation_finding: The findings from the isolation agent.

    Returns:
        A string containing the complete Python mitigation script.
    """
    features_list = str(isolation_finding.drifted_features)
    psi_scores = str(isolation_finding.psi_scores)

    return f'''import logging
import json
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def halt_automated_decisions(ticker: str, drifted_features: list) -> None:
    """Halt all automated decisions for the affected ticker.

    Args:
        ticker: The stock ticker symbol affected by critical drift.
        drifted_features: List of features that have critically drifted.
    """
    logger.critical(
        f"CRITICAL DRIFT DETECTED — Halting automated decisions for {{ticker}}"
    )
    halt_record = {{
        "ticker": ticker,
        "halted_at": datetime.utcnow().isoformat(),
        "drifted_features": drifted_features,
        "incident_id": "{incident.incident_id}",
        "reason": "PSI score exceeded CRITICAL threshold",
        "requires_human_approval": True,
    }}
    with open("logs/halt_record.json", "w") as f:
        json.dump(halt_record, f, indent=2)

    logger.critical("Halt record written. Human approval required to resume.")


def escalate_to_risk_team(ticker: str, revenue_at_risk: float) -> None:
    """Log an escalation notice for the risk and engineering teams.

    Args:
        ticker: The stock ticker symbol affected.
        revenue_at_risk: Estimated revenue at risk in USD.
    """
    escalation = {{
        "escalated_at": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "incident_id": "{incident.incident_id}",
        "revenue_at_risk_usd": revenue_at_risk,
        "severity": "CRITICAL",
        "action_required": "Immediate review by risk and engineering teams",
    }}
    with open("logs/escalation_notice.json", "w") as f:
        json.dump(escalation, f, indent=2)

    logger.critical(f"Escalation notice written. Revenue at risk: ${{revenue_at_risk:,.2f}}")


if __name__ == "__main__":
    ticker = "{incident.ticker}"
    drifted_features = {features_list}
    revenue_at_risk = {incident.revenue_at_risk_usd}

    halt_automated_decisions(ticker, drifted_features)
    escalate_to_risk_team(ticker, revenue_at_risk)

    logger.critical("CRITICAL mitigation complete. Await human approval before resuming.")
    sys.exit(1)
'''


def generate_retrain_and_recalibrate(
    incident: IncidentReport,
    isolation_finding: IsolationFinding,
) -> str:
    """Generate a mitigation script for HIGH severity incidents.

    Args:
        incident: The validated IncidentReport for this incident.
        isolation_finding: The findings from the isolation agent.

    Returns:
        A string containing the complete Python mitigation script.
    """
    features_list = str(isolation_finding.drifted_features)
    psi_scores = str(isolation_finding.psi_scores)

    return f'''import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flag_features_for_retraining(
    ticker: str,
    drifted_features: list,
    psi_scores: dict,
) -> None:
    """Flag drifted features for model retraining.

    Args:
        ticker: The stock ticker symbol affected.
        drifted_features: List of features that have drifted.
        psi_scores: Dictionary mapping feature names to PSI scores.
    """
    retraining_request = {{
        "ticker": ticker,
        "incident_id": "{incident.incident_id}",
        "requested_at": datetime.utcnow().isoformat(),
        "features_to_retrain": drifted_features,
        "psi_scores": psi_scores,
        "priority": "HIGH",
        "deadline_hours": 24,
    }}
    with open("logs/retraining_request.json", "w") as f:
        json.dump(retraining_request, f, indent=2)

    logger.warning(f"Retraining request logged for {{len(drifted_features)}} features on {{ticker}}")


def recalibrate_thresholds(ticker: str, top_feature: str, deviation_pct: float) -> None:
    """Log a recalibration notice for model decision thresholds.

    Args:
        ticker: The stock ticker symbol affected.
        top_feature: The most severely drifted feature.
        deviation_pct: Percentage deviation from baseline.
    """
    recalibration = {{
        "ticker": ticker,
        "incident_id": "{incident.incident_id}",
        "top_feature": top_feature,
        "deviation_pct": deviation_pct,
        "recalibration_recommended": True,
        "logged_at": datetime.utcnow().isoformat(),
    }}
    with open("logs/recalibration_notice.json", "w") as f:
        json.dump(recalibration, f, indent=2)

    logger.warning(f"Recalibration notice written for {{top_feature}} on {{ticker}}")


if __name__ == "__main__":
    ticker = "{incident.ticker}"
    drifted_features = {features_list}
    psi_scores = {psi_scores}
    top_feature = "{isolation_finding.top_feature}"
    deviation_pct = {isolation_finding.deviation_map.get(isolation_finding.top_feature, 0)}

    flag_features_for_retraining(ticker, drifted_features, psi_scores)
    recalibrate_thresholds(ticker, top_feature, deviation_pct)

    logger.warning("HIGH severity mitigation complete. Review retraining request within 24 hours.")
'''


def generate_revalidate_inputs(
    incident: IncidentReport,
    isolation_finding: IsolationFinding,
) -> str:
    """Generate a mitigation script for MEDIUM severity incidents.

    Args:
        incident: The validated IncidentReport for this incident.
        isolation_finding: The findings from the isolation agent.

    Returns:
        A string containing the complete Python mitigation script.
    """
    features_list = str(isolation_finding.drifted_features)

    return f'''import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def revalidate_feature_inputs(ticker: str, drifted_features: list) -> None:
    """Log a revalidation request for drifted feature inputs.

    Args:
        ticker: The stock ticker symbol affected.
        drifted_features: List of features requiring revalidation.
    """
    revalidation = {{
        "ticker": ticker,
        "incident_id": "{incident.incident_id}",
        "requested_at": datetime.utcnow().isoformat(),
        "features_to_revalidate": drifted_features,
        "priority": "MEDIUM",
        "deadline_hours": 48,
    }}
    with open("logs/revalidation_request.json", "w") as f:
        json.dump(revalidation, f, indent=2)

    logger.info(f"Revalidation request logged for {{len(drifted_features)}} features on {{ticker}}")


if __name__ == "__main__":
    ticker = "{incident.ticker}"
    drifted_features = {features_list}

    revalidate_feature_inputs(ticker, drifted_features)
    logger.info("MEDIUM severity mitigation complete. Review within 48 hours.")
'''


def generate_monitor_and_log(
    incident: IncidentReport,
    isolation_finding: IsolationFinding,
) -> str:
    """Generate a mitigation script for LOW severity incidents.

    Args:
        incident: The validated IncidentReport for this incident.
        isolation_finding: The findings from the isolation agent.

    Returns:
        A string containing the complete Python mitigation script.
    """
    features_list = str(isolation_finding.drifted_features)

    return f'''import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_low_severity_drift(ticker: str, drifted_features: list) -> None:
    """Log a low severity drift observation for monitoring purposes.

    Args:
        ticker: The stock ticker symbol affected.
        drifted_features: List of features showing low severity drift.
    """
    observation = {{
        "ticker": ticker,
        "incident_id": "{incident.incident_id}",
        "observed_at": datetime.utcnow().isoformat(),
        "features_observed": drifted_features,
        "severity": "LOW",
        "action": "Monitor over next 24 hours",
    }}
    with open("logs/drift_observations.json", "a") as f:
        f.write(json.dumps(observation) + "\\n")

    logger.info(f"Low severity drift observation logged for {{ticker}}")


if __name__ == "__main__":
    ticker = "{incident.ticker}"
    drifted_features = {features_list}

    log_low_severity_drift(ticker, drifted_features)
    logger.info("LOW severity mitigation complete. Continue monitoring.")
'''


def generate_mitigation_script(
    incident: IncidentReport,
    isolation_finding: IsolationFinding,
) -> str:
    """Generate and write a mitigation script file for the given incident.

    Args:
        incident: The validated IncidentReport for this incident.
        isolation_finding: The findings from the isolation agent.

    Returns:
        The file path of the generated mitigation script.

    Raises:
        OSError: If the script file cannot be written.
    """
    os.makedirs(MITIGATIONS_DIR, exist_ok=True)

    strategy = resolve_mitigation_strategy(
        severity=incident.severity,
        top_feature=incident.top_feature,
    )

    script_generators = {
        "halt_and_escalate": generate_halt_and_escalate,
        "retrain_and_recalibrate": generate_retrain_and_recalibrate,
        "revalidate_inputs": generate_revalidate_inputs,
        "monitor_and_log": generate_monitor_and_log,
    }

    generator = script_generators.get(strategy, generate_monitor_and_log)
    script_content = generator(incident, isolation_finding)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"mitigation_{incident.ticker}_{incident.incident_id}_{timestamp}.py"
    script_path = os.path.join(MITIGATIONS_DIR, filename)

    try:
        with open(script_path, "w") as script_file:
            script_file.write(script_content)

        logger.info(f"Mitigation script generated at {script_path}")

    except OSError as error:
        logger.error(f"Failed to write mitigation script: {error}")
        raise

    return script_path