import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    render_severity_badge,
    render_metric_card,
    render_pipeline_health,
    render_incident_table,
    render_psi_chart,
    render_evidence_expander,
    render_heatmap,
    render_confidence_chart,
    render_trend_analysis,
)

INCIDENTS_LOG_PATH = "logs/incidents.json"
PIPELINE_LOG_PATH = "logs/pipeline.log"


def load_incidents() -> list[dict]:
    """Load all incidents from the incidents log file.

    Returns:
        A list of incident dictionaries loaded from disk.
    """
    if not os.path.exists(INCIDENTS_LOG_PATH):
        return []

    if os.path.getsize(INCIDENTS_LOG_PATH) == 0:
        return []

    try:
        with open(INCIDENTS_LOG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []


def load_pipeline_log() -> list[dict]:
    """Load all pipeline cycle entries from the pipeline log file.

    Returns:
        A list of pipeline log entry dictionaries.
    """
    if not os.path.exists(PIPELINE_LOG_PATH):
        return []

    entries = []

    try:
        with open(PIPELINE_LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []

    return entries


def build_psi_history(pipeline_entries: list[dict]) -> pd.DataFrame:
    """Build a PSI history DataFrame from pipeline log entries.

    Args:
        pipeline_entries: List of pipeline log entry dictionaries.

    Returns:
        A DataFrame with timestamp, ticker, feature, and PSI score columns.
    """
    rows = []

    for entry in pipeline_entries:
        timestamp = entry.get("timestamp", "")
        ticker = entry.get("ticker", "")

        for result in entry.get("results", []):
            rows.append({
                "timestamp": timestamp,
                "ticker": ticker,
                "feature": result.get("feature", ""),
                "psi_score": result.get("psi_score", 0.0),
                "severity": result.get("severity", "LOW"),
                "deviation_pct": result.get("deviation_pct", 0.0),
            })

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "ticker", "feature", "psi_score", "severity", "deviation_pct"]
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.sort_values("timestamp")


def build_heatmap_data(pipeline_entries: list[dict]) -> pd.DataFrame:
    """Build a heatmap DataFrame of latest PSI scores per ticker and feature.

    Args:
        pipeline_entries: List of pipeline log entry dictionaries.

    Returns:
        A pivot DataFrame with tickers as columns and features as rows.
    """
    latest = {}

    for entry in pipeline_entries:
        ticker = entry.get("ticker", "")
        for result in entry.get("results", []):
            feature = result.get("feature", "")
            psi_score = result.get("psi_score", 0.0)
            latest[(ticker, feature)] = psi_score

    if not latest:
        return pd.DataFrame()

    rows = []
    for (ticker, feature), psi in latest.items():
        rows.append({"ticker": ticker, "feature": feature, "psi_score": psi})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="feature", columns="ticker", values="psi_score")

    return pivot.fillna(0)


def build_confidence_history(incidents: list[dict]) -> pd.DataFrame:
    """Build a confidence score history DataFrame from incidents.

    Args:
        incidents: List of incident dictionaries.

    Returns:
        A DataFrame with timestamp, ticker, and confidence score columns.
    """
    rows = []

    for incident in incidents:
        rows.append({
            "timestamp": incident.get("timestamp", "")[:19].replace("T", " "),
            "ticker": incident.get("ticker", ""),
            "confidence": incident.get("root_cause_confidence", 0.0),
            "incident_id": incident.get("incident_id", ""),
        })

    if not rows:
        return pd.DataFrame(columns=["timestamp", "ticker", "confidence", "incident_id"])

    return pd.DataFrame(rows)


def build_trend_data(pipeline_entries: list[dict]) -> pd.DataFrame:
    """Build drift trend data comparing current vs previous cycle PSI scores.

    Args:
        pipeline_entries: List of pipeline log entry dictionaries.

    Returns:
        A DataFrame with ticker, feature, current PSI, previous PSI, and trend.
    """
    cycles = {}

    for entry in pipeline_entries:
        ticker = entry.get("ticker", "")
        timestamp = entry.get("timestamp", "")

        if ticker not in cycles:
            cycles[ticker] = []

        cycles[ticker].append({
            "timestamp": timestamp,
            "results": entry.get("results", []),
        })

    rows = []

    for ticker, cycle_list in cycles.items():
        if len(cycle_list) < 2:
            continue

        current = cycle_list[-1]["results"]
        previous = cycle_list[-2]["results"]

        current_map = {r["feature"]: r["psi_score"] for r in current}
        previous_map = {r["feature"]: r["psi_score"] for r in previous}

        for feature, current_psi in current_map.items():
            prev_psi = previous_map.get(feature, current_psi)
            delta = current_psi - prev_psi

            if delta > 0.01:
                trend = "Worsening ↑"
                trend_color = "red"
            elif delta < -0.01:
                trend = "Recovering ↓"
                trend_color = "green"
            else:
                trend = "Stable →"
                trend_color = "yellow"

            rows.append({
                "ticker": ticker,
                "feature": feature,
                "current_psi": round(current_psi, 6),
                "previous_psi": round(prev_psi, 6),
                "delta": round(delta, 6),
                "trend": trend,
                "trend_color": trend_color,
            })

    return pd.DataFrame(rows)


def get_pipeline_health(pipeline_entries: list[dict]) -> dict:
    """Extract pipeline health statistics from log entries.

    Args:
        pipeline_entries: List of pipeline log entry dictionaries.

    Returns:
        A dictionary containing health metrics and last run timestamp.
    """
    if not pipeline_entries:
        return {
            "status": "No data",
            "last_run": "Never",
            "total_cycles": 0,
            "tickers_monitored": 0,
        }

    last_entry = pipeline_entries[-1]
    last_run = last_entry.get("timestamp", "Unknown")

    try:
        last_run_dt = datetime.fromisoformat(last_run)
        last_run_str = last_run_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        last_run_str = last_run

    tickers = set(e.get("ticker", "") for e in pipeline_entries)

    return {
        "status": "Active",
        "last_run": last_run_str,
        "total_cycles": len(pipeline_entries),
        "tickers_monitored": len(tickers),
    }


def run_replay_script(incident_id: str) -> str:
    """Run the replay script for a specific incident and capture output.

    Args:
        incident_id: The incident ID to replay.

    Returns:
        The captured stdout output from the replay script.
    """
    try:
        result = subprocess.run(
            [sys.executable, "scripts/replay.py", "--incident-id", incident_id],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout if result.stdout else result.stderr
    except subprocess.TimeoutExpired:
        return "Replay timed out after 60 seconds."
    except OSError as error:
        return f"Failed to run replay script: {error}"


def generate_pdf_report(incidents: list[dict], total_revenue: float) -> bytes:
    """Generate a PDF summary report of all active incidents.

    Args:
        incidents: List of incident dictionaries.
        total_revenue: Total revenue at risk across all incidents.

    Returns:
        PDF file content as bytes.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=12,
            textColor=colors.HexColor("#00d4ff"),
            alignment=TA_CENTER,
        )

        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=11,
            spaceAfter=6,
            textColor=colors.HexColor("#888888"),
            alignment=TA_CENTER,
        )

        header_style = ParagraphStyle(
            "Header",
            parent=styles["Heading2"],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.HexColor("#ff4444"),
        )

        story.append(Paragraph("DriftGuard IQ", title_style))
        story.append(Paragraph("Autonomous Financial Data Drift Detection Report", subtitle_style))
        story.append(Paragraph(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            subtitle_style,
        ))
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("Executive Summary", header_style))
        story.append(Paragraph(
            f"Total incidents detected: {len(incidents)}",
            styles["Normal"],
        ))
        story.append(Paragraph(
            f"Total revenue at risk: ${total_revenue:,.2f}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Incident Summary", header_style))

        table_data = [
            ["Incident ID", "Ticker", "Severity", "PSI Score", "Revenue at Risk"],
        ]

        for incident in incidents:
            table_data.append([
                incident.get("incident_id", "")[:20],
                incident.get("ticker", ""),
                incident.get("severity", ""),
                f"{incident.get('top_psi_score', 0):.4f}",
                f"${incident.get('revenue_at_risk_usd', 0):,.2f}",
            ])

        table = Table(table_data, colWidths=[2.2 * inch, 0.8 * inch, 1 * inch, 1 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00d4ff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(table)
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("Incident Details", header_style))

        for incident in incidents:
            story.append(Paragraph(
                f"{incident.get('incident_id')} — {incident.get('ticker')}",
                styles["Heading3"],
            ))
            story.append(Paragraph(
                f"Root Cause: {incident.get('root_cause_hypothesis', 'N/A')[:300]}",
                styles["Normal"],
            ))
            story.append(Paragraph(
                f"Recommended Action: {incident.get('recommended_action', 'N/A')}",
                styles["Normal"],
            ))
            story.append(Spacer(1, 0.15 * inch))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        return b""


def main() -> None:
    """Main entry point for the DriftGuard IQ Streamlit dashboard."""
    st.set_page_config(
        page_title="DriftGuard IQ",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .main-header {
            font-size: 2rem;
            font-weight: 700;
            color: #00d4ff;
            margin-bottom: 0.25rem;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #888;
            margin-bottom: 2rem;
        }
        .severity-critical { color: #ff4444; font-weight: bold; }
        .severity-high { color: #ff8c00; font-weight: bold; }
        .severity-medium { color: #ffd700; font-weight: bold; }
        .severity-low { color: #00cc44; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="main-header">🛡️ DriftGuard IQ</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Autonomous Financial Data Drift Detection & Mitigation — Powered by Azure Foundry IQ</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.title("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    selected_ticker = st.sidebar.selectbox(
        "Filter by ticker",
        ["All", "JPM", "BAC", "GS", "MS", "C"],
    )
    selected_feature = st.sidebar.selectbox(
        "Filter by feature",
        ["All", "close_price", "volume", "price_change_pct", "volatility_7d", "relative_volume"],
    )

    if auto_refresh:
        st.sidebar.info("Dashboard refreshes every 30 seconds.")

    incidents = load_incidents()
    pipeline_entries = load_pipeline_log()
    psi_history = build_psi_history(pipeline_entries)
    heatmap_data = build_heatmap_data(pipeline_entries)
    confidence_data = build_confidence_history(incidents)
    trend_data = build_trend_data(pipeline_entries)
    health = get_pipeline_health(pipeline_entries)

    total_revenue_at_risk = sum(
        i.get("revenue_at_risk_usd", 0) for i in incidents
    )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a0a0a, #2d0a0a);
            border: 1px solid #ff4444;
            border-radius: 12px;
            padding: 1.2rem 2rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div>
                <div style="font-size: 0.85rem; color: #ff8888;
                    font-weight: 500; letter-spacing: 0.05em;">
                    TOTAL REVENUE AT RISK ACROSS ALL ACTIVE INCIDENTS
                </div>
                <div style="font-size: 2.4rem; font-weight: 700;
                    color: #ff4444; margin-top: 4px;">
                    ${total_revenue_at_risk:,.2f}
                </div>
            </div>
            <div style="font-size: 3rem;">🚨</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            "Pipeline Status",
            health["status"],
            "🟢" if health["status"] == "Active" else "🔴",
        )

    with col2:
        render_metric_card("Total Incidents", str(len(incidents)), "🚨")

    with col3:
        render_metric_card("Pipeline Cycles", str(health["total_cycles"]), "🔄")

    with col4:
        render_metric_card("Tickers Monitored", str(health["tickers_monitored"]), "📊")

    st.divider()

    render_pipeline_health(health)

    st.divider()

    st.subheader("PSI Score History")

    if psi_history.empty:
        st.info("PSI history will populate after the next pipeline cycle completes.")
    else:
        filtered_psi = psi_history.copy()

        if selected_ticker != "All":
            filtered_psi = filtered_psi[filtered_psi["ticker"] == selected_ticker]

        if selected_feature != "All":
            filtered_psi = filtered_psi[filtered_psi["feature"] == selected_feature]

        render_psi_chart(filtered_psi)

    st.divider()

    st.subheader("Drift Severity Heatmap")

    if heatmap_data.empty:
        st.info("Heatmap will populate after the first pipeline cycle.")
    else:
        render_heatmap(heatmap_data)

    st.divider()

    st.subheader("Drift Trend Analysis")

    if trend_data.empty:
        st.info("Trend analysis requires at least two pipeline cycles.")
    else:
        filtered_trend = trend_data.copy()
        if selected_ticker != "All":
            filtered_trend = filtered_trend[filtered_trend["ticker"] == selected_ticker]
        render_trend_analysis(filtered_trend)

    st.divider()

    st.subheader("Agent Reasoning Confidence")

    if confidence_data.empty:
        st.info("Confidence tracking will populate after the first incident.")
    else:
        render_confidence_chart(confidence_data)

    st.divider()

    st.subheader("Incident History")

    if not incidents:
        st.info("No incidents recorded yet. The system is monitoring for drift.")
    else:
        filtered_incidents = incidents

        if selected_ticker != "All":
            filtered_incidents = [
                i for i in incidents if i.get("ticker") == selected_ticker
            ]

        render_incident_table(filtered_incidents)

        for incident in filtered_incidents:
            render_evidence_expander(incident)

        st.subheader("Incident Replay")
        incident_ids = [i.get("incident_id", "") for i in filtered_incidents]

        if incident_ids:
            selected_incident_id = st.selectbox(
                "Select an incident to replay",
                incident_ids,
            )

            if st.button("▶ Replay Investigation", type="primary"):
                with st.spinner("Running agent replay..."):
                    replay_output = run_replay_script(selected_incident_id)

                with st.expander("Replay Output", expanded=True):
                    st.code(replay_output, language="text")

    st.divider()

    st.subheader("Export Report")

    if incidents:
        if st.button("📄 Download PDF Incident Report", type="secondary"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_bytes = generate_pdf_report(incidents, total_revenue_at_risk)
                    if pdf_bytes:
                        st.download_button(
                            label="📥 Click to Download PDF",
                            data=pdf_bytes,
                            file_name=f"driftguard_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                        )
                    else:
                        st.warning("Install reportlab to enable PDF export: pip install reportlab")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()