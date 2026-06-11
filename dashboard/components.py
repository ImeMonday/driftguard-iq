import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SEVERITY_COLORS = {
    "CRITICAL": "#ff4444",
    "HIGH": "#ff8c00",
    "MEDIUM": "#ffd700",
    "LOW": "#00cc44",
}

SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢",
}


def render_severity_badge(severity: str) -> str:
    """Render an HTML severity badge for use in Streamlit markdown.

    Args:
        severity: The severity level string to render.

    Returns:
        An HTML string containing the styled severity badge.
    """
    color = SEVERITY_COLORS.get(severity, "#888888")
    emoji = SEVERITY_EMOJI.get(severity, "⚪")

    return (
        f'<span style="background-color:{color}; color:white; '
        f'padding:2px 8px; border-radius:4px; font-weight:bold; '
        f'font-size:0.85rem;">{emoji} {severity}</span>'
    )


def render_metric_card(label: str, value: str, icon: str) -> None:
    """Render a single metric card in the Streamlit dashboard.

    Args:
        label: The label to display below the metric value.
        value: The metric value to display prominently.
        icon: An emoji icon to display alongside the value.
    """
    st.markdown(
        f"""
        <div style="
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        ">
            <div style="font-size: 1.8rem;">{icon}</div>
            <div style="font-size: 1.5rem; font-weight: 700;
                color: #00d4ff;">{value}</div>
            <div style="font-size: 0.85rem; color: #888;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_health(health: dict) -> None:
    """Render the pipeline health status panel.

    Args:
        health: A dictionary containing pipeline health metrics.
    """
    status = health.get("status", "Unknown")
    last_run = health.get("last_run", "Never")
    status_color = "#00cc44" if status == "Active" else "#ff4444"

    st.markdown(
        f"""
        <div style="
            background: #0d1117;
            border-left: 4px solid {status_color};
            border-radius: 4px;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
        ">
            <span style="color:{status_color}; font-weight:700;">
                ● Pipeline {status}
            </span>
            <span style="color:#888; font-size:0.85rem; margin-left:1rem;">
                Last run: {last_run}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_psi_chart(psi_history: pd.DataFrame) -> None:
    """Render an interactive Plotly PSI score history chart.

    Args:
        psi_history: A DataFrame with timestamp, ticker, feature,
            and psi_score columns.
    """
    if psi_history.empty:
        st.warning("No PSI history data available for the selected filters.")
        return

    fig = px.line(
        psi_history,
        x="timestamp",
        y="psi_score",
        color="feature",
        facet_col="ticker",
        facet_col_wrap=3,
        title="PSI Score History by Ticker and Feature",
        labels={
            "psi_score": "PSI Score",
            "timestamp": "Time",
            "feature": "Feature",
        },
        template="plotly_dark",
        height=500,
    )

    fig.add_hline(
        y=0.1,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Warning (0.1)",
        annotation_position="bottom right",
    )

    fig.add_hline(
        y=0.2,
        line_dash="dash",
        line_color="orange",
        annotation_text="High (0.2)",
        annotation_position="bottom right",
    )

    fig.add_hline(
        y=0.35,
        line_dash="dash",
        line_color="red",
        annotation_text="Critical (0.35)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#ffffff",
        legend_title_text="Feature",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(heatmap_data: pd.DataFrame) -> None:
    """Render a PSI severity heatmap across all tickers and features.

    Args:
        heatmap_data: A pivot DataFrame with features as rows and tickers as columns.
    """
    if heatmap_data.empty:
        st.warning("No heatmap data available.")
        return

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        colorscale=[
            [0.0, "#00cc44"],
            [0.1, "#ffd700"],
            [0.2, "#ff8c00"],
            [0.35, "#ff4444"],
            [1.0, "#8b0000"],
        ],
        text=[[f"{v:.4f}" for v in row] for row in heatmap_data.values],
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        hoverongaps=False,
        showscale=True,
        colorbar=dict(
            title="PSI Score",
            tickvals=[0, 0.1, 0.2, 0.35, 1.0],
            ticktext=["LOW", "WARNING", "HIGH", "CRITICAL", "EXTREME"],
            tickfont=dict(color="white"),
            titlefont=dict(color="white"),
        ),
    ))

    fig.update_layout(
        title="Live PSI Severity Heatmap — All Tickers vs Features",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#ffffff",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(title="Ticker", tickfont=dict(color="white")),
        yaxis=dict(title="Feature", tickfont=dict(color="white")),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_trend_analysis(trend_data: pd.DataFrame) -> None:
    """Render a drift trend analysis table showing worsening or recovering features.

    Args:
        trend_data: A DataFrame with ticker, feature, current PSI,
            previous PSI, delta, and trend columns.
    """
    if trend_data.empty:
        st.info("No trend data available yet.")
        return

    trend_colors = {
        "Worsening ↑": "#ff4444",
        "Recovering ↓": "#00cc44",
        "Stable →": "#ffd700",
    }

    rows_html = ""
    for _, row in trend_data.iterrows():
        color = trend_colors.get(row["trend"], "#888")
        rows_html += f"""
        <tr>
            <td style="padding:6px 12px;">{row['ticker']}</td>
            <td style="padding:6px 12px;">{row['feature']}</td>
            <td style="padding:6px 12px;">{row['current_psi']:.6f}</td>
            <td style="padding:6px 12px;">{row['previous_psi']:.6f}</td>
            <td style="padding:6px 12px; color:{color};">{row['delta']:+.6f}</td>
            <td style="padding:6px 12px;">
                <span style="color:{color}; font-weight:bold;">{row['trend']}</span>
            </td>
        </tr>
        """

    st.markdown(
        f"""
        <table style="width:100%; border-collapse:collapse;
            background:#0d1117; color:white; font-size:0.9rem;">
            <thead>
                <tr style="background:#1a1a2e; color:#00d4ff;">
                    <th style="padding:8px 12px; text-align:left;">Ticker</th>
                    <th style="padding:8px 12px; text-align:left;">Feature</th>
                    <th style="padding:8px 12px; text-align:left;">Current PSI</th>
                    <th style="padding:8px 12px; text-align:left;">Previous PSI</th>
                    <th style="padding:8px 12px; text-align:left;">Delta</th>
                    <th style="padding:8px 12px; text-align:left;">Trend</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_chart(confidence_data: pd.DataFrame) -> None:
    """Render an agent reasoning confidence trend chart across incidents.

    Args:
        confidence_data: A DataFrame with timestamp, ticker, and confidence columns.
    """
    if confidence_data.empty:
        st.info("No confidence data available yet.")
        return

    fig = px.scatter(
        confidence_data,
        x="timestamp",
        y="confidence",
        color="ticker",
        hover_data=["incident_id"],
        title="Agent 2 Reasoning Confidence Score per Incident",
        labels={
            "confidence": "Confidence Score",
            "timestamp": "Time",
            "ticker": "Ticker",
        },
        template="plotly_dark",
        height=350,
    )

    fig.add_hline(
        y=0.6,
        line_dash="dash",
        line_color="red",
        annotation_text="Low confidence threshold (0.6)",
        annotation_position="bottom right",
    )

    fig.update_traces(marker=dict(size=12))

    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#ffffff",
        yaxis=dict(range=[0, 1.1]),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_incident_table(incidents: list[dict]) -> None:
    """Render a styled incident history table with severity badges.

    Args:
        incidents: List of incident dictionaries loaded from the log file.
    """
    if not incidents:
        st.info("No incidents to display.")
        return

    rows = []

    for incident in reversed(incidents):
        severity = incident.get("severity", "LOW")
        badge = render_severity_badge(severity)

        rows.append({
            "Incident ID": incident.get("incident_id", ""),
            "Timestamp": incident.get("timestamp", "")[:19].replace("T", " "),
            "Ticker": incident.get("ticker", ""),
            "Severity": badge,
            "Top Feature": incident.get("top_feature", ""),
            "PSI Score": f"{incident.get('top_psi_score', 0):.6f}",
            "Revenue at Risk": f"${incident.get('revenue_at_risk_usd', 0):,.2f}",
            "Engine": incident.get("reasoning_engine", ""),
        })

    df = pd.DataFrame(rows)

    st.markdown(
        df.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )


def render_evidence_expander(incident: dict) -> None:
    """Render a collapsible expander showing incident supporting evidence.

    Args:
        incident: A single incident dictionary containing evidence fields.
    """
    with st.expander(
        f"Evidence — {incident.get('incident_id', '')} ({incident.get('ticker', '')})"
    ):
        st.markdown("**Root Cause Hypothesis**")
        st.info(incident.get("root_cause_hypothesis", "No hypothesis available."))

        st.markdown("**Supporting Evidence**")
        for i, evidence in enumerate(incident.get("supporting_evidence", []), 1):
            st.markdown(f"{i}. {evidence}")

        st.markdown("**Executive Impact Summary**")
        st.warning(incident.get("impact_summary", "No impact summary available."))

        st.markdown("**Recommended Action**")
        severity = incident.get("severity", "LOW")
        color = SEVERITY_COLORS.get(severity, "#888")

        st.markdown(
            f'<div style="border-left:4px solid {color}; '
            f'padding:0.5rem 1rem; background:#1a1a2e;">'
            f'{incident.get("recommended_action", "No action recommended.")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if incident.get("mitigation_script_path"):
            st.markdown("**Mitigation Script**")
            st.code(incident.get("mitigation_script_path"), language="text")