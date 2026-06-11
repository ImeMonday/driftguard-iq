# DriftGuard IQ 🛡️

### Autonomous Financial Data Drift Detection & Mitigation Agent

> An autonomous financial auditor that catches silent data drift in real-time banking pipelines, investigates the root cause using a three-agent reasoning chain powered by Microsoft Azure Foundry IQ, and delivers structured incident reports with automated mitigation scripts.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Azure Foundry IQ](https://img.shields.io/badge/Microsoft-Foundry%20IQ-purple)
![Agents League](https://img.shields.io/badge/Agents%20League-Reasoning%20Agents-green)

## The Problem

In digital banking, pipelines rarely fail loudly. A minor API update or schema change silently shifts the incoming data distribution. The system keeps running — but the underlying ML models start making corrupted, costly predictions because the data they depend on has drifted away from what they were trained on.

By the time anyone notices, the revenue damage is already done.

DriftGuard IQ sits in front of that problem and catches it before it becomes a crisis.

---

## What It Does

DriftGuard IQ is a production-grade autonomous multi-agent system that:

- Monitors live financial market data for JPMorgan, Bank of America, Goldman Sachs, Morgan Stanley, and Citigroup in real time using Yahoo Finance
- Calculates Population Stability Index (PSI) across five key financial features every 60 seconds
- Triggers a three-agent investigation chain powered by Microsoft Azure Foundry IQ when drift is detected
- Delivers a structured incident post-mortem with revenue at risk quantified in USD
- Auto-generates a Python mitigation script tailored to the severity and drift type
- Displays everything on a live Streamlit dashboard with PSI history charts, severity heatmaps, and a PDF export

---

## Architecture
┌─────────────────────────────────────────────────────────────┐
│                      DRIFTGUARD IQ                          │
│                                                             │
│  ┌─────────────────┐     ┌───────────────────────────────┐  │
│  │  DETECTION      │────▶│   REASONING LAYER             │  │
│  │  LAYER          │     │   Microsoft Azure Foundry IQ  │  │
│  │                 │     │                               │  │
│  │  Yahoo Finance  │     │  Agent 1: Feature Isolation   │  │
│  │  Live Data      │     │                               │  │
│  │                 │     │  Agent 2: Root Cause Analysis │  │
│  │  PSI Calculator │     │                               │  │
│  │  5 Features     │     │  Agent 3: Revenue Impact      │  │
│  │  5 Tickers      │     │  Scoring        
│  │
│  └─────────────────┘     └───────────────┬───────────────┘  │
│                                          │                  │
│                          ┌───────────────▼───────────────┐  │
│                          │      ACTION LAYER             │  │
│                          │                               │  │
│                          │  Pydantic Validated Report    │  │
│                          │  Mitigation Script Generator  │  │
│                          │  Streamlit Dashboard          │  │
│                          │  PDF Export                   │  │
│                          └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
---

## How It Works

### Detection Layer
An AsyncIO pipeline fetches live OHLCV data for all five tickers every 60 seconds. It computes five financial features — close price, volume, price change percentage, 7-day rolling volatility, and relative volume — then calculates the Population Stability Index for each feature against a 365-day historical baseline. PSI scores are classified into four severity levels: LOW, MEDIUM, HIGH, and CRITICAL.

### Reasoning Layer — Microsoft Azure Foundry IQ
When any feature exceeds the warning threshold, Foundry IQ orchestrates a sequential three-agent investigation:

**Agent 1 — Feature Isolation:** Identifies which features have drifted, ranks them by PSI score, and produces a structured isolation finding.

**Agent 2 — Root Cause Analysis:** Correlates the drift onset against recent price action and volume anomalies to construct a root cause hypothesis with a confidence score.

**Agent 3 — Revenue Impact Scoring:** Maps the PSI score and severity to a model degradation coefficient, multiplies it against the configured daily decision value for each ticker, and returns a revenue at risk estimate with a confidence interval.

### Action Layer
The system validates the full investigation against a strict Pydantic v2 schema, writes an atomic JSON incident report to disk, renders a formatted rich terminal summary, and auto-generates a Python mitigation script — halt and escalate for CRITICAL, retrain and recalibrate for HIGH, revalidate inputs for MEDIUM, and monitor and log for LOW.

---

## Dashboard

The Streamlit dashboard provides:

- Total revenue at risk banner across all active incidents
- Live PSI score history charts per ticker and feature
- Drift severity heatmap showing all tickers vs features
- Drift trend analysis showing worsening, stable, or recovering features
- Agent reasoning confidence tracker per incident
- Full incident history table with severity badges
- Evidence expanders with root cause hypothesis and supporting evidence
- Incident replay — re-run the full agent chain for any past incident
- PDF export of the full incident report

---

## Sample Incident Output

```json
{
  "incident_id": "DG-20260611-8ADB8FAB",
  "timestamp": "2026-06-11T09:08:41.142331",
  "ticker": "GS",
  "top_feature": "close_price",
  "top_psi_score": 4.865636,
  "severity": "CRITICAL",
  "root_cause_hypothesis": "The drift in close_price is likely driven by significant shifts in market dynamics, possibly due to macroeconomic factors or changes in market sentiment impacting trading volumes and price stability.",
  "root_cause_confidence": 0.85,
  "revenue_at_risk_usd": 6127580.13,
  "confidence_interval": "$4,595,685 — $8,272,233",
  "recommended_action": "CRITICAL: Halt automated decisions dependent on close_price for GS. Escalate to engineering and risk teams immediately.",
  "reasoning_engine": "Azure Foundry IQ (gpt-4o-mini)"
}
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Live data | Yahoo Finance via yfinance |
| Drift detection | Population Stability Index (PSI) |
| Agent orchestration | Microsoft Azure Foundry IQ |
| AI model | gpt-4o-mini |
| Schema validation | Pydantic v2 |
| Pipeline | AsyncIO |
| Terminal output | Rich |
| Dashboard | Streamlit + Plotly |
| PDF export | ReportLab |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Microsoft Azure subscription with Azure AI Foundry project
- gpt-4o-mini deployment on Azure OpenAI

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/driftguard-iq.git
cd driftguard-iq
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```plaintext
FOUNDRY_IQ_ENDPOINT=https://your-resource.openai.azure.com/
FOUNDRY_IQ_API_KEY=your_api_key_here
FOUNDRY_IQ_DEPLOYMENT=gpt-4o-mini
FOUNDRY_IQ_API_VERSION=2025-01-01-preview
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
PSI_WARNING_THRESHOLD=0.1
PSI_CRITICAL_THRESHOLD=0.2
PIPELINE_INTERVAL_SECONDS=60
BASELINE_WINDOW_DAYS=365
```

### Run

```bash
python main.py
```

Open your browser at `http://localhost:8501` to view the dashboard.

---

## Project Structure
driftguard-iq/
├── config.py
├── main.py
├── requirements.txt
├── detection/
│   ├── pipeline.py
│   ├── psi_calculator.py
│   ├── data_fetcher.py
│   └── baseline.py
├── reasoning/
│   ├── orchestrator.py
│   ├── agent_isolation.py
│   ├── agent_rootcause.py
│   └── agent_impact.py
├── action/
│   ├── schema.py
│   ├── incident_reporter.py
│   └── mitigation_generator.py
├── dashboard/
│   ├── app.py
│   └── components.py
├── scripts/
│   └── replay.py
├── data/
│   └── baseline_cache.json
└── logs/
├── incidents.json
├── pipeline.log
└── mitigations/
---

## Hackathon

Built for the **[Agents League Hackathon 2026](https://aka.ms/AgentsLeagueFAQ)** — Reasoning Agents track.

Microsoft IQ Integration: **Foundry IQ** via Azure OpenAI gpt-4o-mini

---

## Key Results

- Detects critical data drift across 5 major banking tickers in real time
- Quantifies revenue at risk up to $6.2M per ticker per incident
- Generates root cause hypotheses with 85% confidence via Azure Foundry IQ
- Produces actionable mitigation scripts automatically classified by severity
- Full incident lifecycle from detection to report in under 60 seconds
