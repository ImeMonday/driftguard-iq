import os
from dotenv import load_dotenv

load_dotenv(override=True)

FOUNDRY_IQ_ENDPOINT = "https://driftguard-iq-resource.openai.azure.com/"
FOUNDRY_IQ_API_KEY = ""
FOUNDRY_IQ_DEPLOYMENT = "gpt-4o-mini"
FOUNDRY_IQ_API_VERSION = "2025-01-01-preview"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

PSI_WARNING_THRESHOLD = float(os.getenv("PSI_WARNING_THRESHOLD", 0.1))
PSI_CRITICAL_THRESHOLD = float(os.getenv("PSI_CRITICAL_THRESHOLD", 0.2))
PIPELINE_INTERVAL_SECONDS = int(os.getenv("PIPELINE_INTERVAL_SECONDS", 60))
BASELINE_WINDOW_DAYS = 365

SEVERITY_LEVELS = {
    "LOW": (0.0, PSI_WARNING_THRESHOLD),
    "MEDIUM": (PSI_WARNING_THRESHOLD, PSI_CRITICAL_THRESHOLD),
    "HIGH": (PSI_CRITICAL_THRESHOLD, 0.35),
    "CRITICAL": (0.35, float("inf")),
}

TRACKED_TICKERS = ["JPM", "BAC", "GS", "MS", "C"]

TRACKED_FEATURES = [
    "close_price",
    "volume",
    "price_change_pct",
    "volatility_7d",
    "relative_volume",
]

USE_FOUNDRY = bool(FOUNDRY_IQ_ENDPOINT and FOUNDRY_IQ_API_KEY)