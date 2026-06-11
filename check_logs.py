import json
import os
import sys

LOG_FILE = os.path.join("logs", "pipeline.log")


def inspect_pipeline_logs(lines_to_read: int = 5):
    """Parses and prints the latest entries from the DriftGuardIQ pipeline log."""
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found at: {LOG_FILE}")
        print("Ensure your main pipeline has run at least once.")
        return

    print(f"\nInspecting Last {lines_to_read} Entries inside {LOG_FILE}:")
    print("=" * 70)

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f if line.strip()]
            tail_lines = all_lines[-lines_to_read:]

        for line in tail_lines:
            try:
                payload = json.loads(line)
                timestamp = payload.get("timestamp", "UNKNOWN-TIME")
                ticker = payload.get("ticker", "UNKN")
                results = payload.get("results", [])

                short_ts = timestamp.split(".")[0].replace("T", " ")

                if not results:
                    print(
                        f"[{short_ts}] Ticker: {ticker:<4} | "
                        "Status: EMPTY - Data/Baseline Starvation"
                    )
                else:
                    print(
                        f"[{short_ts}] Ticker: {ticker:<4} | "
                        f"Status: ACTIVE METRICS ({len(results)} features)"
                    )
                    for item in results:
                        feature = item.get("feature", "N/A")
                        psi = item.get("psi_score", 0.0)
                        severity = item.get("severity", "LOW")
                        deviation = item.get("deviation_pct", 0.0)
                        print(
                            f"    {feature:<22} "
                            f"PSI: {psi:.6f}  "
                            f"Severity: {severity:<10}  "
                            f"Drift: {deviation:+.2f}%"
                        )

            except json.JSONDecodeError:
                print(f"Raw Line (Non-JSON): {line}")

    except Exception as e:
        print(f"Failed to parse log file: {e}")

    print("=" * 70)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    inspect_pipeline_logs(count)