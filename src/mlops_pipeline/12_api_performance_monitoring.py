
import json
from pathlib import Path

import pandas as pd

LOG_PATH = Path("logs/inference/inference_log.jsonl")
REPORT_PATH = Path("artifacts/api_performance_report.json")


def load_logs():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    with LOG_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("Inference log is empty")

    return pd.DataFrame(records)


def analyze_performance(df):
    total_requests = len(df)
    success_count = int((df["status"] == "success").sum())
    failure_count = int((df["status"] == "failure").sum())

    latency = df["latency_ms"].dropna()
    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    elapsed_seconds = (
        timestamps.max() - timestamps.min()
    ).total_seconds()

    report = {
        "total_requests": total_requests,
        "successful_requests": success_count,
        "failed_requests": failure_count,
        "success_rate_percent": round(
            success_count / total_requests * 100, 2
        ),
        "failure_rate_percent": round(
            failure_count / total_requests * 100, 2
        ),
        "latency_ms": {},
        "throughput_requests_per_second": (
            round(total_requests / elapsed_seconds, 6)
            if elapsed_seconds > 0 else None
        ),
        "status_distribution": {
            str(k): int(v)
            for k, v in df["status"].value_counts().items()
        },
    }

    if not latency.empty:
        report["latency_ms"] = {
            "average": round(float(latency.mean()), 3),
            "median": round(float(latency.median()), 3),
            "p95": round(float(latency.quantile(0.95)), 3),
            "minimum": round(float(latency.min()), 3),
            "maximum": round(float(latency.max()), 3),
        }

    print("\n" + "=" * 60)
    print("API PERFORMANCE MONITORING")
    print("=" * 60)
    print(f"\nTotal requests : {total_requests}")
    print(f"Successful     : {success_count}")
    print(f"Failed         : {failure_count}")
    print(f"Success rate   : {report['success_rate_percent']:.2f}%")
    print(f"Failure rate   : {report['failure_rate_percent']:.2f}%")

    if report["latency_ms"]:
        print("\nLatency:")
        for name, value in report["latency_ms"].items():
            print(f"{name.capitalize():15}: {value:.3f} ms")

    throughput = report["throughput_requests_per_second"]
    print(
        f"\nThroughput     : {throughput:.3f} requests/sec"
        if throughput is not None
        else "\nThroughput     : Cannot calculate from one timestamp"
    )

    print("\nStatus distribution:")
    print(df["status"].value_counts())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\nReport saved to: {REPORT_PATH}")


def main():
    df = load_logs()
    analyze_performance(df)


if __name__ == "__main__":
    main()
