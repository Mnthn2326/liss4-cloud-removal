"""End-to-end traffic and monitoring demonstration script (Phase 11).

Generates representative API traffic (health probes, valid inference requests,
and invalid edge cases) to populate Prometheus metrics and demonstrate Grafana dashboards.

Usage:
    python scripts/demo_monitoring.py
    python scripts/demo_monitoring.py --host localhost --port 8000 --requests 10
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

import numpy as np


def run_monitoring_demo(host: str = "localhost", port: int = 8000, num_requests: int = 10) -> None:
    base_url = f"http://{host}:{port}"
    print("=" * 70)
    print("LISS-IV Cloud Removal - End-to-End Monitoring & Metrics Demo")
    print("=" * 70)
    print(f"Target API Service:  {base_url}")
    print(f"Prometheus Endpoint: {base_url}/metrics")
    print("Grafana Dashboard:   http://localhost:3001 (admin / admin)")
    print("Prometheus Server:   http://localhost:9090")
    print("-" * 70)

    # 1. Health Probe
    print("\n[1/4] Probing Liveness (/health)...")
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  HTTP {resp.status}: {data}")
    except Exception as exc:
        print(f"  [ERROR] Health probe failed: {exc}")
        return

    # 2. Readiness Probe
    print("\n[2/4] Probing Readiness (/ready)...")
    try:
        with urllib.request.urlopen(f"{base_url}/ready", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(
                f"  HTTP {resp.status}: Model={data.get('model_type')} Device={data.get('device')} Checkpoint={data.get('checkpoint')}"
            )
    except Exception as exc:
        print(f"  [ERROR] Readiness probe failed: {exc}")
        return

    # 3. Generating Valid Inference Traffic
    print(f"\n[3/4] Generating {num_requests} Valid Inference Requests (/predict)...")
    rng = np.random.default_rng(42)
    latencies = []
    for i in range(num_requests):
        size = 32 if i % 2 == 0 else 64
        patch = rng.uniform(0.10, 0.45, size=(3, size, size)).tolist()
        payload = json.dumps({"image": patch}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as resp:
            latency = time.perf_counter() - t0
            latencies.append(latency)
            data = json.loads(resp.read().decode("utf-8"))
            print(
                f"  Req #{i+1:02d} ({size}x{size}): HTTP {resp.status} in {latency*1000:.1f}ms | Mean Refl={data['prediction_mean']:.4f}"
            )
        time.sleep(0.1)

    print(
        f"  Average Inference Latency: {np.mean(latencies)*1000:.2f}ms (p95: {np.percentile(latencies, 95)*1000:.2f}ms)"
    )

    # 4. Generating Client Error Traffic (422) for Error Rate Monitoring
    print("\n[4/4] Generating Client Rejection Edge Cases (HTTP 422 for error metrics)...")
    edge_cases = [
        ("Incorrect 4 Channels", {"image": [[[0.3] * 32] * 32] * 4}),
        ("Unscaled Reflectance (>1.0)", {"image": [[[1.5] * 32] * 32] * 3}),
        ("Small Spatial Size (8x8)", {"image": [[[0.3] * 8] * 8] * 3}),
    ]
    for label, payload_dict in edge_cases:
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=json.dumps(payload_dict).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"  {label}: Unexpected status {resp.status}")
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read().decode("utf-8"))
            print(f"  {label}: HTTP {exc.code} (Correctly rejected: {body.get('detail')[:55]}...)")

    # 5. Summary from /metrics
    print("\n" + "=" * 70)
    print("Prometheus Metrics Snapshot (/metrics):")
    print("-" * 70)
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as resp:
            metrics_text = resp.read().decode("utf-8")
            for line in metrics_text.splitlines():
                if (
                    line.startswith("api_requests_total")
                    or line.startswith("model_inference_requests_total")
                    or line.startswith("model_ready_status")
                ):
                    print(" ", line)
    except Exception as exc:
        print(f"  [ERROR] Scraping /metrics failed: {exc}")

    print("=" * 70)
    print("Demo complete! View real-time graphs in Grafana: http://localhost:3001")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end monitoring and inference demo")
    parser.add_argument("--host", type=str, default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument(
        "--requests", type=int, default=8, help="Number of valid inference requests"
    )
    args = parser.parse_args()

    run_monitoring_demo(host=args.host, port=args.port, num_requests=args.requests)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
