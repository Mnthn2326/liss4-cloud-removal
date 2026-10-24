"""Benchmark script measuring sequential vs concurrent inference performance (Phase 4).

Measures requests per second, median latency, and 95th percentile latency
under controlled sequential and concurrent workloads.
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi.testclient import TestClient

from cloud_removal.api import ModelManager, create_app
from cloud_removal.models.unet import UNet


def run_benchmark(num_requests: int = 16, patch_size: int = 64) -> dict:
    # Set up model and app with concurrency limiter = 4
    model = UNet(in_channels=3, out_channels=3, base_channels=8, bilinear=False, use_sigmoid=True)
    model.eval()

    manager = ModelManager(device_str="cpu", max_concurrent=4, queue_timeout=10.0)
    manager.set_model(model, checkpoint_name="benchmark_model.pt")
    app = create_app(model_manager=manager)
    client = TestClient(app)

    # Prepare synthetic payloads
    rng = np.random.default_rng(42)
    payloads = [
        {"image": rng.uniform(0.1, 0.9, size=(3, patch_size, patch_size)).astype(float).tolist()}
        for _ in range(num_requests)
    ]

    # Warm-up
    client.post("/predict", json=payloads[0])

    # 1. Sequential Benchmark
    seq_latencies = []
    t_seq_start = time.perf_counter()
    for p in payloads:
        t0 = time.perf_counter()
        resp = client.post("/predict", json=p)
        assert resp.status_code == 200
        seq_latencies.append(time.perf_counter() - t0)
    t_seq_total = time.perf_counter() - t_seq_start
    seq_rps = num_requests / t_seq_total

    # 2. Concurrent Benchmark (4 workers)
    def send_req(payload):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload)
        lat = time.perf_counter() - t0
        return resp.status_code, lat

    con_latencies = []
    t_con_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(send_req, payloads))
    t_con_total = time.perf_counter() - t_con_start

    for status_code, lat in results:
        assert status_code == 200
        con_latencies.append(lat)
    con_rps = num_requests / t_con_total

    results_summary = {
        "num_requests": num_requests,
        "patch_size": f"{patch_size}x{patch_size}",
        "sequential": {
            "total_time_s": round(t_seq_total, 4),
            "requests_per_second": round(seq_rps, 2),
            "median_latency_ms": round(statistics.median(seq_latencies) * 1000, 2),
            "p95_latency_ms": round(np.percentile(seq_latencies, 95) * 1000, 2),
        },
        "concurrent_4_workers": {
            "total_time_s": round(t_con_total, 4),
            "requests_per_second": round(con_rps, 2),
            "median_latency_ms": round(statistics.median(con_latencies) * 1000, 2),
            "p95_latency_ms": round(np.percentile(con_latencies, 95) * 1000, 2),
        },
        "throughput_speedup": round(con_rps / seq_rps, 2),
    }
    return results_summary


if __name__ == "__main__":
    import json

    res = run_benchmark(num_requests=20, patch_size=64)
    print(json.dumps(res, indent=2))
