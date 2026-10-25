"""Tests for Prometheus metrics instrumentation in FastAPI service (Phase 11).

Verifies metrics collection, request counts, error tracking, latency histograms,
model readiness gauges, and /metrics scraping output.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from cloud_removal.api import MetricsManager, ModelManager, create_app
from cloud_removal.models.unet import UNet


@pytest.fixture
def isolated_app() -> TestClient:
    """Create a test client with isolated model and metrics managers."""
    registry = CollectorRegistry(auto_describe=True)
    metrics_mgr = MetricsManager(registry=registry)
    model_mgr = ModelManager(checkpoint_path=None, device_str="cpu")

    tiny_model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=4,
        bilinear=False,
        use_sigmoid=True,
        prior_reflectance=0.107,
    )
    tiny_model.eval()
    model_mgr.set_model(tiny_model, checkpoint_name="in_memory_metrics_test.pt")

    app = create_app(model_manager=model_mgr, metrics_manager=metrics_mgr)
    return TestClient(app)


def test_metrics_endpoint_accessible(isolated_app: TestClient):
    """GET /metrics must return HTTP 200 with Prometheus text content type."""
    resp = isolated_app.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    assert "api_requests_total" in resp.text
    assert "model_ready_status" in resp.text


def test_api_requests_counter_increments(isolated_app: TestClient):
    """Calling /health must increment api_requests_total with method and status 200."""
    isolated_app.get("/health")
    isolated_app.get("/health")

    metrics_resp = isolated_app.get("/metrics")
    body = metrics_resp.text
    assert 'api_requests_total{endpoint="/health",method="GET",status_code="200"} 2.0' in body


def test_model_ready_gauge_behavior(isolated_app: TestClient):
    """model_ready_status must report 1 when ready, and 0 when unloaded."""
    # When initialized
    resp1 = isolated_app.get("/metrics")
    assert "model_ready_status 1.0" in resp1.text

    # When unloaded
    model_mgr: ModelManager = isolated_app.app.state.model_manager
    model_mgr.unload()
    resp2 = isolated_app.get("/metrics")
    assert "model_ready_status 0.0" in resp2.text


def test_inference_metrics_recorded(isolated_app: TestClient):
    """Successful POST /predict must increment model_inference_requests_total success."""
    sample = np.full((3, 32, 32), 0.25, dtype=float).tolist()
    pred_resp = isolated_app.post("/predict", json={"image": sample})
    assert pred_resp.status_code == 200

    metrics_resp = isolated_app.get("/metrics")
    body = metrics_resp.text
    assert 'model_inference_requests_total{model_type="UNet",status="success"} 1.0' in body
    assert 'model_inference_duration_seconds_count{model_type="UNet"} 1.0' in body


def test_invalid_request_metrics_recorded(isolated_app: TestClient):
    """Invalid requests returning 422 must be recorded in api_requests_total."""
    # Send malformed 1-channel array
    bad_sample = [[[0.5] * 32] * 32]
    pred_resp = isolated_app.post("/predict", json={"image": bad_sample})
    assert pred_resp.status_code == 422

    metrics_resp = isolated_app.get("/metrics")
    body = metrics_resp.text
    assert 'api_requests_total{endpoint="/predict",method="POST",status_code="422"} 1.0' in body


def test_inference_error_metrics_recorded(isolated_app: TestClient):
    """Inference execution failure must increment model_inference_requests_total error."""
    model_mgr: ModelManager = isolated_app.app.state.model_manager
    sample = np.full((3, 32, 32), 0.25, dtype=float).tolist()

    with patch.object(
        model_mgr, "predict", side_effect=RuntimeError("Simulated execution failure")
    ):
        resp = isolated_app.post("/predict", json={"image": sample})
        assert resp.status_code == 500

    metrics_resp = isolated_app.get("/metrics")
    body = metrics_resp.text
    assert 'model_inference_requests_total{model_type="UNet",status="error"} 1.0' in body
    assert 'api_requests_total{endpoint="/predict",method="POST",status_code="500"} 1.0' in body


def test_input_monitoring_and_rejected_reasons(isolated_app: TestClient):
    """Input dimensions and reflectance histograms must be recorded, plus rejection reasons."""
    # Successful request
    sample = np.full((3, 32, 32), 0.40, dtype=float).tolist()
    pred_resp = isolated_app.post("/predict", json={"image": sample})
    assert pred_resp.status_code == 200

    # Rejected request (invalid channels)
    bad_sample = [[[0.5] * 32] * 32]
    pred_bad = isolated_app.post("/predict", json={"image": bad_sample})
    assert pred_bad.status_code == 422

    metrics_resp = isolated_app.get("/metrics")
    body = metrics_resp.text
    assert "input_image_height_pixels_count 1.0" in body
    assert "input_image_width_pixels_count 1.0" in body
    assert "input_pixel_mean_reflectance_count 1.0" in body
    assert 'rejected_requests_total{reason="invalid_channel_count"} 1.0' in body


def test_model_version_info_metric(isolated_app: TestClient):
    """model_version_info gauge must reflect current model and dataset versions."""
    metrics_resp = isolated_app.get("/metrics")
    body = metrics_resp.text
    assert "model_version_info{" in body
    assert 'dataset_version="RICE2-v1"' in body
    assert 'model_version="1.0.0"' in body
