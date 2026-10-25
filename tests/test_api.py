"""Unit and integration tests for FastAPI inference service (Phase 10).

Tests health, readiness, input validation, error handling, thread safety,
and deterministic CPU inference using lightweight in-memory models.
Does not require real checkpoint files or satellite datasets.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import torch
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from cloud_removal.api import (
    ModelManager,
    create_app,
)
from cloud_removal.models.unet import UNet


@pytest.fixture
def tiny_model() -> UNet:
    """Create a lightweight in-memory UNet model for testing."""
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=4,
        bilinear=False,
        use_sigmoid=True,
        prior_reflectance=0.107,
    )
    model.eval()
    return model


@pytest.fixture
def unready_app() -> TestClient:
    """FastAPI TestClient with an uninitialized model manager."""
    manager = ModelManager(checkpoint_path=None, device_str="cpu")
    app = create_app(model_manager=manager)
    return TestClient(app)


@pytest.fixture
def ready_app(tiny_model: UNet) -> TestClient:
    """FastAPI TestClient with a pre-loaded in-memory model."""
    manager = ModelManager(checkpoint_path=None, device_str="cpu")
    manager.set_model(tiny_model, checkpoint_name="test_in_memory_model.pt")
    app = create_app(model_manager=manager)
    return TestClient(app)


# ============================================================================
# 1. Health Endpoint Tests
# ============================================================================


def test_health_before_model_initialization(unready_app: TestClient):
    """GET /health must return 200 with model_loaded=False before model is loaded."""
    resp = unready_app.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "liss4-cloud-removal-api"
    assert data["version"] == "0.1.0"
    assert data["model_loaded"] is False
    assert data["ready"] is False


def test_health_after_model_initialization(ready_app: TestClient):
    """GET /health must report model_loaded=True and ready=True once initialized."""
    resp = ready_app.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is True
    assert data["ready"] is True


# ============================================================================
# 2 & 3. Readiness Endpoint Tests
# ============================================================================


def test_readiness_unavailable_when_uninitialized(unready_app: TestClient):
    """GET /ready must return 503 when the model is not loaded."""
    resp = unready_app.get("/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["ready"] is False
    assert data["status"] == "unavailable"
    assert "detail" in data


def test_readiness_success_when_initialized(ready_app: TestClient):
    """GET /ready must return 200 with model details when ready."""
    resp = ready_app.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["status"] == "ready"
    assert data["model_type"] == "UNet"
    assert data["model_version"] == "1.0.0"
    assert data["dataset_version"] == "RICE2-v1"
    assert data["device"] == "cpu"
    assert data["in_channels"] == 3
    assert data["out_channels"] == 3
    assert data["checkpoint"] == "test_in_memory_model.pt"


# ============================================================================
# 4. Valid Inference Request & Response Schema
# ============================================================================


def test_valid_inference_request(ready_app: TestClient):
    """POST /predict must accept valid [3, 32, 32] patch and return valid schema."""
    rng = np.random.default_rng(42)
    sample_image = rng.uniform(0.05, 0.85, size=(3, 32, 32)).astype(float).tolist()

    resp = ready_app.post("/predict", json={"image": sample_image})
    assert resp.status_code == 200
    data = resp.json()

    assert "prediction" in data
    assert data["output_shape"] == [3, 32, 32]
    assert data["model_type"] == "UNet"
    assert data["checkpoint"] == "test_in_memory_model.pt"

    pred_arr = np.array(data["prediction"])
    assert pred_arr.shape == (3, 32, 32)
    assert np.all(np.isfinite(pred_arr))
    assert 0.0 <= data["prediction_min"] <= 1.0
    assert 0.0 <= data["prediction_max"] <= 1.0
    assert 0.0 <= data["prediction_mean"] <= 1.0
    assert np.isclose(pred_arr.min(), data["prediction_min"], atol=1e-5)
    assert np.isclose(pred_arr.max(), data["prediction_max"], atol=1e-5)


# ============================================================================
# 5. Invalid Dimensions Rejection
# ============================================================================


def test_invalid_dimensions_2d(ready_app: TestClient):
    """POST /predict must reject 2D image with 422."""
    image_2d = [[0.1, 0.2], [0.3, 0.4]]
    resp = ready_app.post("/predict", json={"image": image_2d})
    assert resp.status_code == 422


def test_invalid_dimensions_4d(ready_app: TestClient):
    """POST /predict must reject 4D image with 422."""
    image_4d = [[[[0.1, 0.2], [0.3, 0.4]]]]
    resp = ready_app.post("/predict", json={"image": image_4d})
    assert resp.status_code == 422


def test_ragged_rows_rejection(ready_app: TestClient):
    """POST /predict must reject ragged arrays with inconsistent row lengths."""
    # Channel 0 has rows of length 32 except one of length 31
    ragged = []
    for c in range(3):
        ch = []
        for r in range(32):
            row_len = 31 if (c == 0 and r == 5) else 32
            ch.append([0.5] * row_len)
        ragged.append(ch)

    resp = ready_app.post("/predict", json={"image": ragged})
    assert resp.status_code == 422
    assert "ragged" in resp.json()["detail"].lower() or "malformed" in resp.json()["detail"].lower()


# ============================================================================
# 6. Unsupported Channel Count Rejection
# ============================================================================


def test_unsupported_channel_count_single_channel(ready_app: TestClient):
    """POST /predict must reject 1-channel grayscale image."""
    image_1ch = [[[0.5] * 32 for _ in range(32)]]
    resp = ready_app.post("/predict", json={"image": image_1ch})
    assert resp.status_code == 422
    assert "3 channels" in resp.json()["detail"]


def test_unsupported_channel_count_four_channels(ready_app: TestClient):
    """POST /predict must reject 4-channel image (RGB+NIR)."""
    image_4ch = [[[0.5] * 32 for _ in range(32)] for _ in range(4)]
    resp = ready_app.post("/predict", json={"image": image_4ch})
    assert resp.status_code == 422
    assert "3 channels" in resp.json()["detail"]


# ============================================================================
# 7. NaN and Infinity Rejection
# ============================================================================


def test_nan_rejection(ready_app: TestClient):
    """POST /predict must reject NaN values with 422."""
    arr = np.full((3, 32, 32), 0.5, dtype=float)
    arr[0, 5, 5] = float("nan")
    payload = json.dumps({"image": arr.tolist()}, allow_nan=True)
    resp = ready_app.post("/predict", content=payload, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert "non-finite" in resp.json()["detail"].lower() or "nan" in resp.json()["detail"].lower()


def test_infinity_rejection(ready_app: TestClient):
    """POST /predict must reject Infinity values with 422."""
    arr = np.full((3, 32, 32), 0.5, dtype=float)
    arr[1, 10, 10] = float("inf")
    payload = json.dumps({"image": arr.tolist()}, allow_nan=True)
    resp = ready_app.post("/predict", content=payload, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert "non-finite" in resp.json()["detail"].lower() or "inf" in resp.json()["detail"].lower()


# ============================================================================
# 8. Oversized, Out-of-Range, or Malformed Payload Rejection
# ============================================================================


def test_negative_reflectance_rejection(ready_app: TestClient):
    """POST /predict must reject unphysical negative reflectance."""
    arr = np.full((3, 32, 32), 0.5, dtype=float)
    arr[0, 0, 0] = -0.05
    resp = ready_app.post("/predict", json={"image": arr.tolist()})
    assert resp.status_code == 422
    assert "strictly in [0.0, 1.0]" in resp.json()["detail"]


def test_unnormalized_reflectance_rejection(ready_app: TestClient):
    """POST /predict must reject unnormalized raw integer/DN reflectance (> 1.0)."""
    # Typical unscaled Sentinel-2 DN value (e.g., 2500 instead of 0.25)
    arr = np.full((3, 32, 32), 2500.0, dtype=float)
    resp = ready_app.post("/predict", json={"image": arr.tolist()})
    assert resp.status_code == 422
    assert "strictly in [0.0, 1.0]" in resp.json()["detail"]


def test_spatial_dimensions_too_small(ready_app: TestClient):
    """POST /predict must reject patches smaller than MIN_DIM (16x16)."""
    small_arr = np.full((3, 8, 8), 0.5, dtype=float)
    resp = ready_app.post("/predict", json={"image": small_arr.tolist()})
    assert resp.status_code == 422
    assert "too small" in resp.json()["detail"].lower()


def test_spatial_dimensions_exceed_max(ready_app: TestClient):
    """POST /predict must reject patches exceeding MAX_DIM (512x512)."""
    # Create ragged test with 513 height
    oversized = [[[0.5] * 32 for _ in range(513)] for _ in range(3)]
    resp = ready_app.post("/predict", json={"image": oversized})
    assert resp.status_code == 422
    assert "exceed maximum" in resp.json()["detail"].lower()


def test_empty_request_body_rejection(ready_app: TestClient):
    """POST /predict must reject empty JSON body."""
    resp = ready_app.post("/predict", json={})
    assert resp.status_code == 422


# ============================================================================
# 9. Model Loading Failure & Unavailable Model Handling
# ============================================================================


def test_model_loading_missing_checkpoint():
    """ModelManager must report clear failure when checkpoint is missing."""
    manager = ModelManager(checkpoint_path="nonexistent_checkpoint_file.pt")
    success = manager.load()
    assert success is False
    assert manager.is_ready is False
    assert manager.model is None
    assert "not found" in manager.load_error.lower()


def test_predict_when_model_unavailable(unready_app: TestClient):
    """POST /predict must return 503 when model is unavailable."""
    sample = np.full((3, 32, 32), 0.5, dtype=float).tolist()
    resp = unready_app.post("/predict", json={"image": sample})
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_model_unload_transitions_service_to_503(ready_app: TestClient):
    """Unloading the model must immediately cause /ready and /predict to return 503."""
    app_manager: ModelManager = ready_app.app.state.model_manager
    app_manager.unload()

    ready_resp = ready_app.get("/ready")
    assert ready_resp.status_code == 503

    sample = np.full((3, 32, 32), 0.5, dtype=float).tolist()
    pred_resp = ready_app.post("/predict", json={"image": sample})
    assert pred_resp.status_code == 503


# ============================================================================
# 10. Prediction Output Dimensions & Finite Values
# ============================================================================


@pytest.mark.parametrize("size", [32, 64])
def test_prediction_output_dimensions_and_finiteness(ready_app: TestClient, size: int):
    """Predictions across multiple spatial sizes must preserve shape and bounded values."""
    arr = np.full((3, size, size), 0.35, dtype=float).tolist()
    resp = ready_app.post("/predict", json={"image": arr})
    assert resp.status_code == 200
    data = resp.json()

    assert data["output_shape"] == [3, size, size]
    pred = np.array(data["prediction"])
    assert pred.shape == (3, size, size)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0.0)
    assert np.all(pred <= 1.0)


# ============================================================================
# 11. Error Handling Without Leaking Stack Traces or Local Paths
# ============================================================================


def test_safe_error_handling_no_traceback_leak(ready_app: TestClient):
    """Internal inference exceptions must return 500 without leaking stack traces or local paths."""
    app_manager: ModelManager = ready_app.app.state.model_manager

    # Mock predict method to raise an unexpected internal exception with local paths
    with patch.object(
        app_manager,
        "predict",
        side_effect=RuntimeError("Internal CUDA failure at /home/user/secret/path/cuda.cu:42"),
    ):
        sample = np.full((3, 32, 32), 0.5, dtype=float).tolist()
        resp = ready_app.post("/predict", json={"image": sample})

        assert resp.status_code == 500
        body = resp.text
        # Assert no host filesystem or traceback leaks
        assert "secret" not in body
        assert "cuda.cu" not in body
        assert "Traceback" not in body
        assert "RuntimeError" not in body
        assert "Inference execution failed" in resp.json()["detail"]


# ============================================================================
# 12. Existing Inference & Preprocessing Parity
# ============================================================================


def test_inference_parity_direct_vs_manager(tiny_model: UNet):
    """ModelManager inference must yield identical results to direct predict_patch."""
    from cloud_removal.infer import predict_patch

    manager = ModelManager(device_str="cpu")
    manager.set_model(tiny_model)

    x = torch.rand(3, 32, 32)
    direct_out = predict_patch(tiny_model, x, device=torch.device("cpu"))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    manager_out = loop.run_until_complete(manager.predict(x))
    loop.close()

    assert torch.allclose(direct_out, manager_out, atol=1e-7)


# ============================================================================
# 13. OpenAPI Documentation Endpoints
# ============================================================================


def test_openapi_docs_endpoints(ready_app: TestClient):
    """OpenAPI JSON and Swagger UI endpoints must be accessible."""
    docs_resp = ready_app.get("/docs")
    assert docs_resp.status_code == 200

    openapi_resp = ready_app.get("/openapi.json")
    assert openapi_resp.status_code == 200
    schema = openapi_resp.json()
    assert "/predict" in schema["paths"]
    assert "/predict/raster" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]


# ============================================================================
# 14. Binary Raster Upload Endpoint Tests (POST /predict/raster)
# ============================================================================


def create_synthetic_geotiff_bytes(
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    dtype: str = "float32",
    crs: str = "EPSG:32643",
    fill_value: float = 0.4,
) -> bytes:
    """Helper to generate an in-memory GeoTIFF file as bytes."""
    transform = from_origin(500000.0, 3000000.0, 10.0, 10.0)
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": channels,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
    }
    raw_data = np.full((channels, height, width), fill_value, dtype=dtype)
    with rasterio.MemoryFile() as mem:
        with mem.open(**meta) as dst:
            dst.write(raw_data)
        return bytes(mem.getbuffer())


def test_predict_raster_valid_geotiff(ready_app: TestClient):
    """POST /predict/raster must accept GeoTIFF and return downloadable GeoTIFF with headers."""
    tiff_bytes = create_synthetic_geotiff_bytes(channels=3, height=64, width=64, fill_value=0.35)

    resp = ready_app.post(
        "/predict/raster?tile_size=64&overlap=16",
        files={"file": ("test_scene.tif", tiff_bytes, "image/tiff")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/tiff"
    assert "attachment; filename=" in resp.headers["content-disposition"]
    assert resp.headers["x-model-type"] == "UNet"
    assert resp.headers["x-raster-dimensions"] == "3x64x64"
    assert resp.headers["x-raster-crs"] == "EPSG:32643"
    assert float(resp.headers["x-processing-duration-seconds"]) >= 0.0

    # Verify reconstructed GeoTIFF content
    with rasterio.MemoryFile(resp.content) as mem:
        with mem.open() as ds:
            assert ds.count == 3
            assert ds.height == 64
            assert ds.width == 64
            assert str(ds.crs) == "EPSG:32643"
            data = ds.read()
            assert np.all(np.isfinite(data))
            assert 0.0 <= np.min(data) <= 1.0
            assert 0.0 <= np.max(data) <= 1.0


def test_predict_raster_valid_geotiff_json_response(ready_app: TestClient):
    """POST /predict/raster with response_type=json must return structured metadata."""
    tiff_bytes = create_synthetic_geotiff_bytes(channels=3, height=64, width=64, fill_value=0.25)

    resp = ready_app.post(
        "/predict/raster?response_type=json&tile_size=64&overlap=16",
        files={"file": ("scene_metadata.tif", tiff_bytes, "image/tiff")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["model_type"] == "UNet"
    assert data["raster_shape"] == [3, 64, 64]
    assert data["crs"] == "EPSG:32643"
    assert data["tile_size"] == 64
    assert data["overlap"] == 16
    assert 0.0 <= data["prediction_min"] <= 1.0
    assert 0.0 <= data["prediction_max"] <= 1.0
    assert data["processing_duration_seconds"] >= 0.0


def test_predict_raster_valid_png(ready_app: TestClient):
    """POST /predict/raster must accept a valid 3-channel PNG."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    resp = ready_app.post(
        "/predict/raster?tile_size=64&overlap=16",
        files={"file": ("input.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/tiff"
    assert resp.headers["x-raster-dimensions"] == "3x64x64"


def test_predict_raster_missing_file(ready_app: TestClient):
    """POST /predict/raster without file field must return 422."""
    resp = ready_app.post("/predict/raster")
    assert resp.status_code == 422


def test_predict_raster_corrupted_file(ready_app: TestClient):
    """POST /predict/raster with corrupted file must return 422."""
    corrupted_bytes = b"NOT_A_VALID_RASTER_FILE_CONTENT"
    resp = ready_app.post(
        "/predict/raster",
        files={"file": ("corrupt.tif", corrupted_bytes, "image/tiff")},
    )
    assert resp.status_code == 422
    assert "Invalid or unsupported raster" in resp.json()["detail"]


def test_predict_raster_unsupported_channels(ready_app: TestClient):
    """POST /predict/raster with 1-channel or 4-channel GeoTIFF must return 422."""
    tiff_1band = create_synthetic_geotiff_bytes(channels=1, height=32, width=32)
    resp = ready_app.post(
        "/predict/raster",
        files={"file": ("1band.tif", tiff_1band, "image/tiff")},
    )
    assert resp.status_code == 422
    assert "Unsupported channel count" in resp.json()["detail"]


def test_predict_raster_empty_file(ready_app: TestClient):
    """POST /predict/raster with 0-byte file must return 422."""
    resp = ready_app.post(
        "/predict/raster",
        files={"file": ("empty.tif", b"", "image/tiff")},
    )
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"]


def test_predict_raster_dimensions_too_small(ready_app: TestClient):
    """POST /predict/raster with dimensions < 16 must return 422."""
    small_tiff = create_synthetic_geotiff_bytes(channels=3, height=8, width=8)
    resp = ready_app.post(
        "/predict/raster",
        files={"file": ("too_small.tif", small_tiff, "image/tiff")},
    )
    assert resp.status_code == 422
    assert "too small" in resp.json()["detail"]


def test_predict_raster_oversized_rejection(ready_app: TestClient, monkeypatch):
    """POST /predict/raster must reject files exceeding MAX_UPLOAD_SIZE_BYTES with 413."""
    # Set limit to 500 bytes for test
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "500")
    large_payload = b"X" * 1500

    resp = ready_app.post(
        "/predict/raster",
        files={"file": ("large.tif", large_payload, "image/tiff")},
    )
    assert resp.status_code == 413
    assert "exceeds maximum allowed limit" in resp.json()["detail"]


def test_predict_raster_unready_service(unready_app: TestClient):
    """POST /predict/raster must return 503 when model is uninitialized."""
    tiff_bytes = create_synthetic_geotiff_bytes(channels=3, height=32, width=32)
    resp = unready_app.post(
        "/predict/raster",
        files={"file": ("test.tif", tiff_bytes, "image/tiff")},
    )
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"]


# ============================================================================
# 15. Concurrency & Queue Overload Tests
# ============================================================================


def test_concurrent_requests_safe_execution(ready_app: TestClient):
    """Verify that concurrent inference requests succeed without deadlock or corrupted state."""
    sample_images = [
        np.full((3, 32, 32), float(val), dtype=float).tolist() for val in [0.2, 0.4, 0.6, 0.8]
    ]

    def make_request(img):
        return ready_app.post("/predict", json={"image": img})

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(make_request, sample_images))

    assert len(responses) == 4
    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert data["output_shape"] == [3, 32, 32]
        assert np.all(np.isfinite(data["prediction"]))


def test_concurrency_overload_timeout(ready_app: TestClient):
    """Verify that saturated concurrency queue times out and returns HTTP 503 Overloaded."""
    app_manager: ModelManager = ready_app.app.state.model_manager
    # Artificially limit semaphore to 1 and timeout to 0.05 seconds
    app_manager.semaphore = threading.Semaphore(1)
    app_manager.queue_timeout = 0.05

    original_model = app_manager.model
    assert original_model is not None

    class SlowModel(torch.nn.Module):
        def forward(self, x):
            time.sleep(0.3)
            return original_model(x)

    app_manager.model = SlowModel()
    try:
        sample = np.full((3, 32, 32), 0.5, dtype=float).tolist()

        def submit_request():
            return ready_app.post("/predict", json={"image": sample})

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(submit_request)
            time.sleep(0.02)  # Ensure f1 acquires semaphore first
            f2 = executor.submit(submit_request)

            resp1 = f1.result()
            resp2 = f2.result()

        statuses = {resp1.status_code, resp2.status_code}
        assert 200 in statuses
        assert 503 in statuses
        overloaded_resp = resp2 if resp2.status_code == 503 else resp1
        assert "overloaded" in overloaded_resp.json()["detail"]
        assert overloaded_resp.headers.get("Retry-After") == "5"
    finally:
        app_manager.model = original_model
        app_manager.semaphore = threading.Semaphore(2)
        app_manager.queue_timeout = 10.0
