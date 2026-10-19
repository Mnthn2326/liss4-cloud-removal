"""Tests for the interactive Web UI and associated API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
import torch

from cloud_removal.api import ModelManager, create_app
from cloud_removal.models.unet import UNet


def create_test_model() -> UNet:
    """Create a lightweight deterministic UNet model for testing without checkpoints."""
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=8,
        bilinear=True,
        use_sigmoid=True,
    )
    model.eval()
    return model


def make_png_bytes(h: int = 32, w: int = 32, color=(120, 150, 180)) -> bytes:
    """Create an in-memory RGB PNG file bytes."""
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_geotiff_bytes(count: int = 3, h: int = 32, w: int = 32, dtype="float32", val: float = 0.25) -> bytes:
    """Create an in-memory GeoTIFF raster with optional CRS and Affine transform."""
    data = np.full((count, h, w), val, dtype=np.float32)
    with MemoryFile() as mem:
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": count,
            "dtype": dtype,
            "crs": "EPSG:32632",
            "transform": rasterio.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0),
        }
        if dtype == "uint16":
            profile["dtype"] = "uint16"
            data = (data * 10000.0).astype(np.uint16)
        elif dtype == "uint8":
            profile["dtype"] = "uint8"
            data = (data * 255.0).astype(np.uint8)

        with mem.open(**profile) as dst:
            dst.write(data)
        return mem.read()


@pytest.fixture
def test_client():
    """FastAPI TestClient fixture with in-memory U-Net initialized."""
    manager = ModelManager(device_str="cpu")
    model = create_test_model()
    manager.set_model(model, checkpoint_name="test_in_memory_unet")
    app = create_app(model_manager=manager)
    with TestClient(app) as client:
        yield client


# ============================================================================
# Web UI HTML & Preset Route Tests
# ============================================================================

def test_get_ui_html(test_client):
    """Verify that GET / and GET /ui serve the responsive HTML interface."""
    for path in ("/", "/ui"):
        resp = test_client.get(path)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text
        # Verify key elements
        assert "Satellite Optical Cloud Removal" in html
        assert "LISS-IV" in html
        assert "Sentinel-2" in html
        assert "Split Slider" in html
        assert "Side-by-Side" in html
        assert "Sensor & Domain Boundary Notice" in html


def test_get_presets(test_client):
    """Verify that GET /api/ui/presets returns valid test samples."""
    resp = test_client.get("/api/ui/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert isinstance(presets, list)
    assert len(presets) >= 3

    # Check required fields
    for p in presets:
        assert "id" in p
        assert "name" in p
        assert "category" in p
        assert "is_synthetic" in p
        assert "has_ground_truth" in p


# ============================================================================
# Interactive Inference: Presets
# ============================================================================

def test_predict_real_preset(test_client):
    """Verify prediction with real SEN12MS-CR benchmark sample."""
    resp = test_client.post("/api/ui/predict", data={"preset_id": "sen12mscr_val_0"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_synthetic"] is False
    assert data["has_ground_truth"] is True
    assert data["metrics"] is not None
    assert "input_mae" in data["metrics"]
    assert "pred_mae" in data["metrics"]
    assert "delta_mae" in data["metrics"]
    assert "input_psnr" in data["metrics"]
    assert "pred_psnr" in data["metrics"]
    assert "delta_psnr" in data["metrics"]
    assert data["cloudy_preview"].startswith("data:image/png;base64,")
    assert data["pred_preview"].startswith("data:image/png;base64,")
    assert data["clear_preview"].startswith("data:image/png;base64,")
    assert data["latency_ms"] > 0
    assert "session_id" in data


def test_predict_synthetic_preset(test_client):
    """Verify prediction with synthetic cloud simulation sample clearly labeled."""
    resp = test_client.post("/api/ui/predict", data={"preset_id": "synthetic_dense_0"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_synthetic"] is True
    assert data["has_ground_truth"] is True
    assert data["metrics"] is not None
    assert data["cloudy_preview"].startswith("data:image/png;base64,")
    assert data["pred_preview"].startswith("data:image/png;base64,")


def test_predict_invalid_preset(test_client):
    """Verify 404 error when invalid preset_id is passed."""
    resp = test_client.post("/api/ui/predict", data={"preset_id": "nonexistent_preset_xyz"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ============================================================================
# Interactive Inference: File Uploads
# ============================================================================

def test_predict_upload_png_without_ground_truth(test_client):
    """Verify that uploading an optical RGB image without ground truth omits MAE/PSNR."""
    png_bytes = make_png_bytes(h=32, w=32)
    resp = test_client.post(
        "/api/ui/predict",
        files={"file": ("test_cloudy.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["has_ground_truth"] is False
    assert data["metrics"] is None
    assert data["clear_preview"] is None
    assert data["is_geotiff"] is False
    assert data["crs"] is None
    assert data["input_shape"] == [3, 32, 32]
    assert data["output_shape"] == [3, 32, 32]
    assert data["cloudy_preview"].startswith("data:image/png;base64,")
    assert data["pred_preview"].startswith("data:image/png;base64,")


def test_predict_upload_png_with_ground_truth(test_client):
    """Verify that uploading an image along with a clear reference computes MAE and PSNR."""
    cloudy_png = make_png_bytes(h=32, w=32, color=(200, 200, 200))
    clear_png = make_png_bytes(h=32, w=32, color=(50, 100, 50))

    resp = test_client.post(
        "/api/ui/predict",
        files={
            "file": ("cloudy.png", cloudy_png, "image/png"),
            "clear_file": ("clear.png", clear_png, "image/png"),
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["has_ground_truth"] is True
    assert data["metrics"] is not None
    assert data["clear_preview"] is not None
    assert data["metrics"]["input_mae"] > 0
    assert data["metrics"]["pred_mae"] >= 0


def test_predict_upload_geotiff_3band(test_client):
    """Verify that uploading a 3-band GeoTIFF preserves CRS and Affine metadata."""
    tiff_bytes = make_geotiff_bytes(count=3, h=32, w=32, dtype="float32", val=0.3)
    resp = test_client.post(
        "/api/ui/predict",
        files={"file": ("scene.tif", tiff_bytes, "image/tiff")},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_geotiff"] is True
    assert "32632" in (data["crs"] or "")
    assert data["input_shape"] == [3, 32, 32]


def test_predict_upload_single_band_geotiff_rejected(test_client):
    """Verify that single-band rasters (such as raw LISS-IV individual bands) are rejected with 422."""
    single_band_bytes = make_geotiff_bytes(count=1, h=32, w=32)
    resp = test_client.post(
        "/api/ui/predict",
        files={"file": ("BAND2.tif", single_band_bytes, "image/tiff")},
    )
    assert resp.status_code == 422
    err_msg = resp.json()["detail"]
    assert "Single-band raster detected" in err_msg
    assert "LISS-IV" in err_msg


def test_predict_upload_unsupported_band_count_rejected(test_client):
    """Verify that 2-band or 4-band rasters are rejected with 422."""
    four_band_bytes = make_geotiff_bytes(count=4, h=32, w=32)
    resp = test_client.post(
        "/api/ui/predict",
        files={"file": ("four_band.tif", four_band_bytes, "image/tiff")},
    )
    assert resp.status_code == 422
    assert "Unsupported raster channel count" in resp.json()["detail"]


def test_predict_upload_too_small_rejected(test_client):
    """Verify that spatial dimensions below 16x16 are rejected."""
    small_bytes = make_png_bytes(h=8, w=8)
    resp = test_client.post(
        "/api/ui/predict",
        files={"file": ("tiny.png", small_bytes, "image/png")},
    )
    assert resp.status_code == 422
    assert "too small" in resp.json()["detail"].lower()


def test_predict_no_input_rejected(test_client):
    """Verify that posting without file or preset_id returns 400."""
    resp = test_client.post("/api/ui/predict", data={})
    assert resp.status_code == 400


# ============================================================================
# Download Prediction Exports
# ============================================================================

def test_download_prediction_png_and_geotiff(test_client):
    """Verify downloading predictions in PNG and GeoTIFF formats."""
    tiff_bytes = make_geotiff_bytes(count=3, h=32, w=32)
    predict_resp = test_client.post(
        "/api/ui/predict",
        files={"file": ("scene.tif", tiff_bytes, "image/tiff")},
    )
    assert predict_resp.status_code == 200
    session_id = predict_resp.json()["session_id"]

    # Download PNG
    png_resp = test_client.get(f"/api/ui/download/{session_id}?format=png")
    assert png_resp.status_code == 200
    assert png_resp.headers["content-type"] == "image/png"
    # Ensure it is a valid PNG
    img = Image.open(io.BytesIO(png_resp.content))
    assert img.size == (32, 32)

    # Download GeoTIFF
    tif_resp = test_client.get(f"/api/ui/download/{session_id}?format=geotiff")
    assert tif_resp.status_code == 200
    assert tif_resp.headers["content-type"] == "image/tiff"
    with MemoryFile(tif_resp.content) as mem:
        with mem.open() as src:
            assert src.count == 3
            assert src.height == 32
            assert src.width == 32
            assert "32632" in (src.crs.to_string() if src.crs else "")


def test_download_nonexistent_session(test_client):
    """Verify 404 when downloading an invalid or expired session."""
    resp = test_client.get("/api/ui/download/invalid_session_id?format=png")
    assert resp.status_code == 404
