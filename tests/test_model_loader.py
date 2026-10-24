"""Unit tests for dynamic model loading module (Phase 4).

Tests local checkpoint loading, MLflow Model Registry mock interactions,
S3 artifact downloading with boto3 mocks, warm-up validation, and caching.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from cloud_removal.model_loader import (
    ModelLoadError,
    download_s3_checkpoint,
    load_from_mlflow,
    load_model_dynamically,
    parse_s3_uri,
    validate_and_warmup_model,
)
from cloud_removal.models.unet import UNet


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Save a minimal valid UNet checkpoint to disk for testing."""
    model = UNet(in_channels=3, out_channels=3, base_channels=4)
    ckpt_file = tmp_path / "test_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "epoch": 1,
            "val_mae": 0.05,
        },
        ckpt_file,
    )
    return ckpt_file


# ============================================================================
# 1. Local Checkpoint Loading Tests
# ============================================================================


def test_load_local_checkpoint_success(dummy_checkpoint: Path):
    """Test successful dynamic loading from local checkpoint path."""
    model, cfg, meta = load_model_dynamically(
        config={"source": "local", "checkpoint_path": str(dummy_checkpoint)},
        device_str="cpu",
    )
    assert isinstance(model, torch.nn.Module)
    assert cfg.get("in_channels") == 3
    assert meta["source"] == "local"
    assert meta["checkpoint_path"] == str(dummy_checkpoint)


def test_load_local_checkpoint_missing(tmp_path: Path):
    """Test that a missing checkpoint file raises ModelLoadError with actionable message."""
    missing = tmp_path / "non_existent.pt"
    with pytest.raises(ModelLoadError, match="not found"):
        load_model_dynamically(
            config={"source": "local", "checkpoint_path": str(missing)},
            device_str="cpu",
        )


def test_unsupported_source():
    """Test that an invalid source raises ModelLoadError."""
    with pytest.raises(ModelLoadError, match="Unsupported model source"):
        load_model_dynamically(config={"source": "gcs_bucket"})


# ============================================================================
# 2. S3 Integration Tests (Mocked)
# ============================================================================


def test_parse_s3_uri():
    """Verify S3 URI parsing into bucket and key."""
    bucket, key = parse_s3_uri("s3://my-bucket/checkpoints/best_model.pt")
    assert bucket == "my-bucket"
    assert key == "checkpoints/best_model.pt"

    with pytest.raises(ModelLoadError, match="Invalid S3 URI"):
        parse_s3_uri("http://example.com/model.pt")


@patch("boto3.client")
def test_download_s3_checkpoint_mocked(mock_boto_client, tmp_path: Path):
    """Verify S3 download via boto3 client."""
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    def fake_download(bucket, key, dest):
        # Create a non-empty file at dest
        Path(dest).write_bytes(b"dummy_weights")

    mock_s3.download_file.side_effect = fake_download

    cache_dir = tmp_path / "s3_cache"
    dest = download_s3_checkpoint("s3://test-bucket/models/unet.pt", cache_dir=cache_dir)

    assert dest.is_file()
    assert dest.read_bytes() == b"dummy_weights"
    mock_s3.download_file.assert_called_once_with("test-bucket", "models/unet.pt", str(dest))

    # Second call should use cache without invoking s3.download_file again
    mock_s3.download_file.reset_mock()
    dest2 = download_s3_checkpoint("s3://test-bucket/models/unet.pt", cache_dir=cache_dir)
    assert dest2 == dest
    mock_s3.download_file.assert_not_called()


@patch("cloud_removal.model_loader.download_s3_checkpoint")
def test_load_dynamically_s3_mocked(mock_download, dummy_checkpoint: Path):
    """Test full dynamic load flow from S3 with mocked download."""
    mock_download.return_value = dummy_checkpoint

    model, cfg, meta = load_model_dynamically(
        config={"source": "s3", "s3_uri": "s3://bucket/model.pt"},
        device_str="cpu",
    )
    assert isinstance(model, torch.nn.Module)
    assert meta["source"] == "s3"
    assert meta["s3_uri"] == "s3://bucket/model.pt"


# ============================================================================
# 3. MLflow Integration Tests (Mocked)
# ============================================================================


@patch("mlflow.pytorch.load_model")
def test_load_from_mlflow_with_alias(mock_mlflow_load):
    """Test MLflow model loading with model alias."""
    dummy_model = UNet(in_channels=3, out_channels=3, base_channels=4)
    mock_mlflow_load.return_value = dummy_model

    model, cfg, meta = load_from_mlflow(
        model_name="cloud-removal-unet",
        alias="champion",
        tracking_uri="http://localhost:5000",
        device=torch.device("cpu"),
    )

    assert model is dummy_model
    assert meta["model_uri"] == "models:/cloud-removal-unet@champion"
    assert meta["identifier"] == "cloud-removal-unet@champion"
    mock_mlflow_load.assert_called_once_with(
        "models:/cloud-removal-unet@champion", map_location=torch.device("cpu")
    )


@patch("mlflow.pytorch.load_model")
def test_load_from_mlflow_with_version(mock_mlflow_load):
    """Test MLflow model loading with specific version."""
    dummy_model = UNet(in_channels=3, out_channels=3, base_channels=4)
    mock_mlflow_load.return_value = dummy_model

    model, cfg, meta = load_from_mlflow(
        model_name="cloud-removal-unet",
        version="2",
        device=torch.device("cpu"),
    )
    assert meta["model_uri"] == "models:/cloud-removal-unet/2"


# ============================================================================
# 4. Warm-Up & Validation Tests
# ============================================================================


def test_validate_and_warmup_success():
    """Verify successful warm-up on valid UNet."""
    model = UNet(in_channels=3, out_channels=3, base_channels=4)
    validate_and_warmup_model(model, in_channels=3, device=torch.device("cpu"))


def test_validate_and_warmup_failure():
    """Verify warm-up failure detection when forward pass raises an exception."""

    class BrokenModel(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("CUDA out of memory simulation")

    broken = BrokenModel()
    with pytest.raises(ModelLoadError, match="warm-up forward pass failed"):
        validate_and_warmup_model(broken, in_channels=3, device=torch.device("cpu"))
