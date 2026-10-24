"""Unit tests for MLflow model logging, signature inference, and model registry workflows."""

import os
from pathlib import Path
from typing import Generator

import mlflow
import mlflow.pytorch
import numpy as np
import pytest
import torch
from mlflow.models.signature import infer_signature

from cloud_removal.models.unet import UNet
from cloud_removal.registry import get_model_version_details, promote_model


@pytest.fixture(autouse=True)
def setup_mlflow_env() -> Generator[None, None, None]:
    """Ensure MLflow file store and tracing hints are configured for testing."""
    old_allow = os.environ.get("MLFLOW_ALLOW_FILE_STORE")
    old_hint = os.environ.get("MLFLOW_DISABLE_AGENT_HINT")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    yield
    if old_allow is not None:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = old_allow
    else:
        os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
    if old_hint is not None:
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = old_hint
    else:
        os.environ.pop("MLFLOW_DISABLE_AGENT_HINT", None)


def test_signature_inference_and_model_logging(tmp_path: Path) -> None:
    """Test model signature inference and logging with PyTorch model and metadata."""
    tracking_uri = f"file:///{tmp_path.as_posix()}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-model-logging")

    model = UNet(in_channels=3, out_channels=3, base_channels=4)
    model.eval()

    input_example = np.random.uniform(0.0, 1.0, size=(1, 3, 64, 64)).astype(np.float32)
    with torch.no_grad():
        output_example = model(torch.from_numpy(input_example)).cpu().numpy()

    signature = infer_signature(input_example, output_example)
    assert signature is not None
    assert signature.inputs is not None
    assert signature.outputs is not None

    metadata = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 4,
        "composite_mode": "rgb",
    }

    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            metadata=metadata,
        )
        assert model_info is not None
        assert model_info.model_uri is not None
        assert "model" in model_info.model_uri


def test_promote_model_alias_and_stage(tmp_path: Path) -> None:
    """Test model promotion with alias and stage on SQLite-backed registry."""
    db_path = tmp_path / "mlflow_test.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-promotion")

    model = UNet(in_channels=3, out_channels=3, base_channels=4)
    model.eval()
    input_example = np.random.uniform(0.0, 1.0, size=(1, 3, 32, 32)).astype(np.float32)

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=input_example,
            registered_model_name="CloudRemovalTestModel",
        )

    # Promote to Staging with alias 'challenger'
    res = promote_model(
        model_name="CloudRemovalTestModel",
        version=1,
        stage="Staging",
        alias="challenger",
        tracking_uri=tracking_uri,
    )
    assert res["status"] == "success"
    assert res["stage"] == "Staging"
    assert res["alias"] == "challenger"

    # Promote to Production with alias 'champion'
    res_prod = promote_model(
        model_name="CloudRemovalTestModel",
        version=1,
        stage="Production",
        alias="champion",
        tracking_uri=tracking_uri,
    )
    assert res_prod["status"] == "success"
    assert res_prod["stage"] == "Production"
    assert res_prod["alias"] == "champion"

    # Query details
    details = get_model_version_details(
        model_name="CloudRemovalTestModel",
        version_or_alias="champion",
        tracking_uri=tracking_uri,
    )
    assert details["name"] == "CloudRemovalTestModel"
    assert details["current_stage"] == "Production"


def test_promote_model_missing_parameters() -> None:
    """promote_model must raise ValueError if neither stage nor alias is provided."""
    with pytest.raises(ValueError, match="At least one of 'stage' or 'alias'"):
        promote_model(model_name="DummyModel", version=1)
