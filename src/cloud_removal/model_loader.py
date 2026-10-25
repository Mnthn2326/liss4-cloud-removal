"""Dynamic model loader supporting local checkpoints, MLflow Model Registry, and S3.

Provides secure, cached, validated dynamic model initialization with warm-up checks
and actionable error diagnostics for production deployment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import torch

from cloud_removal.infer import load_model_from_checkpoint

logger = logging.getLogger("cloud_removal.model_loader")


class ModelLoadError(RuntimeError):
    """Raised when model loading, validation, or artifact retrieval fails."""

    pass


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse an s3://bucket/key URI into (bucket, key)."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ModelLoadError(
            f"Invalid S3 URI '{s3_uri}'. Expected format: 's3://<bucket>/<path-to-checkpoint>'"
        )
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def download_s3_checkpoint(
    s3_uri: str,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
) -> Path:
    """Download model checkpoint from an S3 URI with local caching.

    Uses standard AWS credential resolution (environment variables, IAM roles,
    AWS profile) via boto3 without hardcoded credentials.

    Args:
        s3_uri: S3 URI (s3://bucket/key).
        cache_dir: Local directory for caching artifacts (default: .cache/models).
        force_download: If True, re-download even if file is cached.

    Returns:
        Local Path to the downloaded checkpoint file.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        raise ModelLoadError(
            "boto3 and botocore are required for S3 model loading. Install with `pip install boto3`."
        )

    bucket, key = parse_s3_uri(s3_uri)
    c_dir = Path(cache_dir) if cache_dir else Path(".cache/models")
    dest_path = c_dir / bucket / key
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.is_file() and not force_download and dest_path.stat().st_size > 0:
        logger.info("Using cached S3 model artifact from %s", dest_path)
        return dest_path

    logger.info("Downloading model artifact from s3://%s/%s to %s", bucket, key, dest_path)
    try:
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(dest_path))
    except NoCredentialsError as exc:
        raise ModelLoadError(
            f"Failed to access S3 URI '{s3_uri}': No AWS credentials found. "
            "Configure AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or assign an IAM role."
        ) from exc
    except ClientError as exc:
        err_code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise ModelLoadError(
            f"S3 download failed for '{s3_uri}' (Error {err_code}): {exc}"
        ) from exc
    except Exception as exc:
        raise ModelLoadError(f"Failed to download model artifact from '{s3_uri}': {exc}") from exc

    if not dest_path.is_file() or dest_path.stat().st_size == 0:
        raise ModelLoadError(f"Downloaded S3 artifact at {dest_path} is empty or missing.")

    return dest_path


def load_from_mlflow(
    model_name: str,
    version: Optional[str] = None,
    alias: Optional[str] = None,
    stage: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, str]]:
    """Load model from MLflow Model Registry using version, alias, or stage.

    Args:
        model_name: Registered model name.
        version: Model version number (e.g., '1').
        alias: Model alias (e.g., 'champion', 'challenger').
        stage: Model stage (e.g., 'Production', 'Staging').
        tracking_uri: MLflow tracking server URI.
        device: PyTorch device.

    Returns:
        Tuple of (model, config, metadata_dict).
    """
    try:
        import mlflow
        import mlflow.pytorch
    except ImportError:
        raise ModelLoadError(
            "mlflow is required for MLflow model loading. Install with `pip install mlflow`."
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if alias:
        model_uri = f"models:/{model_name}@{alias}"
        identifier = f"{model_name}@{alias}"
    elif version:
        model_uri = f"models:/{model_name}/{version}"
        identifier = f"{model_name}/v{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
        identifier = f"{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"
        identifier = f"{model_name}/latest"

    logger.info("Loading model from MLflow URI: %s", model_uri)
    try:
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load model from MLflow Model Registry at '{model_uri}': {exc}"
        ) from exc

    if not isinstance(model, torch.nn.Module):
        raise ModelLoadError(
            f"Loaded MLflow artifact from '{model_uri}' is not a valid PyTorch Module."
        )

    model.eval()
    if device is not None:
        model.to(device)

    # Extract model config if available
    raw_cfg: Any = getattr(model, "get_config", lambda: {})()
    config: Dict[str, Any] = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
    if not config and hasattr(model, "in_channels"):
        config = {
            "model_type": type(model).__name__,
            "in_channels": getattr(model, "in_channels", 3),
            "out_channels": getattr(model, "out_channels", 3),
        }

    meta = {
        "source": "mlflow",
        "model_uri": model_uri,
        "identifier": identifier,
        "tracking_uri": tracking_uri or mlflow.get_tracking_uri(),
    }
    return model, config, meta


def validate_and_warmup_model(
    model: torch.nn.Module,
    in_channels: int = 3,
    device: Optional[torch.device] = None,
) -> None:
    """Run a lightweight dry-run inference pass to validate model architecture and warm up device."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    dummy_input = torch.zeros((1, in_channels, 64, 64), dtype=torch.float32, device=device)
    model.eval()
    try:
        with torch.inference_mode():
            output = model(dummy_input)
    except Exception as exc:
        raise ModelLoadError(
            f"Model warm-up forward pass failed with input shape {dummy_input.shape}: {exc}"
        ) from exc

    if output.ndim != 4 or output.shape[0] != 1:
        raise ModelLoadError(
            f"Model produced invalid output tensor shape: expected 4D batch [1, C, H, W], got {output.shape}"
        )


def load_model_dynamically(
    config: Optional[Dict[str, Any]] = None,
    device_str: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, str]]:
    """Dynamically resolve and load a cloud-removal model from local, MLflow, or S3.

    Configuration can be passed explicitly or read from environment variables:
    - MODEL_SOURCE: 'local' (default), 'mlflow', or 's3'
    - MODEL_CHECKPOINT: local file path
    - MODEL_S3_URI: s3:// URI
    - MLFLOW_TRACKING_URI: MLflow tracking server URI
    - MLFLOW_MODEL_NAME: Registered model name
    - MLFLOW_MODEL_VERSION: Registered model version
    - MLFLOW_MODEL_ALIAS: Registered model alias (e.g. 'champion')
    - MLFLOW_MODEL_STAGE: Legacy model stage (e.g. 'Production')
    - DEVICE: 'cpu', 'cuda', or 'auto'

    Returns:
        Tuple of (model, model_config, source_metadata).
    """
    cfg = config or {}
    source_val = cfg.get("source") or os.getenv("MODEL_SOURCE", "local")
    source = str(source_val).lower().strip()
    dev_str = str(device_str or cfg.get("device") or os.getenv("DEVICE", "cpu"))

    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    model: torch.nn.Module
    model_cfg: Dict[str, Any]
    meta: Dict[str, str]

    if source == "local":
        ckpt_path_str = (
            cfg.get("checkpoint_path")
            or os.getenv("MODEL_CHECKPOINT")
            or (
                "checkpoints/best_unet.pt"
                if Path("checkpoints/best_unet.pt").is_file()
                else "checkpoints/compact/best_compact_unet.pt"
            )
        )
        ckpt_p = Path(ckpt_path_str)
        if not ckpt_p.is_file():
            raise ModelLoadError(
                f"Local model checkpoint not found at: {ckpt_p.resolve()}. "
                "Ensure the file exists or set MODEL_CHECKPOINT to a valid path."
            )
        try:
            model, ckpt_dict = load_model_from_checkpoint(ckpt_p, device=device)
            model_cfg = ckpt_dict.get("model_config", model.get_config())
        except Exception as exc:
            raise ModelLoadError(f"Failed to load local checkpoint '{ckpt_p}': {exc}") from exc

        meta = {
            "source": "local",
            "checkpoint_path": str(ckpt_p),
            "identifier": str(ckpt_p),
        }

    elif source == "s3":
        s3_uri = cfg.get("s3_uri") or os.getenv("MODEL_S3_URI")
        if not s3_uri:
            raise ModelLoadError(
                "MODEL_SOURCE is 's3', but neither 's3_uri' config nor MODEL_S3_URI env var is set."
            )
        cache_dir = cfg.get("cache_dir") or os.getenv("MODEL_CACHE_DIR", ".cache/models")
        local_path = download_s3_checkpoint(s3_uri, cache_dir=cache_dir)
        try:
            model, ckpt_dict = load_model_from_checkpoint(local_path, device=device)
            model_cfg = ckpt_dict.get("model_config", model.get_config())
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load checkpoint downloaded from S3 '{s3_uri}' ({local_path}): {exc}"
            ) from exc

        meta = {
            "source": "s3",
            "s3_uri": s3_uri,
            "cached_path": str(local_path),
            "identifier": s3_uri,
        }

    elif source == "mlflow":
        model_name = cfg.get("mlflow_model_name") or os.getenv("MLFLOW_MODEL_NAME")
        if not model_name:
            raise ModelLoadError(
                "MODEL_SOURCE is 'mlflow', but neither 'mlflow_model_name' config nor MLFLOW_MODEL_NAME env var is set."
            )
        version = cfg.get("mlflow_model_version") or os.getenv("MLFLOW_MODEL_VERSION")
        alias = cfg.get("mlflow_model_alias") or os.getenv("MLFLOW_MODEL_ALIAS")
        stage = cfg.get("mlflow_model_stage") or os.getenv("MLFLOW_MODEL_STAGE")
        tracking_uri = cfg.get("mlflow_tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")

        model, model_cfg, meta = load_from_mlflow(
            model_name=model_name,
            version=version,
            alias=alias,
            stage=stage,
            tracking_uri=tracking_uri,
            device=device,
        )

    else:
        raise ModelLoadError(
            f"Unsupported model source '{source}'. Supported sources: 'local', 'mlflow', 's3'."
        )

    in_ch = model_cfg.get("in_channels", 3)
    validate_and_warmup_model(model, in_channels=in_ch, device=device)
    logger.info("Model loaded and validated successfully from source: %s", meta["identifier"])

    return model, model_cfg, meta
