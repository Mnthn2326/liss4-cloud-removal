"""FastAPI inference service for satellite cloud removal using U-Net baseline.

Provides production-oriented endpoints for model health, readiness, and
deterministic CPU-based cloud-removal inference on Sentinel-2 RGB patches.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel, Field
import torch

from cloud_removal.infer import load_model_from_checkpoint, predict_patch
from cloud_removal.models.unet import UNet

logger = logging.getLogger("cloud_removal.api")
logging.basicConfig(level=logging.INFO)

# Dimensions and constraints
MIN_DIM = 16
MAX_DIM = 512
DEFAULT_CHECKPOINT = "checkpoints/compact/best_compact_unet.pt"


# ============================================================================
# Pydantic Request / Response Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = Field("healthy", description="Service process liveness status")
    service: str = Field("liss4-cloud-removal-api", description="Service identifier")
    version: str = Field("0.1.0", description="API version")
    model_loaded: bool = Field(..., description="Whether a model has been successfully loaded into memory")
    ready: bool = Field(..., description="Whether the service is ready to accept inference requests")


class ReadySuccessResponse(BaseModel):
    """Readiness probe success response."""
    ready: bool = Field(True, description="Readiness status")
    status: str = Field("ready", description="Operational status")
    model_type: str = Field(..., description="Model architecture type")
    device: str = Field(..., description="Execution device")
    checkpoint: Optional[str] = Field(None, description="Active checkpoint identifier or path")
    in_channels: int = Field(3, description="Expected input channels")
    out_channels: int = Field(3, description="Produced output channels")


class ReadyUnavailableResponse(BaseModel):
    """Readiness probe unavailable response."""
    ready: bool = Field(False, description="Readiness status")
    status: str = Field("unavailable", description="Operational status")
    detail: str = Field(..., description="Reason the model is currently unavailable")


class PredictRequest(BaseModel):
    """Inference request schema.

    Expects a 3D nested numeric list shaped [channels, height, width] with:
    - channels == 3 (RGB: Sentinel-2 B04, B03, B02).
    - height and width between 16 and 512 (inclusive).
    - all values finite and normalized in [0.0, 1.0] physical surface reflectance.
    """
    image: List[List[List[float]]] = Field(
        ...,
        description=(
            "Nested numeric array representing a single 3D satellite image patch with shape "
            "[channels, height, width]. Expected channels=3 (RGB: Sentinel-2 B04, B03, B02). "
            "Values must be physical surface reflectance strictly normalized in [0.0, 1.0]."
        ),
    )


class PredictResponse(BaseModel):
    """Inference response schema containing predicted cloud-free reflectance patch."""
    prediction: List[List[List[float]]] = Field(
        ...,
        description="Predicted cloud-free surface reflectance patch shaped [3, height, width] bounded in [0.0, 1.0].",
    )
    output_shape: List[int] = Field(..., description="Predicted tensor output dimensions [channels, height, width]")
    model_type: str = Field(..., description="Architecture used for inference")
    checkpoint: Optional[str] = Field(None, description="Checkpoint path or identifier")
    prediction_min: float = Field(..., description="Minimum reflectance value in prediction")
    prediction_max: float = Field(..., description="Maximum reflectance value in prediction")
    prediction_mean: float = Field(..., description="Mean reflectance value in prediction")


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages U-Net model lifecycle, thread-safe inference, and configuration."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device_str: str = "cpu",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(device_str)
        self.model: Optional[UNet] = None
        self.model_config: Dict[str, Any] = {}
        self.is_ready: bool = False
        self.load_error: Optional[str] = None
        self.lock = asyncio.Lock()

    def load(self, checkpoint_path: Optional[Union[str, Path]] = None) -> bool:
        """Safely load model weights from the specified or configured checkpoint path."""
        if checkpoint_path is not None:
            self.checkpoint_path = str(checkpoint_path)

        if not self.checkpoint_path:
            self.is_ready = False
            self.model = None
            self.load_error = "No checkpoint path specified."
            logger.warning("ModelManager.load: %s", self.load_error)
            return False

        ckpt_p = Path(self.checkpoint_path)
        if not ckpt_p.is_file():
            self.is_ready = False
            self.model = None
            self.load_error = f"Checkpoint file not found: {ckpt_p}"
            logger.warning("ModelManager.load: %s", self.load_error)
            return False

        try:
            model, checkpoint = load_model_from_checkpoint(ckpt_p, device=self.device)
            self.model = model
            self.model_config = checkpoint.get("model_config", model.get_config())
            self.is_ready = True
            self.load_error = None
            logger.info("Successfully loaded model from %s on device %s", ckpt_p, self.device)
            return True
        except Exception as exc:
            self.is_ready = False
            self.model = None
            self.load_error = f"Failed to load checkpoint: {type(exc).__name__}: {exc}"
            logger.error("ModelManager.load: %s", self.load_error)
            return False

    def set_model(self, model: UNet, checkpoint_name: str = "in-memory") -> None:
        """Inject a pre-instantiated model instance (useful for unit tests without checkpoint files)."""
        model.eval()
        model.to(self.device)
        self.model = model
        self.model_config = model.get_config()
        self.checkpoint_path = checkpoint_name
        self.is_ready = True
        self.load_error = None

    def unload(self) -> None:
        """Unload the active model and mark the service as unavailable."""
        self.model = None
        self.is_ready = False
        self.load_error = "Model unloaded or unavailable"

    async def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute deterministic inference under an async lock to prevent concurrency conflicts."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Model is not initialized or ready for inference")

        async with self.lock:
            with torch.no_grad():
                output = predict_patch(self.model, input_tensor, device=self.device)
            return output


# ============================================================================
# FastAPI Application Factory
# ============================================================================

def create_app(model_manager: Optional[ModelManager] = None) -> FastAPI:
    """Create and configure the FastAPI inference application."""

    # Initialize manager if not explicitly injected
    manager = model_manager if model_manager is not None else ModelManager(
        device_str=os.getenv("DEVICE", "cpu")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: attempt to load checkpoint if not already injected
        if not manager.is_ready:
            env_ckpt = os.getenv("MODEL_CHECKPOINT")
            if env_ckpt:
                manager.load(env_ckpt)
            else:
                default_p = Path(DEFAULT_CHECKPOINT)
                if default_p.is_file():
                    manager.load(str(default_p))
                else:
                    manager.is_ready = False
                    manager.load_error = (
                        f"Default checkpoint not found at {default_p} and MODEL_CHECKPOINT env var not set."
                    )
                    logger.info("Lifespan: %s", manager.load_error)
        yield
        # Shutdown cleanup
        logger.info("Application shutting down.")

    app = FastAPI(
        title="LISS-IV Cloud Removal Inference Service",
        description=(
            "Production-oriented FastAPI inference service for optical satellite imagery cloud removal "
            "using a compact U-Net baseline trained on Sentinel-2 SEN12MS-CR data.\n\n"
            "**Scientific Boundary Notice**: Cross-sensor inference on Resourcesat-2A LISS-IV imagery remains "
            "unvalidated due to spectral band differences, spatial resolution differences (5m vs 10m), and "
            "unresolved radiometric calibration parameters. The model expects normalized Sentinel-2 RGB surface "
            "reflectance in [0.0, 1.0]."
        ),
        version="0.1.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Attach manager to app state for dependency access
    app.state.model_manager = manager

    # ------------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------------
    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Liveness Probe",
        description="Returns basic process health and indicates whether a model has been loaded into memory.",
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service="liss4-cloud-removal-api",
            version="0.1.0",
            model_loaded=manager.model is not None,
            ready=manager.is_ready,
        )

    # ------------------------------------------------------------------------
    # GET /ready
    # ------------------------------------------------------------------------
    @app.get(
        "/ready",
        response_model=ReadySuccessResponse,
        responses={
            status.HTTP_200_OK: {"model": ReadySuccessResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyUnavailableResponse},
        },
        summary="Service Readiness Probe",
        description=(
            "Returns HTTP 200 when the model is loaded and ready to serve inference. "
            "Returns HTTP 503 if the model is unavailable or failed to initialize."
        ),
    )
    async def ready() -> Response:
        if manager.is_ready and manager.model is not None:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "ready": True,
                    "status": "ready",
                    "model_type": manager.model_config.get("model_type", "UNet"),
                    "device": str(manager.device),
                    "checkpoint": manager.checkpoint_path,
                    "in_channels": manager.model_config.get("in_channels", 3),
                    "out_channels": manager.model_config.get("out_channels", 3),
                },
            )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "status": "unavailable",
                "detail": manager.load_error or "Model is not loaded or unavailable for inference.",
            },
        )

    # ------------------------------------------------------------------------
    # POST /predict
    # ------------------------------------------------------------------------
    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            status.HTTP_200_OK: {"model": PredictResponse},
            422: {"description": "Malformed input array, unsupported channels, or out-of-range reflectance"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model is not loaded or unavailable"},
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Inference execution error"},
        },
        summary="Run Cloud Removal Inference",
        description=(
            "Executes deterministic cloud removal inference on a single 3D cloudy patch [3, H, W].\n\n"
            "**Requirements**:\n"
            "- Exactly 3 channels: RGB (Sentinel-2 B04, B03, B02).\n"
            "- Height and Width between 16 and 512.\n"
            "- All values strictly finite and normalized in [0.0, 1.0] physical surface reflectance."
        ),
    )
    async def predict(request: PredictRequest) -> PredictResponse:
        # 1. Verify model readiness
        if not manager.is_ready or manager.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded or unavailable for inference.",
            )

        raw_image = request.image

        # 2. Outer dimension checks
        if not isinstance(raw_image, list) or len(raw_image) != 3:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported channel count: expected 3 channels (RGB: [B04, B03, B02]), got {len(raw_image) if isinstance(raw_image, list) else 'non-list'}",
            )

        h_len = len(raw_image[0]) if len(raw_image) > 0 and isinstance(raw_image[0], list) else 0
        if h_len == 0:
            raise HTTPException(
                status_code=422,
                detail="Malformed image dimensions: height dimension must be non-empty.",
            )

        w_len = len(raw_image[0][0]) if h_len > 0 and isinstance(raw_image[0][0], list) else 0
        if w_len == 0:
            raise HTTPException(
                status_code=422,
                detail="Malformed image dimensions: width dimension must be non-empty.",
            )

        # 3. Ragged row / channel validation
        for c_idx, ch in enumerate(raw_image):
            if not isinstance(ch, list) or len(ch) != h_len:
                raise HTTPException(
                    status_code=422,
                    detail=f"Malformed image dimensions: channel {c_idx} has height {len(ch) if isinstance(ch, list) else 'invalid'}, expected {h_len}.",
                )
            for r_idx, row in enumerate(ch):
                if not isinstance(row, list) or len(row) != w_len:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Malformed image dimensions: ragged row at channel {c_idx}, row {r_idx} (len {len(row) if isinstance(row, list) else 'invalid'} vs expected {w_len}).",
                    )

        # 4. Dimension bounds
        if h_len < MIN_DIM or w_len < MIN_DIM:
            raise HTTPException(
                status_code=422,
                detail=f"Image dimensions ({h_len}x{w_len}) too small: minimum spatial size is {MIN_DIM}x{MIN_DIM}.",
            )
        if h_len > MAX_DIM or w_len > MAX_DIM:
            raise HTTPException(
                status_code=422,
                detail=f"Image dimensions ({h_len}x{w_len}) exceed maximum allowed size of {MAX_DIM}x{MAX_DIM}.",
            )

        # 5. Numerical conversion and finite value checks
        try:
            arr = np.array(raw_image, dtype=np.float32)
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="Malformed numeric array in request payload.",
            )

        if not np.all(np.isfinite(arr)):
            raise HTTPException(
                status_code=422,
                detail="Input array contains non-finite values (NaN or Inf).",
            )

        # 6. Physical reflectance range validation [0.0, 1.0]
        # Do NOT silently clip or scale: reject invalid values explicitly
        min_val = float(arr.min())
        max_val = float(arr.max())
        if min_val < 0.0 or max_val > 1.0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Input values must be physical surface reflectance normalized strictly in [0.0, 1.0]. "
                    f"Observed value range: [{min_val:.5f}, {max_val:.5f}]."
                ),
            )

        # 7. Execute inference
        input_tensor = torch.from_numpy(arr)  # shape [3, H, W]
        try:
            output_tensor = await manager.predict(input_tensor)
        except Exception as exc:
            logger.error("Inference execution failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference execution failed due to an internal error.",
            )

        pred_arr = output_tensor.numpy()

        return PredictResponse(
            prediction=pred_arr.tolist(),
            output_shape=list(pred_arr.shape),
            model_type=manager.model_config.get("model_type", "UNet"),
            checkpoint=manager.checkpoint_path,
            prediction_min=float(pred_arr.min()),
            prediction_max=float(pred_arr.max()),
            prediction_mean=float(pred_arr.mean()),
        )

    return app


# ============================================================================
# Default ASGI Application Entrypoint
# ============================================================================

app = create_app()
