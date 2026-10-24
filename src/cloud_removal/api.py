"""FastAPI inference service for satellite cloud removal using U-Net baseline.

Provides production-oriented endpoints for model health, readiness, and
deterministic CPU-based cloud-removal inference on Sentinel-2 RGB patches.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import numpy as np
import rasterio
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from cloud_removal.infer import (
    generate_tile_coordinates,
    predict_patch,
    predict_tiled_array,
    predict_tiled_raster,
)
from cloud_removal.model_loader import load_model_dynamically

logger = logging.getLogger("cloud_removal.api")
logging.basicConfig(level=logging.INFO)

# Dimensions and constraints
MIN_DIM = 16
MAX_DIM = 512
DEFAULT_CHECKPOINT = "checkpoints/compact/best_compact_unet.pt"


class ServiceOverloadedError(RuntimeError):
    """Raised when inference queue / concurrency limit is saturated."""

    pass


# ============================================================================
# Pydantic Request / Response Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field("healthy", description="Service process liveness status")
    service: str = Field("liss4-cloud-removal-api", description="Service identifier")
    version: str = Field("0.1.0", description="API version")
    model_loaded: bool = Field(
        ..., description="Whether a model has been successfully loaded into memory"
    )
    ready: bool = Field(
        ..., description="Whether the service is ready to accept inference requests"
    )


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
    output_shape: List[int] = Field(
        ..., description="Predicted tensor output dimensions [channels, height, width]"
    )
    model_type: str = Field(..., description="Architecture used for inference")
    checkpoint: Optional[str] = Field(None, description="Checkpoint path or identifier")
    prediction_min: float = Field(..., description="Minimum reflectance value in prediction")
    prediction_max: float = Field(..., description="Maximum reflectance value in prediction")
    prediction_mean: float = Field(..., description="Mean reflectance value in prediction")


# ============================================================================
# Model Manager
# ============================================================================


class ModelManager:
    """Manages model lifecycle, bounded concurrency, and dynamic artifact loading."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device_str: str = "cpu",
        max_concurrent: Optional[int] = None,
        queue_timeout: Optional[float] = None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(device_str)
        self.model: Optional[torch.nn.Module] = None
        self.model_config: Dict[str, Any] = {}
        self.model_source_meta: Dict[str, str] = {}
        self.is_ready: bool = False
        self.load_error: Optional[str] = None

        # Bounded concurrency limiting (eliminates global lock bottleneck)
        mc = (
            max_concurrent
            if max_concurrent is not None
            else int(os.getenv("MAX_CONCURRENT_INFERENCES", "2"))
        )
        qt = (
            queue_timeout
            if queue_timeout is not None
            else float(os.getenv("INFERENCE_QUEUE_TIMEOUT", "10.0"))
        )
        self.max_concurrent = max(1, mc)
        self.queue_timeout = max(0.1, qt)
        self.semaphore = threading.Semaphore(self.max_concurrent)

    @property
    def lock(self) -> threading.Semaphore:
        """Backward compatibility property for components expecting a lock."""
        return self.semaphore

    def load(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        source_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Safely load model weights from checkpoint path, MLflow, or S3."""
        if checkpoint_path is not None:
            self.checkpoint_path = str(checkpoint_path)

        cfg = source_config or {}
        if self.checkpoint_path and "checkpoint_path" not in cfg:
            cfg["checkpoint_path"] = self.checkpoint_path
            if "source" not in cfg:
                cfg["source"] = "local"

        try:
            model, m_cfg, meta = load_model_dynamically(
                config=cfg,
                device_str=str(self.device),
            )
            self.model = model
            self.model_config = m_cfg
            self.model_source_meta = meta
            self.checkpoint_path = meta.get("checkpoint_path", self.checkpoint_path)
            self.is_ready = True
            self.load_error = None
            logger.info(
                "Successfully loaded model from %s on device %s",
                meta.get("identifier"),
                self.device,
            )
            return True
        except Exception as exc:
            self.is_ready = False
            self.model = None
            self.load_error = f"Failed to load model: {type(exc).__name__}: {exc}"
            logger.warning("ModelManager.load: %s", self.load_error)
            return False

    def set_model(self, model: torch.nn.Module, checkpoint_name: str = "in-memory") -> None:
        """Inject a pre-instantiated model instance (useful for unit tests without checkpoint files)."""
        model.eval()
        model.to(self.device)
        self.model = model
        self.model_config = getattr(
            model, "get_config", lambda: {"model_type": type(model).__name__}
        )()
        self.checkpoint_path = checkpoint_name
        self.model_source_meta = {"source": "in-memory", "identifier": checkpoint_name}
        self.is_ready = True
        self.load_error = None

    def unload(self) -> None:
        """Unload the active model and mark the service as unavailable."""
        self.model = None
        self.is_ready = False
        self.load_error = "Model unloaded or unavailable"

    async def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute deterministic inference bounded by semaphore and offloaded to worker thread."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Model is not initialized or ready for inference")

        return await asyncio.to_thread(self._sync_predict, input_tensor)

    def _sync_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if not self.semaphore.acquire(timeout=self.queue_timeout):
            raise ServiceOverloadedError(
                "Inference service is currently overloaded. Please retry later."
            )
        try:
            if self.model is None:
                raise RuntimeError("Model is not loaded")
            with torch.inference_mode():
                return predict_patch(self.model, input_tensor, device=self.device)
        finally:
            self.semaphore.release()

    async def predict_raster(
        self,
        input_source: Union[str, Path, rasterio.io.DatasetReaderBase],
        output_target: Union[str, Path, rasterio.io.MemoryFile],
        tile_size: int = 256,
        overlap: int = 32,
        batch_size: int = 4,
        scale_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute windowed tiled raster inference bounded by concurrency semaphore."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Model is not initialized or ready for inference")

        return await asyncio.to_thread(
            self._sync_predict_raster,
            input_source,
            output_target,
            tile_size,
            overlap,
            batch_size,
            scale_factor,
        )

    def _sync_predict_raster(
        self,
        input_source: Union[str, Path, rasterio.io.DatasetReaderBase],
        output_target: Union[str, Path, rasterio.io.MemoryFile],
        tile_size: int = 256,
        overlap: int = 32,
        batch_size: int = 4,
        scale_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.semaphore.acquire(timeout=self.queue_timeout):
            raise ServiceOverloadedError(
                "Inference service is currently overloaded. Please retry later."
            )
        try:
            if self.model is None:
                raise RuntimeError("Model is not loaded")
            return predict_tiled_raster(
                model=self.model,
                input_source=input_source,
                output_target=output_target,
                tile_size=tile_size,
                overlap=overlap,
                device=self.device,
                batch_size=batch_size,
                scale_factor=scale_factor,
            )
        finally:
            self.semaphore.release()


# ============================================================================
# Prometheus Metrics Manager
# ============================================================================


class MetricsManager:
    """Manages Prometheus metrics for API requests, latency, and model inference."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry(auto_describe=True)

        self.requests_total = Counter(
            "api_requests_total",
            "Total count of HTTP requests processed by method, endpoint, and status code",
            ["method", "endpoint", "status_code"],
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "api_request_duration_seconds",
            "Histogram of request processing duration in seconds",
            ["method", "endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        self.model_inference_duration_seconds = Histogram(
            "model_inference_duration_seconds",
            "Histogram of model forward pass execution duration in seconds",
            ["model_type"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.model_inference_requests_total = Counter(
            "model_inference_requests_total",
            "Total count of model inference invocations by model type and status",
            ["model_type", "status"],
            registry=self.registry,
        )
        self.model_ready_gauge = Gauge(
            "model_ready_status",
            "Gauge indicating whether model is currently loaded and ready for inference (1=ready, 0=not ready)",
            registry=self.registry,
        )


# ============================================================================
# FastAPI Application Factory
# ============================================================================


def create_app(
    model_manager: Optional[ModelManager] = None,
    metrics_manager: Optional[MetricsManager] = None,
) -> FastAPI:
    """Create and configure the FastAPI inference application with Prometheus metrics."""

    # Initialize manager and metrics if not explicitly injected
    manager = (
        model_manager
        if model_manager is not None
        else ModelManager(device_str=os.getenv("DEVICE", "cpu"))
    )
    metrics_mgr = metrics_manager if metrics_manager is not None else MetricsManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: attempt to load configured model if not already injected
        if not manager.is_ready:
            env_source = os.getenv("MODEL_SOURCE", "local").lower().strip()
            env_ckpt = os.getenv("MODEL_CHECKPOINT")
            if env_ckpt or env_source != "local":
                manager.load()
            else:
                default_p = Path(DEFAULT_CHECKPOINT)
                if default_p.is_file():
                    manager.load(str(default_p))
                else:
                    manager.is_ready = False
                    manager.load_error = f"Default checkpoint not found at {default_p} and MODEL_CHECKPOINT env var not set."
                    logger.info("Lifespan: %s", manager.load_error)
        metrics_mgr.model_ready_gauge.set(
            1.0 if manager.is_ready and manager.model is not None else 0.0
        )
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

    # Attach managers to app state for dependency access
    app.state.model_manager = manager
    app.state.metrics_manager = metrics_mgr

    # ------------------------------------------------------------------------
    # Prometheus HTTP Request Middleware
    # ------------------------------------------------------------------------
    @app.middleware("http")
    async def prometheus_metrics_middleware(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start_time
            metrics_mgr.requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=str(status_code),
            ).inc()
            metrics_mgr.request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)

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
            422: {
                "description": "Malformed input array, unsupported channels, or out-of-range reflectance"
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "Model is not loaded or unavailable"
            },
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
        t0 = time.perf_counter()
        try:
            output_tensor = await manager.predict(input_tensor)
            inf_duration = time.perf_counter() - t0
            m_type = manager.model_config.get("model_type", "UNet")
            metrics_mgr.model_inference_duration_seconds.labels(model_type=m_type).observe(
                inf_duration
            )
            metrics_mgr.model_inference_requests_total.labels(
                model_type=m_type, status="success"
            ).inc()
        except ServiceOverloadedError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
                headers={"Retry-After": "5"},
            )
        except HTTPException:
            raise
        except Exception as exc:
            m_type = manager.model_config.get("model_type", "UNet")
            metrics_mgr.model_inference_requests_total.labels(
                model_type=m_type, status="error"
            ).inc()
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

    # ------------------------------------------------------------------------
    # POST /predict/raster
    # ------------------------------------------------------------------------
    @app.post(
        "/predict/raster",
        summary="Full-Scene Raster Cloud Removal Inference",
        description=(
            "Executes memory-bounded, windowed tiled cloud-removal inference on an uploaded GeoTIFF or PNG image.\n\n"
            "Features:\n"
            "- Streaming upload to avoid loading large files entirely into RAM.\n"
            "- Windowed tile reading and overlap blending to eliminate boundary seams.\n"
            "- Returns either a reconstructed GeoTIFF file download or a structured JSON metadata response."
        ),
        responses={
            status.HTTP_200_OK: {
                "description": "Reconstructed GeoTIFF binary stream or JSON metadata summary"
            },
            status.HTTP_413_CONTENT_TOO_LARGE: {
                "description": "Upload size exceeds configured maximum limit"
            },
            422: {"description": "Invalid file content, unsupported channels, or malformed raster"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "Model is not loaded or inference service is overloaded"
            },
        },
    )
    async def predict_raster(
        file: UploadFile = File(..., description="Satellite image file (GeoTIFF or PNG)"),
        tile_size: int = Query(256, ge=16, le=1024, description="Spatial tile size in pixels"),
        overlap: int = Query(
            32, ge=0, le=256, description="Spatial overlap between adjacent tiles in pixels"
        ),
        response_type: str = Query(
            "geotiff",
            pattern="^(geotiff|json)$",
            description="Response format: 'geotiff' or 'json'",
        ),
        scale_factor: Optional[float] = Query(
            None, description="Optional scaling divisor to convert DN to [0, 1] reflectance"
        ),
    ) -> Response:
        if not manager.is_ready or manager.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded or unavailable for inference.",
            )

        if tile_size <= overlap:
            raise HTTPException(
                status_code=422,
                detail=f"tile_size ({tile_size}) must be strictly greater than overlap ({overlap}).",
            )

        # 1. Enforce max upload size via streaming chunks to temporary file
        max_upload_size = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(100 * 1024 * 1024)))
        total_bytes = 0

        suffix = Path(file.filename or "upload.tif").suffix or ".tif"
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_in_path = Path(tmp_in.name)

        try:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_upload_size:
                    tmp_in.close()
                    tmp_in_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Upload size ({total_bytes} bytes) exceeds maximum allowed limit of {max_upload_size} bytes.",
                    )
                tmp_in.write(chunk)
            tmp_in.close()

            if total_bytes == 0:
                tmp_in_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=422,
                    detail="Uploaded file is empty (0 bytes).",
                )

            # 2. Inspect raster structure using rasterio or PIL
            is_geotiff = False
            is_png = False
            in_channels = manager.model_config.get("in_channels", 3)

            try:
                with rasterio.open(tmp_in_path) as src_inspect:
                    src_count = src_inspect.count
                    src_h = src_inspect.height
                    src_w = src_inspect.width
                    is_geotiff = True
            except Exception:
                is_geotiff = False

            if not is_geotiff:
                try:
                    from PIL import Image

                    with Image.open(tmp_in_path) as img:
                        if img.format != "PNG":
                            raise ValueError(f"Unsupported format: {img.format}")
                        src_w, src_h = img.size
                        png_mode = img.mode
                        is_png = True
                except Exception:
                    tmp_in_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=422,
                        detail="Invalid or unsupported raster file: file is neither a valid GeoTIFF nor a supported PNG.",
                    )

            # 3. Validate channel count and dimensions
            if is_geotiff:
                if src_count != in_channels:
                    tmp_in_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Unsupported channel count: model expects {in_channels} channels, uploaded GeoTIFF has {src_count}.",
                    )
                if src_h < MIN_DIM or src_w < MIN_DIM:
                    tmp_in_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Raster dimensions ({src_h}x{src_w}) too small: minimum spatial size is {MIN_DIM}x{MIN_DIM}.",
                    )
            elif is_png:
                if png_mode not in ("RGB", "L"):
                    tmp_in_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Unsupported PNG mode '{png_mode}': expected 3-channel RGB image.",
                    )
                png_channels = 3 if png_mode == "RGB" else 1
                if png_channels != in_channels:
                    tmp_in_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Unsupported channel count: model expects {in_channels} channels, PNG has {png_channels}.",
                    )
                if src_h < MIN_DIM or src_w < MIN_DIM:
                    tmp_in_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=422,
                        detail=f"PNG dimensions ({src_h}x{src_w}) too small: minimum spatial size is {MIN_DIM}x{MIN_DIM}.",
                    )

            # 4. Execute windowed tiled inference
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            tmp_out_path = Path(tmp_out.name)
            tmp_out.close()

            start_time = time.perf_counter()

            if is_geotiff:
                try:
                    summary = await manager.predict_raster(
                        input_source=tmp_in_path,
                        output_target=tmp_out_path,
                        tile_size=tile_size,
                        overlap=overlap,
                        scale_factor=scale_factor,
                    )
                except ServiceOverloadedError as exc:
                    tmp_out_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=str(exc),
                        headers={"Retry-After": "5"},
                    )
            else:
                from PIL import Image

                with Image.open(tmp_in_path) as img:
                    arr = np.array(img.convert("RGB")).transpose(2, 0, 1).astype(np.float32) / 255.0
                pred_arr = await asyncio.to_thread(
                    predict_tiled_array,
                    model=manager.model,
                    cloudy=arr,
                    tile_size=tile_size,
                    overlap=overlap,
                    device=manager.device,
                )
                with rasterio.open(
                    tmp_out_path,
                    "w",
                    driver="GTiff",
                    height=src_h,
                    width=src_w,
                    count=in_channels,
                    dtype="float32",
                ) as dst:
                    dst.write(pred_arr)
                summary = {
                    "height": src_h,
                    "width": src_w,
                    "channels": in_channels,
                    "crs": None,
                    "transform": None,
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "total_tiles": len(generate_tile_coordinates(src_h, src_w, tile_size, overlap)),
                }

            duration = time.perf_counter() - start_time

            m_type = manager.model_config.get("model_type", "UNet")
            metrics_mgr.model_inference_duration_seconds.labels(model_type=m_type).observe(duration)
            metrics_mgr.model_inference_requests_total.labels(
                model_type=m_type, status="success"
            ).inc()

            # 5. Build response
            if response_type == "json":
                with rasterio.open(tmp_out_path) as out_src:
                    out_sample = out_src.read()
                    min_v = float(out_sample.min())
                    max_v = float(out_sample.max())
                    mean_v = float(out_sample.mean())

                tmp_out_path.unlink(missing_ok=True)

                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "status": "success",
                        "model_type": m_type,
                        "model_identifier": manager.model_source_meta.get(
                            "identifier", manager.checkpoint_path or "in-memory"
                        ),
                        "input_filename": file.filename,
                        "raster_shape": [in_channels, summary["height"], summary["width"]],
                        "crs": summary.get("crs"),
                        "transform": summary.get("transform"),
                        "tile_size": tile_size,
                        "overlap": overlap,
                        "total_tiles": summary.get("total_tiles"),
                        "processing_duration_seconds": round(duration, 4),
                        "prediction_min": min_v,
                        "prediction_max": max_v,
                        "prediction_mean": mean_v,
                    },
                )

            output_bytes = tmp_out_path.read_bytes()
            tmp_out_path.unlink(missing_ok=True)

            orig_stem = Path(file.filename or "scene").stem
            return Response(
                content=output_bytes,
                media_type="image/tiff",
                headers={
                    "Content-Disposition": f'attachment; filename="predicted_{orig_stem}.tif"',
                    "X-Model-Type": str(m_type),
                    "X-Model-Identifier": str(
                        manager.model_source_meta.get(
                            "identifier", manager.checkpoint_path or "in-memory"
                        )
                    ),
                    "X-Raster-Dimensions": f"{in_channels}x{summary['height']}x{summary['width']}",
                    "X-Raster-CRS": str(summary.get("crs")),
                    "X-Processing-Duration-Seconds": f"{duration:.4f}",
                },
            )

        finally:
            tmp_in_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------------
    # GET /metrics
    # ------------------------------------------------------------------------
    @app.get(
        "/metrics",
        summary="Prometheus Metrics",
        description="Exposes Prometheus metrics for API requests, latency, model inferences, and readiness.",
    )
    async def metrics() -> Response:
        metrics_mgr.model_ready_gauge.set(
            1.0 if manager.is_ready and manager.model is not None else 0.0
        )
        data = generate_latest(metrics_mgr.registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    # Attach interactive Web UI and endpoints
    from cloud_removal.ui import setup_ui

    setup_ui(app)

    return app


# ============================================================================
# Default ASGI Application Entrypoint
# ============================================================================

app = create_app()
