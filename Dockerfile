# ============================================================================
# Phase 4: Production Dockerfile for LISS-IV Cloud Removal Inference Service
# ============================================================================
# Base image: Official Python 3.11 slim (Debian bookworm)
FROM python:3.11-slim AS runtime

# Set execution environment flags and production serving defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    DEVICE=cpu \
    MODEL_SOURCE=local \
    MODEL_CHECKPOINT=checkpoints/rice2/best_rice2_compact_unet.pt \
    MAX_CONCURRENT_INFERENCES=2 \
    INFERENCE_QUEUE_TIMEOUT=10.0 \
    MAX_UPLOAD_SIZE_BYTES=104857600 \
    MODEL_CACHE_DIR=/app/.cache/models

# Create non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Install minimal system runtime dependencies (libexpat1 required for rasterio/GDAL bindings)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libexpat1 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CPU-only PyTorch first (avoids heavy CUDA binaries)
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install application dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy package definition and application source code
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# Create model cache directories and set permissions
RUN mkdir -p /app/.cache/models /app/checkpoints && \
    pip install --no-deps . && \
    chown -R appuser:appgroup /app

# Switch to unprivileged non-root user
USER appuser

# Expose inference port
EXPOSE 8000

# Container liveness health check using standard library urllib
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start production ASGI server
# Note on worker scaling:
# On GPU: 1 worker avoids multi-process GPU context switching, thrashing, and VRAM duplication.
# On CPU: 1 worker with MAX_CONCURRENT_INFERENCES=2..4 maximizes throughput via thread-pool offloading.
# Horizontal scaling should be achieved via Kubernetes pod replicas rather than multiple Uvicorn workers.
CMD ["uvicorn", "cloud_removal.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
