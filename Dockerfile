# ============================================================================
# Phase 10: Production Dockerfile for LISS-IV Cloud Removal Inference Service
# ============================================================================
# Base image: Official Python 3.11 slim (Debian bookworm)
FROM python:3.11-slim AS runtime

# Set execution environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

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

# Install the package itself
RUN pip install --no-deps . && \
    chown -R appuser:appgroup /app

# Switch to unprivileged non-root user
USER appuser

# Expose inference port
EXPOSE 8000

# Container liveness health check using standard library urllib
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start production ASGI server
CMD ["uvicorn", "cloud_removal.api:app", "--host", "0.0.0.0", "--port", "8000"]
