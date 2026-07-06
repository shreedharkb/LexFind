# ============================================================
# LexFind API – Production Dockerfile
# Base: Python 3.11 slim

# ============================================================

FROM python:3.11-slim

# Prevent .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps required by PyMuPDF, torch, and native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (maximise Docker layer cache)
COPY requirements.txt .
# Install CPU-only PyTorch FIRST — the CUDA variant pulls 2.5 GB of NVIDIA
# libs that are useless on a CPU-only VM.  sentence-transformers runs fine on CPU.
RUN pip install --upgrade pip && \
    pip install torch==2.6.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
        --no-cache-dir && \
    grep -v "^torch" requirements.txt > /tmp/requirements_no_torch.txt && \
    pip install -r /tmp/requirements_no_torch.txt --no-cache-dir

# Copy the rest of the application
COPY . .

# Temp directory for confidential document uploads
RUN mkdir -p /tmp/confidential_uploads

EXPOSE 8000

# Liveness check against the health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch with uvicorn in factory mode (2 workers for 2-vCPU VMs)
CMD ["uvicorn", "app.main:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "5"]
