# For GPU support with NVIDIA, you can switch to an NVIDIA CUDA base image.
# Default remains slim Python for portability; to use GPU, run container with --gpus or --runtime=nvidia.
# Note: Using --runtime=nvidia requires NVIDIA Container Toolkit to be installed on the host.
# SatoshiRig - Bitcoin solo mining client
FROM python:3.11-slim

# Install system dependencies for GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config

ENV PYTHONPATH=/app/src

# Optional: set WALLET_ADDRESS at runtime or pass via CLI
ENV WALLET_ADDRESS=""
ENV CONFIG_FILE="/app/config/config.toml"
ENV COMPUTE_BACKEND="cpu"
ENV GPU_DEVICE="0"
ENV WEB_PORT="5000"

# Docker Labels for WebUI Navigation (Docker Desktop / Portainer)
LABEL org.opencontainers.image.title="SatoshiRig"
LABEL org.opencontainers.image.description="Bitcoin Solo Mining Client WebUI"
LABEL org.opencontainers.image.url="http://localhost:5000"
LABEL io.portainer.container.http.port="5000"
LABEL io.portainer.container.http.url="http://localhost:5000"
LABEL io.portainer.accesscontrol.public="true"
LABEL com.docker.compose.service.webui="http://localhost:5000"

CMD ["python", "-m", "SatoshiRig"]


