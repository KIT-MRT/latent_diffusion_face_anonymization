# syntax=docker/dockerfile:1.4
# Independent server image — does NOT inherit from ldfa:latest.
# The ldfa base image pins torch for Forge, which conflicts with the torch
# version transformers>=4.57 (for SAM3) requires. We use the upstream PyTorch
# cu121 image so torch matches the host NVIDIA driver (CUDA 12.1 capability).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt /app/server_requirements.txt
RUN pip install --no-cache-dir -r /app/server_requirements.txt

# Pre-download gated SAM3 weights into the image layer.
# Requires a BuildKit secret named hf_token containing an HF token with
# read access to facebook/sam3. Build with:
#   DOCKER_BUILDKIT=1 docker build \
#       --secret id=hf_token,src=$HOME/.cache/huggingface/token \
#       -t ldfa-api -f server/Dockerfile .
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python -c "\
from transformers import Sam3Model, Sam3Processor; \
Sam3Model.from_pretrained('facebook/sam3'); \
Sam3Processor.from_pretrained('facebook/sam3')"

# Weights are baked in — skip network revalidation on from_pretrained().
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY src/ /app/src/
COPY setup.py setup.cfg /app/
RUN pip install -e . --no-cache-dir

COPY server/ /app/server/
RUN mkdir -p /var/log/anonymization/flagged

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Single worker — each worker loads its own SAM3 copy, so >1 is a fast path to OOM.
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
