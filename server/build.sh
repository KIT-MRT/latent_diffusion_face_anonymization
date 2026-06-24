#!/bin/bash
# Build the ldfa-api server image (independent of the ldfa base image;
# SAM3 weights baked in) and smoke-test the import.
# Requires an HF token with access to facebook/sam3 at ~/.cache/huggingface/token.
set -e

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "HF token not found at $HF_TOKEN_FILE"
    echo "run: huggingface-cli login  (needs access to facebook/sam3)"
    exit 1
fi

DOCKER_BUILDKIT=1 docker build \
    --secret id=hf_token,src="$HF_TOKEN_FILE" \
    -t ldfa-api \
    -f server/Dockerfile \
    "$(dirname "$0")/.."
docker run --rm ldfa-api python -c 'import server.main; print("OK")'
