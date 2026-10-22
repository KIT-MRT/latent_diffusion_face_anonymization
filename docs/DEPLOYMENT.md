# Deployment

How to run the LDFA anonymization server with N Stable-Diffusion backends and the Gradio demo client.

## Architecture

```
client/app.py  ──HTTP──▶  server (FastAPI, port 8000)  ──HTTP──▶  sd-api-gpuN (Forge, ports 7860..)
     laptop                     1 container                          1 container per GPU
```

- `Dockerfile` builds the SD backend image (tag `ldfa`). Forge + SD 1.5 + ControlNet.
- `server/Dockerfile` builds the FastAPI/SAM3 image (tag `ldfa-api`). Independent from `ldfa` — uses its own torch cu121 base to avoid Forge's torch pin.
- `deploy/setup_server_compose.py` generates `docker-compose.generated.yml` with one `sd-api-gpuN` service per GPU plus the `anonymization-api` service.

## Build

The two images are independent — build in any order.

```bash
# SD backend image (large — downloads SD 1.5 + ControlNet weights)
docker build -t ldfa .

# Server image (needs an HF token with access to facebook/sam3 for the
# SAM3 weights that are baked into the image)
bash server/build.sh
```

`server/build.sh` reads `~/.cache/huggingface/token` (override with `HF_TOKEN_FILE=...`), runs `docker build` with BuildKit + the `hf_token` secret, and ends with a `docker run` import smoke-test.

## Run

Both images must be built before running the stack. The generator pulls the `ldfa-api` image by tag (it can't rebuild it under compose — the build needs a BuildKit secret).

```bash
# Start N GPUs worth of backends + the API server
python deploy/setup_server_compose.py --action start --gpu-count 4

# Stop
python deploy/setup_server_compose.py --action stop

# Health check
python deploy/health_check.py --server http://localhost:8000
```

The generated compose places `anonymization-api` on GPU 0 alongside `sd-api-gpu0`; SAM3 + one SD instance fit comfortably on a 48GB card. Config lives in `server/config/server_config.yaml`. Override GPU count from the CLI with `--gpu-count`.

## Client

```bash
python client/app.py --server-url http://<server-host>:8000
```

Open the Gradio URL it prints, grant webcam permission, pick targets + method.

## Content filtering

Disabled by default. `server/services/content_filter.py` is a no-op passthrough — if you need NSFW/gore blocking for a public-facing deploy, wire up AWS Rekognition / Azure Content Moderator / a local NSFW model inside `ContentFilterService.check_image`.

## Troubleshooting

- **SAM3 weights 401 / gated repo** when building `ldfa-api` — `HF_TOKEN_FILE` must point to a token with access to `facebook/sam3` (accept the license on the HF repo page first).
- **SD API connection refused** on port 7860+ — Forge takes 30–60s to load weights on first start. Check `docker logs sd-api-gpu0`.
- **OOM during SD image build** — `Dockerfile` splits model downloads across RUN layers. If your builder still OOMs, build with `--memory=8g` or build on a bigger host.
- **GPU not visible in container** — verify `nvidia-smi` works on the host and `nvidia-container-toolkit` is installed.
