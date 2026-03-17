FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Split apt-get to avoid OOM
RUN apt-get update && apt-get install -y --no-install-recommends wget git curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends git-lfs ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender1 libglib2.0-0 libsndfile1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /opt/gui

# Clone SD WebUI Forge (no longer depends on unavailable Stability-AI repos)
RUN git clone --depth 1 https://github.com/lllyasviel/stable-diffusion-webui-forge.git .

# Clone required repositories (Forge only needs these)
RUN mkdir -p repositories && \
    git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git repositories/stable-diffusion-webui-assets && \
    git clone --depth 1 https://github.com/lllyasviel/huggingface_guess.git repositories/huggingface_guess && \
    git clone --depth 1 https://github.com/salesforce/BLIP.git repositories/BLIP

# Create model directories
RUN mkdir -p models/Stable-diffusion models/Lora models/VAE embeddings

# Install Forge's requirements - use their requirements_versions.txt
# First install typing_extensions to avoid import errors
RUN uv pip install --system "typing_extensions>=4.10"

# Install xformers for GPU acceleration (optional, may fail on some systems)
RUN uv pip install --system xformers==0.0.27 || true

# Install requirements from Forge's requirements file
RUN uv pip install --system -r requirements_versions.txt

# Download SD 1.5 model from HuggingFace using curl (more memory efficient)
RUN cd models/Stable-diffusion && \
    curl -L --progress-bar -o v1-5-pruned-emaonly.safetensors \
    "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" && \
    ls -lh v1-5-pruned-emaonly.safetensors

# Download ControlNet models one at a time
RUN mkdir -p models/ControlNet && cd models/ControlNet && \
    curl -L --progress-bar -o control_v11p_sd15_canny.pth \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth" && \
    ls -lh control_v11p_sd15_canny.pth

RUN cd models/ControlNet && \
    curl -L --progress-bar -o control_v11f1p_sd15_depth.pth \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" && \
    ls -lh control_v11f1p_sd15_depth.pth

RUN cd models/ControlNet && \
    curl -L --progress-bar -o control_v11p_sd15_openpose.pth \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" && \
    ls -lh control_v11p_sd15_openpose.pth
# Install CLIP
RUN uv pip install --system git+https://github.com/openai/CLIP.git

# Pre-install bitsandbytes and joblib (Forge installs these at runtime otherwise)
RUN uv pip install --system bitsandbytes==0.45.3 joblib

# Tool dependencies (install after Forge requirements to avoid conflicts)
RUN uv pip install --system pillow scikit-image pyyaml tqdm
RUN uv pip install --system retina-face gdown
RUN uv pip install --system ultralytics

# Copy and install the tool
COPY . /tool
WORKDIR /tool
RUN uv pip install --system -e .

WORKDIR /opt/gui
EXPOSE 7860

ENTRYPOINT ["python3", "webui.py", "--listen", "--api", "--skip-torch-cuda-test"]
