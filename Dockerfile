FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Create a folder with enough space for pip temporary files
RUN mkdir -p /bigtmp
ENV TMPDIR=/bigtmp

# Install system packages and required libraries

RUN apt-get update && apt-get install -y wget git curl git-lfs
RUN apt-get install -y build-essential python3 python3-pip ffmpeg
RUN apt-get install -y libsm6 libxext6 libxrender1 libglib2.0-0 libsndfile1 libgl1 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
# Debug: show disk usage
RUN df -h && du -sh /tmp /var/tmp /bigtmp

# Clone Stable Diffusion WebUI
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /opt/gui
WORKDIR /opt/gui
RUN git checkout tags/v1.8.0

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch and related packages
RUN python3 -m pip install -vvv --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install xformers and versioned requirements
RUN python3 -m pip install --no-cache-dir xformers==v0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install --no-cache-dir -r requirements_versions.txt

# Clone extra repositories and models
RUN git clone https://github.com/Stability-AI/stablediffusion repositories/stable-diffusion-stability-ai
RUN git clone https://github.com/Stability-AI/generative-models.git repositories/generative-models
RUN git clone https://github.com/Mikubill/sd-webui-controlnet extensions/sd-webui-controlnet
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git repositories/stable-diffusion-webui-assets

# Download models (Stable Diffusion + LoRA + ControlNet)
RUN mkdir -p models/Stable-diffusion && cd models/Stable-diffusion && \
    wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors && \
    wget https://civitai.com/api/download/models/114600 --content-disposition && \
    wget https://civitai.com/api/download/models/245598 --content-disposition

RUN mkdir -p models/Lora && cd models/Lora && \
    wget https://civitai.com/api/download/models/62833 --content-disposition

RUN mkdir -p extensions/sd-webui-controlnet/models && cd extensions/sd-webui-controlnet/models && \
    wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth && \
    wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth && \
    wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth

# Install k-diffusion pre-release
RUN python3 -m pip install --pre git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu118

# Copy your tool and install
COPY . /tool
WORKDIR /tool
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 setup.py install

# Final workdir and entrypoint
WORKDIR /opt/gui
ENTRYPOINT ["python3", "webui.py", "--listen", "--api", "--xformers"]
