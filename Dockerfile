FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y wget git build-essential python3 python3-pip ffmpeg libsm6 libxext6
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /opt/gui
WORKDIR /opt/gui
RUN git checkout tags/v1.8.0
RUN python3 -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install xformers==v0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install -r requirements_versions.txt
RUN git clone https://github.com/Stability-AI/stablediffusion repositories/stable-diffusion-stability-ai
RUN git clone https://github.com/Stability-AI/generative-models.git repositories/generative-models
RUN git clone https://github.com/Mikubill/sd-webui-controlnet extensions/sd-webui-controlnet
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git repositories/stable-diffusion-webui-assets
RUN cd models/Stable-diffusion && wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors && wget https://civitai.com/api/download/models/114600 --content-disposition && wget https://civitai.com/api/download/models/245598 --content-disposition

RUN mkdir models/Lora && cd models/Lora && wget https://civitai.com/api/download/models/62833 --content-disposition

RUN cd extensions/sd-webui-controlnet/models && wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth && wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth && wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
RUN python3 -m pip install --pre git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary --extra-index-url https://download.pytorch.org/whl/nightly/cu118

COPY . /tool
WORKDIR /tool
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 setup.py install

WORKDIR /opt/gui
ENTRYPOINT python3 webui.py --listen --api --xformers
