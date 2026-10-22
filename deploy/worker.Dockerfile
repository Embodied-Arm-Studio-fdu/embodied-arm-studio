FROM ubuntu:20.04 AS ffmpeg
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config nasm git curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch n12.1.14.0 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && make -C /tmp/nv-codec-headers install
RUN curl --fail --location https://ffmpeg.org/releases/ffmpeg-7.1.5.tar.xz -o /tmp/ffmpeg.tar.xz \
    && mkdir /tmp/ffmpeg && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
    && cd /tmp/ffmpeg && ./configure --prefix=/opt/ffmpeg --enable-nvenc --disable-doc --disable-debug \
    && make -j2 && make install

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip build-essential libvulkan1 vulkan-utils \
    libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libpython3.8 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir 'pip<25.1' 'setuptools<75' wheel \
    && python3 -m pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
RUN mkdir -p /usr/share/vulkan/icd.d && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.0"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json
COPY --from=ffmpeg /opt/ffmpeg /opt/ffmpeg
WORKDIR /app
COPY deploy/worker-requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY simulation ./simulation
COPY robotics ./robotics
COPY deploy ./deploy
ENV PYTHONPATH=/opt/isaacgym/python:/app:/app/simulation/src \
    VK_ICD_FILENAMES=/app/deploy/nvidia-egl-icd.json \
    VK_DRIVER_FILES=/app/deploy/nvidia-egl-icd.json
# SDK and robot assets are supplied read-only by the operator; never baked in.
CMD ["python3", "deploy/worker_entrypoint.py"]
