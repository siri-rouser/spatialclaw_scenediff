# SpatialClaw base dev image.
#
# Provides the system toolchain that spatial_agent/scripts/setup.sh expects
# (conda, git, compilers, ffmpeg). The repo itself is NOT copied in — mount it
# at runtime and run setup.sh inside the container, then `docker commit`.
#
# Build:
#   docker build -t spatialclaw:base docker/
#
# Run:
#   docker run -it --gpus all --ipc=host --name spatialclaw \
#     -v $HOME/SpatialClaw:/workspace/SpatialClaw \
#     spatialclaw:base

# Ubuntu 22.04 (glibc 2.35) matches the manylinux_2_35 wheel tag setup.sh uses
# for deep_gemm. CUDA 12.8 matches the host driver (570.x).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG MINICONDA_VERSION=latest

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        htop \
        less \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        tmux \
        unzip \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Miniconda in /opt/conda.
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

ENV CUDA_HOME=/usr/local/cuda

# The mounted repo is owned by a host UID, not root; let git operate on it.
RUN git config --global --add safe.directory '*'

WORKDIR /workspace

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda init bash \
    && conda config --set auto_activate_base false

CMD ["/bin/bash"]
