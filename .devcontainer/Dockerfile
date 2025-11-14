# RECTOR - Rule-Enforced Constrained Trajectory Optimization and Reasoning
# CUDA 12.2 + cuDNN8 on Ubuntu 22.04 - optimized for RTX A3000 (Ampere, SM 8.6)
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Locales and basics
    locales tzdata \
    sudo curl ca-certificates gnupg lsb-release \
    # Build tools
    build-essential cmake git pkg-config wget unzip \
    # Python 3.10
    python3.10 python3.10-venv python3-pip python3-dev \
    # Protocol buffers for Waymo
    protobuf-compiler libprotobuf-dev \
    # OpenCV and graphics dependencies
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    # ffmpeg for video generation
    ffmpeg \
    # Additional utilities
    vim nano htop tmux screen \
    # Google Cloud SDK dependencies (for gsutil)
    apt-transport-https ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK for gsutil (Waymo dataset downloads)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Locale setup
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TZ=UTC

# Create non-root user with sudo access
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Python virtual environment at /opt/venv
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Upgrade pip, setuptools, wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install base requirements
COPY .devcontainer/requirements.base.txt /tmp/requirements.base.txt
RUN pip install --no-cache-dir -r /tmp/requirements.base.txt

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2 runtime)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121

# Install TensorFlow CPU (for Waymo dataset tools only - avoids CUDA conflicts)
RUN pip install --no-cache-dir tensorflow-cpu==2.11.0

# Install Waymo Open Dataset API
RUN pip install --no-cache-dir waymo-open-dataset-tf-2-11-0

# Install git-lfs for large file support
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Copy helper scripts
COPY .devcontainer/post-create.sh /usr/local/bin/post-create.sh
COPY .devcontainer/fix-permissions.sh /usr/local/bin/fix-permissions.sh
RUN chmod +x /usr/local/bin/*.sh

# Create workspace directory structure
RUN mkdir -p /workspace && chown -R ${USERNAME}:${USERNAME} /workspace

# Switch to non-root user
USER ${USERNAME}
WORKDIR /workspace

# Default command
CMD ["bash"]
