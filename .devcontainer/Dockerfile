# RECTOR - Rule-Enforced Constrained Trajectory Optimization and Reasoning
# CUDA 12.1 + cuDNN8 on Ubuntu 22.04 - matches PyTorch cu121 wheels
#
# LAYER CACHING STRATEGY:
# - Layers are ordered from least-frequently-changed to most-frequently-changed
# - Add new system packages to the "Additional system packages" section at the bottom
# - Add new Python packages to requirements.project.txt (not requirements.base.txt)
# - This ensures most layers remain cached when you add new dependencies
#
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

# =============================================================================
# LAYER 1: Core system packages (RARELY CHANGES)
# =============================================================================
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
    apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# LAYER 2: Locale setup (RARELY CHANGES)
# =============================================================================
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TZ=UTC

# =============================================================================
# LAYER 3: Create non-root user (RARELY CHANGES)
# =============================================================================
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# =============================================================================
# LAYER 4: Python virtual environment setup (RARELY CHANGES)
# =============================================================================
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# LAYER 5: Core ML frameworks - PyTorch (RARELY CHANGES)
# =============================================================================
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121

# =============================================================================
# LAYER 6: Core ML frameworks - TensorFlow & Waymo (RARELY CHANGES)
# =============================================================================
RUN pip install --no-cache-dir tensorflow-cpu==2.11.0
RUN pip install --no-cache-dir waymo-open-dataset-tf-2-11-0

# =============================================================================
# LAYER 6.5: JAX with CUDA 12 support (for Waymax closed-loop simulation)
# JAX must be installed before Waymax. Pin numpy to range compatible with
# both TensorFlow 2.11 (<1.25) and JAX 0.4.x (>=1.22).
# =============================================================================
RUN pip install --no-cache-dir "numpy>=1.24.0,<1.25.0" && \
    pip install --no-cache-dir "jax[cuda12]==0.4.30"

# =============================================================================
# LAYER 6.6: Waymax closed-loop simulator (JAX-based, from Waymo Research)
# Pin to a known-good commit for reproducibility.
# =============================================================================
RUN pip install --no-cache-dir \
    git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax

# =============================================================================
# LAYER 7: Base Python packages (OCCASIONALLY CHANGES)
# These are stable scientific/dev packages. Add new packages to requirements.project.txt
# =============================================================================
COPY .devcontainer/requirements.base.txt /tmp/requirements.base.txt
RUN pip install --no-cache-dir -r /tmp/requirements.base.txt

# =============================================================================
# LAYER 8: Project-specific Python packages (FREQUENTLY CHANGES)
# Add your new Python packages here - this layer rebuilds without affecting above layers
# =============================================================================
COPY .devcontainer/requirements.project.txt* /tmp/
RUN if [ -f /tmp/requirements.project.txt ]; then \
    pip install --no-cache-dir -r /tmp/requirements.project.txt; \
    fi

# =============================================================================
# LAYER 9: External tools from APT repositories (OCCASIONALLY CHANGES)
# =============================================================================
# Google Cloud SDK for gsutil (Waymo dataset downloads)
RUN echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    mkdir -p /etc/apt/keyrings && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /etc/apt/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Bazel build system (version 5.4.0 required for waymo-open-dataset)
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > /usr/share/keyrings/bazel-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    tee /etc/apt/sources.list.d/bazel.list && \
    apt-get update && apt-get install -y bazel-5.4.0 && \
    ln -sf /usr/bin/bazel-5.4.0 /usr/bin/bazel && \
    rm -rf /var/lib/apt/lists/*

# Git LFS for large file support
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# =============================================================================
# LAYER 10: Additional system packages (ADD NEW PACKAGES HERE)
# This is where you add new apt packages - layers above stay cached
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    parallel \
    # Cairo/Pango for high-quality figure rendering (Waymax viz + publication figures)
    libcairo2-dev libpango1.0-dev \
    # ImageMagick for figure compositing and format conversion
    imagemagick \
    # LaTeX (TeX Live for IEEE papers)
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-bibtex-extra \
    biber \
    latexmk \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# LAYER 11: Helper scripts (FREQUENTLY CHANGES)
# =============================================================================
COPY .devcontainer/post-create.sh /usr/local/bin/post-create.sh
COPY .devcontainer/fix-permissions.sh /usr/local/bin/fix-permissions.sh
RUN chmod +x /usr/local/bin/*.sh

# =============================================================================
# FINAL: Workspace setup and user switch
# =============================================================================
RUN mkdir -p /workspace && chown -R ${USERNAME}:${USERNAME} /workspace

USER ${USERNAME}
WORKDIR /workspace

CMD ["bash"]
