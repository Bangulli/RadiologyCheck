# ============================================================
# Dockerfile: HuggingFace Inference (CLI)
# ============================================================
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# ── Build arguments ──────────────────────────────────────────
ARG PYTHON_VERSION=3.11
ARG HF_HOME=/app/hf_cache

# ── Environment variables ────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=${HF_HOME} \
    TOKENIZERS_PARALLELISM=false

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3-pip \
        python3.12-venv \
        git \
        wget \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN python -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=8

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────
COPY requirements.txt .
COPY src/utils/build_script.py .

COPY src/ /app/src/
COPY main.py /app/main.py
COPY build_config.json /app/config.json
## Copy baseprompt file
COPY v3_fewshot_baseprompt.json /app/baseprompt.json

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt 
RUN pip install flash_attn==2.8.3 --no-build-isolation

# ── Pre-download model weights at build time ─────────────────
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) && \
    mkdir -p ${HF_HOME} && \
    python build_script.py -tok ${HF_TOKEN}

# ── Entrypoint ───────────────────────────────────────────────
ENTRYPOINT ["python", "/app/main.py"]