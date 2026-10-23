# --------------------------------------------------------------------------
# Stage 1 — build: install Python dependencies into an isolated prefix
# --------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed at install time only (not at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch, installed first from PyTorch's CPU index. From PyPI, recent
# torch on Linux (x86_64 and aarch64 alike) is the CUDA build plus several GB of
# nvidia-* wheels this image never uses, which is what fills a 20 GB EBS volume
# or gets the build killed on a 2 GiB instance. PYTHONPATH makes the prefix
# visible to the second install, so pip sees torch as satisfied and does not
# fetch the CUDA build from PyPI on top of it.
ENV PYTHONPATH=/install/lib/python3.12/site-packages
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# --------------------------------------------------------------------------
# Stage 2 — runtime: lean image with just what the app needs
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Suppress Python bytecode writes and buffer stdout/stderr for CloudWatch
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/app/.cache/torch_extensions \
    PORT=8000

# The ColBERT reranker (colbert-ai, via RAGatouille) JIT-compiles a small C++
# MaxSim kernel the first time a checkpoint loads, so the runtime image needs a
# C++ compiler — without one the first reranked query fails. It is compiled once
# below, at build time, so no request pays for it.
# git is there because colbert-ai imports GitPython at package import time
# (colbert/infra/config/base_config.py -> utility.utils.save_metadata), and
# GitPython raises ImportError when no git executable is on PATH. CI runners
# have git, which is why only the image build saw it; installing it, rather
# than silencing GitPython, also covers the code paths that actually call git.
RUN apt-get update && apt-get install -y --no-install-recommends g++ git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as uid 1000, and container hosts
# generally drop root. The user is created before anything is copied so each
# layer can be owned as it lands: a recursive chown afterwards duplicates every
# file it touches into a new layer, which on an image carrying torch and a set
# of model weights is an expensive way to change one metadata field.
# /app is chowned as a single directory, not recursively: the app writes
# index/, logs/ and the model cache into it at build and at runtime.
RUN useradd --create-home --uid 1000 app && \
    mkdir -p /app && \
    chown app:app /app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Every package `api.app` imports, which is checkable rather than remembered:
#   python -c "import api.app" and compare against this list.
# The five after `services` were missing until 2026-08 — the image had not been
# rebuilt since the workspace programme added them, so it would have started
# with ModuleNotFoundError on the first import.
COPY --chown=app:app config/       config/
COPY --chown=app:app chunking/     chunking/
COPY --chown=app:app embeddings/   embeddings/
COPY --chown=app:app generation/   generation/
COPY --chown=app:app retrieval/    retrieval/
COPY --chown=app:app services/     services/
COPY --chown=app:app corpora/      corpora/
COPY --chown=app:app documents/    documents/
COPY --chown=app:app ingestion/    ingestion/
COPY --chown=app:app jobs/         jobs/
COPY --chown=app:app evaluation/   evaluation/
COPY --chown=app:app api/          api/
COPY --chown=app:app aws/          aws/

# Needed only to build the benchmark index below, but they are small and
# keeping them makes the corpus rebuildable inside a running container.
COPY --chown=app:app data/         data/
COPY --chown=app:app scripts/      scripts/

# Bake the benchmark corpus and the model weights into the image.
#
# Both are otherwise downloaded or computed on first request, which on a
# scale-to-zero host means the first visitor waits through a 130 MB model
# download and a full re-index. Doing it here costs image size and buys a cold
# start that only has to load from local disk.
#
# This is also what makes the Evaluation Lab work on a host with an ephemeral
# filesystem: the labelled corpus is part of the image, not part of the volume.
USER app
RUN mkdir -p index logs && \
    python scripts/build_embeddings.py && \
    python scripts/build_index.py && \
    python -c "from colbert.modeling.colbert import ColBERT; ColBERT.try_load_torch_extensions(use_gpu=False)"

# Load the ColBERT checkpoint once, the way the API does. This bakes its
# ~440 MB of weights into the image, so the first reranked query does not
# download them, and it shows in the build log whether this image's
# torch/transformers can load ColBERT at all. Not fatal: if it cannot, the API
# falls back to the cross-encoder (retrieval.reranker.shared_reranker).
RUN python -c "from retrieval.reranker import ColBERTReranker; ColBERTReranker()" \
    && echo "ColBERT checkpoint loads" \
    || echo "WARNING: ColBERT failed to load in this image; the API will fall back to the cross-encoder"

EXPOSE 8000

# Shell form so $PORT is expanded: Spaces and most PaaS hosts inject their own.
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT}
