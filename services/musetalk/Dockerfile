# Official image ships CUDA torch already — avoids Docker Desktop segfault/OOM
# while pip unpacks multi-GB nvidia-* wheels (exit 139 / Bad CRC-32).
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Bundled conda pip is broken on these images; use uv instead.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_NO_CACHE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch is preinstalled; add matching vision/audio wheels without re-pulling CUDA deps.
RUN uv pip install --no-deps \
        "https://download.pytorch.org/whl/cu118/torchvision-0.17.0%2Bcu118-cp310-cp310-linux_x86_64.whl" \
        "https://download.pytorch.org/whl/cu118/torchaudio-2.2.0%2Bcu118-cp310-cp310-linux_x86_64.whl" \
    && uv pip install -r requirements.txt

COPY musetalk_worker.py .
COPY app ./app

RUN mkdir -p /media /app/models

ENV TMPDIR=/media
ENV MUSETALK_PATH=/app/models/MuseTalk
ENV PYTHONPATH=/app/models/MuseTalk
EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8003/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
