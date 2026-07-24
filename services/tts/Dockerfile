# Official image ships CUDA torch already — avoids Docker Desktop segfault/OOM
# while pip unpacks multi-GB nvidia-* wheels (exit 139 / Bad CRC-32).
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# Bundled conda pip is broken on this image (UnboundLocalError / Link TypeError).
# Use uv for all package installs instead.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_NO_CACHE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install order matters for chatterbox-tts:
# 1) numpy first (native deps / build isolation)
# 2) torchaudio via direct wheel (torch already present)
# 3) runtime deps from requirements.txt (intentionally omits Gradio)
# 4) chatterbox-tts --no-deps so it does not pull Gradio/pkuseg/CPU torch
RUN uv pip install "numpy>=1.24.0,<1.26.0" \
    && uv pip install --no-deps \
        "https://download.pytorch.org/whl/cu118/torchaudio-2.6.0%2Bcu118-cp311-cp311-linux_x86_64.whl" \
    && uv pip install -r requirements.txt \
    && uv pip install --no-deps chatterbox-tts==0.1.6

COPY app ./app

RUN mkdir -p /media /app/voice_profiles

ENV TMPDIR=/media
EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8002/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
