FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.10 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip

COPY requirements.txt .

# Install order matters for chatterbox-tts:
# 1) numpy first (native deps / build isolation)
# 2) CUDA torch==2.6.0 to match chatterbox (avoid CPU wheels + version fights)
# 3) runtime deps from requirements.txt (intentionally omits Gradio)
# 4) chatterbox-tts --no-deps so it does not pull Gradio/pkuseg/CPU torch
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir "numpy>=1.24.0,<1.26.0" \
    && pip install --no-cache-dir \
        torch==2.6.0+cu118 \
        torchaudio==2.6.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps chatterbox-tts==0.1.6

COPY app ./app

RUN mkdir -p /media /app/voice_profiles

ENV TMPDIR=/media
EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8002/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
