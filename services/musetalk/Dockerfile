FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.10 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        torch==2.2.0+cu118 \
        torchvision==0.17.0+cu118 \
        torchaudio==2.2.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

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
