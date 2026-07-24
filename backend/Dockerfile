FROM python:3.10-slim-bookworm

# uv avoids pip segfaults (exit 139) when unpacking wheels under Docker Desktop memory pressure.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_NO_CACHE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install -r requirements.txt

COPY . .

RUN mkdir -p /media /tmp/avatars /tmp/videos /tmp/audio voice_profiles \
    && chmod +x entrypoint.sh

ENV TMPDIR=/media

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["./entrypoint.sh"]
