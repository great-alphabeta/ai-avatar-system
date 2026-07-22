# AI Avatar System — Setup Guide

## Prerequisites

- **Docker** (20.10+) and **Docker Compose** (v2+)
- **NVIDIA Container Toolkit** (recommended for STT / TTS / MuseTalk)
- **Python** 3.10+ (only if running the thin API outside Docker)
- **Node.js** 18+ (only if running the frontend outside Docker)
- **Git**
- API keys: Anthropic and/or OpenAI (or Ollama for local LLM)

## Architecture (Docker-isolated ML)

| Container | Role | Python env |
|---|---|---|
| `backend` | Thin FastAPI API + WebSocket | Slim (no torch) |
| `stt` | Whisper STT | Own CUDA image |
| `tts` | Chatterbox TTS (+ edge-tts/gTTS fallback) | Own CUDA image |
| `musetalk` | MuseTalk lip-sync (+ FFmpeg simple fallback) | Own CUDA image |
| `postgres` / `redis` / `frontend` / `celery-worker` | Infra & UI | — |

Shared media lives on the `media_data` volume mounted at `/media` so services can pass filesystem paths.

## Quick Start

```bash
cp .env.example .env
# Fill SECRET_KEY, JWT_SECRET_KEY, ANTHROPIC_API_KEY (or OPENAI_API_KEY)

docker compose up -d --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Flower | http://localhost:5555 |

Health (includes remote ML reachability):

```bash
curl http://localhost:8000/health
docker compose ps
```

## MuseTalk models (~9 GB)

```bash
bash scripts/setup_musetalk.sh
# Writes into services/musetalk/models/MuseTalk
# Set AVATAR_ENGINE=musetalk in .env
docker compose restart musetalk
```

## Configuration

### LLM

```env
LLM_PROVIDER=anthropic          # anthropic | openai | ollama
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=...
```

Local Ollama (compose profile):

```bash
docker compose --profile ollama up -d
```

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
OPENAI_BASE_URL=http://ollama:11434/v1
```

### Avatar / STT / TTS

```env
AVATAR_ENGINE=musetalk          # or simple
WHISPER_MODEL=large-v3-turbo
TTS_PROVIDER=chatterbox
STT_URL=http://stt:8001
TTS_URL=http://tts:8002
MUSETALK_URL=http://musetalk:8003
```

## Production

```bash
cp .env.prod.example .env.prod
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

GPU is reserved for `stt`, `tts`, and `musetalk` only — the API stays CPU.

## Manual thin API (ML still in Docker)

```bash
docker compose up -d postgres redis stt tts musetalk
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --reload --port 8000
```

## Troubleshooting

**ML service unreachable / health degraded**

```bash
docker compose logs stt tts musetalk
docker exec avatar-stt curl -fsS http://localhost:8001/health
```

**GPU not detected**

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then `docker compose up -d` again.

**MuseTalk falls back to simple**

Run `bash scripts/setup_musetalk.sh` and confirm `services/musetalk/models/MuseTalk/scripts/inference.py` exists.

## License

MIT — see LICENSE.
