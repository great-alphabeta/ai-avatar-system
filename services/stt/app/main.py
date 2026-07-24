"""Whisper STT microservice — isolated Python/CUDA environment."""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Optional, Union

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger("stt")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

app = FastAPI(title="AvatarAI STT", version="1.0.0")

_model = None
_load_lock: Optional[asyncio.Lock] = None


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _build_model():
    from faster_whisper import WhisperModel

    device = "cuda" if _cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("Loading Whisper %r on %s (%s)…", WHISPER_MODEL, device, compute_type)
    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
    logger.info("Whisper model loaded")
    return model


async def _ensure_model():
    global _model, _load_lock
    if _model is not None:
        return _model
    if _load_lock is None:
        _load_lock = asyncio.Lock()
    async with _load_lock:
        if _model is None:
            _model = await asyncio.to_thread(_build_model)
    return _model


class PathTranscribeRequest(BaseModel):
    path: str = Field(..., description="Absolute path under shared /media volume")
    language: str = "en"


class TranscribeResponse(BaseModel):
    text: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "stt",
        "model": WHISPER_MODEL,
        "loaded": _model is not None,
        "cuda": _cuda_available(),
    }


def _decode_with_soundfile(audio_data: Union[bytes, str]) -> np.ndarray:
    if isinstance(audio_data, bytes):
        audio, sample_rate = sf.read(io.BytesIO(audio_data))
    else:
        audio, sample_rate = sf.read(audio_data)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    return audio.astype(np.float32)


def _transcribe_sync(model, audio_data: Union[bytes, str], language: str) -> str:
    source = io.BytesIO(audio_data) if isinstance(audio_data, bytes) else audio_data
    try:
        segments, info = model.transcribe(
            source,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = " ".join(seg.text for seg in segments).strip()
    except Exception as decode_err:
        logger.warning("PyAV decode failed (%s); retrying via soundfile", decode_err)
        audio = _decode_with_soundfile(audio_data)
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = " ".join(seg.text for seg in segments).strip()

    logger.info("Transcribed %d chars (lang=%s)", len(text), info.language)
    return text


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_upload(
    file: Optional[UploadFile] = File(None),
    language: str = Form("en"),
    path: Optional[str] = Form(None),
):
    """Accept multipart file upload and/or a shared-volume path."""
    model = await _ensure_model()

    if file is not None:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty audio upload")
        text = await asyncio.to_thread(_transcribe_sync, model, data, language)
        return TranscribeResponse(text=text)

    if path:
        if not os.path.isfile(path):
            raise HTTPException(404, f"Audio path not found: {path}")
        text = await asyncio.to_thread(_transcribe_sync, model, path, language)
        return TranscribeResponse(text=text)

    raise HTTPException(400, "Provide either file or path")


@app.post("/transcribe/path", response_model=TranscribeResponse)
async def transcribe_path(body: PathTranscribeRequest):
    model = await _ensure_model()
    if not os.path.isfile(body.path):
        raise HTTPException(404, f"Audio path not found: {body.path}")
    text = await asyncio.to_thread(_transcribe_sync, model, body.path, body.language)
    return TranscribeResponse(text=text)
