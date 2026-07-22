"""Chatterbox TTS microservice — isolated Python/CUDA environment."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("tts")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "chatterbox")
_LEGACY = {"coqui": "chatterbox", "xtts": "chatterbox", "xtts_v2": "chatterbox"}
if TTS_PROVIDER in _LEGACY:
    TTS_PROVIDER = _LEGACY[TTS_PROVIDER]

_EDGE_VOICES = {
    "ar": "ar-SA-ZariyahNeural",
    "da": "da-DK-ChristelNeural",
    "de": "de-DE-KatjaNeural",
    "el": "el-GR-AthinaNeural",
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "fi": "fi-FI-NooraNeural",
    "fr": "fr-FR-DeniseNeural",
    "he": "he-IL-HilaNeural",
    "hi": "hi-IN-SwaraNeural",
    "it": "it-IT-ElsaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ms": "ms-MY-YasminNeural",
    "nl": "nl-NL-ColetteNeural",
    "no": "nb-NO-PernilleNeural",
    "pl": "pl-PL-ZofiaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "sv": "sv-SE-SofieNeural",
    "sw": "sw-KE-ZuriNeural",
    "tr": "tr-TR-EmelNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}

app = FastAPI(title="AvatarAI TTS", version="1.0.0")

_model = None
_load_lock: Optional[asyncio.Lock] = None


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


async def _ensure_model():
    global _model, _load_lock
    if TTS_PROVIDER != "chatterbox":
        raise HTTPException(500, f"Unsupported TTS provider: {TTS_PROVIDER}")
    if _model is not None:
        return _model
    if _load_lock is None:
        _load_lock = asyncio.Lock()
    async with _load_lock:
        if _model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            import torch

            device = "cuda" if _cuda_available() else "cpu"
            logger.info("Loading Chatterbox multilingual TTS on %s…", device)
            _model = await asyncio.to_thread(
                ChatterboxMultilingualTTS.from_pretrained, device=device
            )
            logger.info("Chatterbox loaded (sr=%s)", _model.sr)
    return _model


class SynthesizeRequest(BaseModel):
    text: str
    output_path: str = Field(..., description="Destination WAV path (shared /media)")
    speaker_wav: Optional[str] = None
    language: str = "en"


class SynthResponse(BaseModel):
    output_path: str
    engine: str
    fallback: bool
    voice_cloned: bool


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "tts",
        "provider": TTS_PROVIDER,
        "loaded": _model is not None,
        "cuda": _cuda_available(),
    }


async def _edge_fallback(text: str, output_path: str, language: str) -> None:
    import edge_tts
    from pydub import AudioSegment

    voice = _EDGE_VOICES.get(language, _EDGE_VOICES["en"])
    mp3_path = output_path.replace(".wav", "_edge.mp3")
    await edge_tts.Communicate(text, voice).save(mp3_path)
    await asyncio.to_thread(
        lambda: AudioSegment.from_mp3(mp3_path).export(output_path, format="wav")
    )
    Path(mp3_path).unlink(missing_ok=True)


async def _gtts_fallback(text: str, output_path: str, language: str) -> None:
    from gtts import gTTS
    from pydub import AudioSegment

    mp3_path = output_path.replace(".wav", "_gtts.mp3")
    await asyncio.to_thread(lambda: gTTS(text=text, lang=language, slow=False).save(mp3_path))
    await asyncio.to_thread(
        lambda: AudioSegment.from_mp3(mp3_path).export(output_path, format="wav")
    )
    Path(mp3_path).unlink(missing_ok=True)


@app.post("/synthesize", response_model=SynthResponse)
async def synthesize(body: SynthesizeRequest):
    if not body.text.strip():
        raise HTTPException(400, "text is required")

    Path(body.output_path).parent.mkdir(parents=True, exist_ok=True)
    speaker_wav = body.speaker_wav
    if speaker_wav and not Path(speaker_wav).exists():
        logger.warning("Speaker WAV not found: %r — using default voice", speaker_wav)
        speaker_wav = None

    try:
        model = await _ensure_model()
        import torchaudio

        kwargs = {"language_id": body.language}
        if speaker_wav:
            kwargs["audio_prompt_path"] = speaker_wav

        wav = await asyncio.to_thread(model.generate, body.text, **kwargs)
        await asyncio.to_thread(torchaudio.save, body.output_path, wav, model.sr)

        return SynthResponse(
            output_path=body.output_path,
            engine="chatterbox",
            fallback=False,
            voice_cloned=bool(speaker_wav),
        )
    except Exception as e:
        logger.warning("Chatterbox failed (%s), falling back", e)
        try:
            await _edge_fallback(body.text, body.output_path, body.language)
            engine = "edge-tts"
        except Exception as edge_err:
            logger.warning("Edge TTS failed (%s), falling back to gTTS", edge_err)
            await _gtts_fallback(body.text, body.output_path, body.language)
            engine = "gtts"

        return SynthResponse(
            output_path=body.output_path,
            engine=engine,
            fallback=True,
            voice_cloned=False,
        )
