"""
Text-to-Speech client — calls the isolated Chatterbox TTS microservice over HTTP.

Fallback (edge-tts → gTTS) is handled inside the TTS container. This client
preserves SynthResult so websocket/tests stay stable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SynthResult:
    output_path: str
    engine: str  # "chatterbox" | "edge-tts" | "gtts"
    fallback: bool
    voice_cloned: bool


_LEGACY_PROVIDER_ALIASES = {"coqui": "chatterbox", "xtts": "chatterbox", "xtts_v2": "chatterbox"}


class TTSService:
    def __init__(self):
        provider = settings.TTS_PROVIDER
        if provider in _LEGACY_PROVIDER_ALIASES:
            logger.warning(
                "TTS_PROVIDER=%r is deprecated — using 'chatterbox'. Update your .env.",
                provider,
            )
            provider = _LEGACY_PROVIDER_ALIASES[provider]
        self.provider = provider
        self.base_url = settings.TTS_URL.rstrip("/")
        self.timeout = httpx.Timeout(settings.TTS_TIMEOUT_SECONDS, connect=10.0)
        self.model = None

    async def initialize(self):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
                logger.info("TTS service reachable: %s", resp.json())
        except Exception as e:
            logger.warning("TTS service not reachable yet: %s", e)

    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    return resp.json()
                return {"status": "error", "code": resp.status_code}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    async def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> SynthResult:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "text": text,
            "output_path": output_path,
            "language": language,
            "speaker_wav": speaker_wav,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/synthesize", json=payload)
            if resp.status_code >= 400:
                logger.error("TTS error %s: %s", resp.status_code, resp.text)
                resp.raise_for_status()
            data = resp.json()

        return SynthResult(
            output_path=data["output_path"],
            engine=data.get("engine", "unknown"),
            fallback=bool(data.get("fallback", False)),
            voice_cloned=bool(data.get("voice_cloned", False)),
        )

    async def synthesize_bytes(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> bytes:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            await self.synthesize(text, tmp_path, speaker_wav, language)
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


__all__ = ["TTSService", "SynthResult", "tts_service"]

tts_service = TTSService()
