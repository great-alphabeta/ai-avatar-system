"""
Speech-to-text client — calls the isolated Whisper STT microservice over HTTP.

Public API matches the former in-process STTService so websocket/tests stay stable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self):
        self.provider = settings.STT_PROVIDER
        self.model_name = settings.WHISPER_MODEL
        self.base_url = settings.STT_URL.rstrip("/")
        self.timeout = httpx.Timeout(settings.STT_TIMEOUT_SECONDS, connect=10.0)
        # Kept for health-check compatibility (never locally loaded).
        self.model = None

    async def initialize(self) -> None:
        """Ping the remote STT service (optional warm-up)."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
                data = resp.json()
                logger.info("STT service reachable: %s", data)
        except Exception as e:
            logger.warning("STT service not reachable yet: %s", e)

    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    return resp.json()
                return {"status": "error", "code": resp.status_code}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    async def transcribe(self, audio_data: Union[bytes, str], language: str = "en") -> str:
        if self.provider != "whisper":
            raise ValueError(f"Unsupported STT provider: {self.provider}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if isinstance(audio_data, (str, Path)):
                path = str(audio_data)
                resp = await client.post(
                    f"{self.base_url}/transcribe/path",
                    json={"path": path, "language": language},
                )
            else:
                resp = await client.post(
                    f"{self.base_url}/transcribe",
                    files={"file": ("audio.webm", audio_data, "application/octet-stream")},
                    data={"language": language},
                )

            if resp.status_code >= 400:
                logger.error("STT error %s: %s", resp.status_code, resp.text)
                resp.raise_for_status()

            text = resp.json().get("text", "")
            logger.info("STT transcribed %d chars", len(text))
            return text


stt_service = STTService()
