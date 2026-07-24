"""
Avatar animation client — calls the isolated MuseTalk microservice over HTTP.

Falls back to a local FFmpeg still-image+audio mux if the remote service is
unreachable (keeps conversations working without lip-sync).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class AvatarAnimator:
    """
    Avatar Animation Service (HTTP client).
    Supported engines (set AVATAR_ENGINE in .env — forwarded to musetalk service):
      - musetalk : MuseTalk V1.5 lip-sync via remote worker
      - simple   : ffmpeg static image + audio
    """

    def __init__(self):
        self.engine = settings.AVATAR_ENGINE
        self.resolution = settings.AVATAR_RESOLUTION
        self.fps = settings.AVATAR_FPS
        self.base_url = settings.MUSETALK_URL.rstrip("/")
        self.timeout = httpx.Timeout(settings.MUSETALK_TIMEOUT_SECONDS, connect=10.0)
        self._initialised = False

    async def initialize(self):
        if self._initialised:
            return
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
                data = resp.json()
                logger.info("MuseTalk service reachable: %s", data)
                if data.get("engine"):
                    self.engine = data["engine"]
        except Exception as e:
            logger.warning(
                "MuseTalk service not reachable (%s) — local simple fallback will be used",
                e,
            )
        self._initialised = True

    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    return resp.json()
                return {"status": "error", "code": resp.status_code}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    async def animate(
        self,
        avatar_image_path: str,
        audio_path: str,
        output_path: str,
        cache_key: Optional[str] = None,
    ) -> str:
        if not self._initialised:
            await self.initialize()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Animating via %s image=%s audio=%s",
            self.base_url,
            avatar_image_path,
            audio_path,
        )

        payload = {
            "image": str(Path(avatar_image_path).resolve()),
            "audio": str(Path(audio_path).resolve()),
            "output": str(Path(output_path).resolve()),
            "engine": self.engine,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/animate", json=payload)
                if resp.status_code >= 400:
                    logger.error("Animate error %s: %s", resp.status_code, resp.text)
                    resp.raise_for_status()
                data = resp.json()
                return data.get("output_path", output_path)
        except Exception as e:
            logger.error("Remote animation failed: %s. Falling back to local simple.", e)
            return await self._animate_simple(avatar_image_path, audio_path, output_path)

    async def _animate_simple(
        self,
        avatar_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Local FFmpeg fallback — static image + audio, no lip-sync."""
        logger.info("Using local simple animation (static image + audio)")

        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(avatar_path),
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            "-vf",
            (
                f"fps={self.fps},"
                f"scale={self.resolution}:{self.resolution}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={self.resolution}:{self.resolution}:(ow-iw)/2:(oh-ih)/2"
            ),
            output_path,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")
            logger.error("FFmpeg error:\n%s", err)
            raise RuntimeError("Simple animation (ffmpeg) failed")

        return output_path

    def generate_cache_key(self, text: str, avatar_id: str) -> str:
        return hashlib.md5(f"{avatar_id}:{text}".encode()).hexdigest()


avatar_animator = AvatarAnimator()
