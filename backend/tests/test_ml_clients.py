"""Unit tests for the HTTP STT / MuseTalk clients."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.animator import AvatarAnimator
from app.services.stt import STTService

pytestmark = pytest.mark.asyncio


async def test_stt_transcribe_path(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")

    service = STTService()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "hello there"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post = AsyncMock(return_value=mock_resp)

    monkeypatch.setattr("app.services.stt.httpx.AsyncClient", lambda **kwargs: mock_client)

    text = await service.transcribe(str(audio), language="en")
    assert text == "hello there"
    args, kwargs = mock_client.post.await_args
    assert args[0].endswith("/transcribe/path")


async def test_animator_calls_remote_animate(monkeypatch, tmp_path):
    img = tmp_path / "face.jpg"
    wav = tmp_path / "a.wav"
    out = tmp_path / "out.mp4"
    img.write_bytes(b"x")
    wav.write_bytes(b"y")

    animator = AvatarAnimator()
    animator._initialised = True

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"output_path": str(out), "engine": "simple"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post = AsyncMock(return_value=mock_resp)

    monkeypatch.setattr("app.services.animator.httpx.AsyncClient", lambda **kwargs: mock_client)

    result = await animator.animate(str(img), str(wav), str(out))
    assert result == str(out)
    mock_client.post.assert_awaited()
