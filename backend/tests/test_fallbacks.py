"""
Tests for the TTS HTTP client and LLM provider wiring.

Fallback (edge-tts → gTTS) runs inside the isolated TTS container; the API
client surfaces the engine/fallback fields returned by that service.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.tts import SynthResult, TTSService

pytestmark = pytest.mark.asyncio


def test_legacy_coqui_provider_aliases_to_chatterbox(monkeypatch):
    """Old .env files with TTS_PROVIDER=coqui must keep working."""
    from app.services import tts as tts_module

    monkeypatch.setattr(tts_module.settings, "TTS_PROVIDER", "coqui")
    service = TTSService()
    assert service.provider == "chatterbox"


async def test_tts_client_maps_remote_synth_result(monkeypatch, tmp_path):
    """HTTP client maps remote JSON into SynthResult (including fallback)."""
    service = TTSService()
    out = str(tmp_path / "out.wav")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output_path": out,
        "engine": "edge-tts",
        "fallback": True,
        "voice_cloned": False,
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post = AsyncMock(return_value=mock_resp)

    monkeypatch.setattr("app.services.tts.httpx.AsyncClient", lambda **kwargs: mock_client)

    result = await service.synthesize("Hello world", out, speaker_wav=None, language="en")

    assert isinstance(result, SynthResult)
    assert result.engine == "edge-tts"
    assert result.fallback is True
    assert result.voice_cloned is False
    mock_client.post.assert_awaited()


async def test_tts_client_maps_chatterbox_success(monkeypatch, tmp_path):
    service = TTSService()
    out = str(tmp_path / "out.wav")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output_path": out,
        "engine": "chatterbox",
        "fallback": False,
        "voice_cloned": True,
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post = AsyncMock(return_value=mock_resp)

    monkeypatch.setattr("app.services.tts.httpx.AsyncClient", lambda **kwargs: mock_client)

    result = await service.synthesize("Hi", out, speaker_wav="/media/voice.wav", language="en")
    assert result.engine == "chatterbox"
    assert result.fallback is False
    assert result.voice_cloned is True


def test_llm_ollama_provider_uses_openai_compatible_client(monkeypatch):
    """LLM_PROVIDER=ollama wires an OpenAI client at the local base URL."""
    from app.services import llm as llm_module

    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_module.settings, "OPENAI_BASE_URL", None)
    monkeypatch.setattr(llm_module.settings, "OPENAI_API_KEY", "")

    service = llm_module.LLMService()
    assert service.provider == "openai"  # downstream paths are the OpenAI ones
    assert "localhost:11434" in str(service.client.base_url)


def test_llm_openai_provider_respects_custom_base_url(monkeypatch):
    from app.services import llm as llm_module

    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_module.settings, "OPENAI_BASE_URL", "http://vllm:8001/v1")
    monkeypatch.setattr(llm_module.settings, "OPENAI_API_KEY", "k")

    service = llm_module.LLMService()
    assert "vllm:8001" in str(service.client.base_url)
