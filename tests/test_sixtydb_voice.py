"""Unit tests for the 60db voice (SPEAK_PROVIDER=60db) integration.

Covers:
- config wiring (SPEAK_PROVIDER / SIXTYDB_* env vars)
- SixtyDBTTSClient message handling (create_context payload, audio_chunk decode)
- DeepgramAgentClient routing when 60db is the speak provider
"""

import base64
import json
import os
from importlib import reload
from unittest.mock import patch

import pytest

from app.utils.codec import Codec


class TestSpeakProviderConfig:
    """SPEAK_PROVIDER / SIXTYDB_* config fields."""

    def test_defaults_to_deepgram(self) -> None:
        with patch.dict(os.environ, {"AI_VENDOR": "deepgram"}, clear=False):
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.speak_provider == "deepgram"

    def test_sixtydb_env_overrides(self) -> None:
        env = {
            "AI_VENDOR": "deepgram",
            "SPEAK_PROVIDER": "60db",
            "SIXTYDB_API_KEY": "sk_live_test",
            "SIXTYDB_VOICE_ID": "voice-123",
        }
        with patch.dict(os.environ, env, clear=False):
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.speak_provider == "60db"
            assert cfg_module.config.ai.sixtydb_api_key == "sk_live_test"
            assert cfg_module.config.ai.sixtydb_voice_id == "voice-123"


class TestSixtyDBTTSClient:
    """SixtyDBTTSClient construction and message handling."""

    def test_missing_api_key_raises(self) -> None:
        from app.ai.sixtydb_tts import SixtyDBTTSClient
        with pytest.raises(ValueError):
            SixtyDBTTSClient(api_key="")

    def test_non_8k_rejected(self) -> None:
        from app.ai.sixtydb_tts import SixtyDBTTSClient
        with pytest.raises(ValueError):
            SixtyDBTTSClient(api_key="k", sample_rate=16000)

    def test_create_context_payload(self) -> None:
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        sent: list[str] = []

        class _FakeWS:
            async def send(self, msg: str) -> None:
                sent.append(msg)

        client = SixtyDBTTSClient(api_key="k", voice_id="voice-xyz")
        client._ws = _FakeWS()  # type: ignore[assignment]

        import asyncio
        asyncio.run(client._send_create_context())

        payload = json.loads(sent[0])["create_context"]
        assert payload["voice_id"] == "voice-xyz"
        assert payload["audio_config"] == {
            "audio_encoding": "MULAW",
            "sample_rate_hertz": 8000,
        }

    def test_audio_chunk_decoded_and_forwarded(self) -> None:
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        received: list[bytes] = []

        async def _on_audio(b: bytes) -> None:
            received.append(b)

        client = SixtyDBTTSClient(api_key="k", on_audio=_on_audio)

        raw = b"\xff\x7f\x00\x10"
        msg = json.dumps(
            {"audio_chunk": {"context_id": "c", "audioContent": base64.b64encode(raw).decode()}}
        )

        import asyncio
        asyncio.run(client._handle_message(msg))
        assert received == [raw]

    def test_context_created_sets_ready(self) -> None:
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        client = SixtyDBTTSClient(api_key="k")
        import asyncio
        asyncio.run(client._handle_message(json.dumps({"context_created": {"context_id": "c"}})))
        assert client._context_ready.is_set()


class TestDeepgramSpeakRouting:
    """DeepgramAgentClient behavior when 60db is the speak provider."""

    def _client(self):
        from app.ai.deepgram_agent import DeepgramAgentClient
        return DeepgramAgentClient(
            api_key="dg",
            speak_provider="60db",
            sixtydb_api_key="sk_live_test",
            sixtydb_voice_id="voice-1",
        )

    def test_requires_sixtydb_key(self) -> None:
        from app.ai.deepgram_agent import DeepgramAgentClient
        with pytest.raises(ValueError):
            DeepgramAgentClient(api_key="dg", speak_provider="60db", sixtydb_api_key="")

    def test_creates_sixtydb_client(self) -> None:
        client = self._client()
        assert client._sixtydb is not None

    def test_deepgram_binary_audio_ignored(self) -> None:
        """In 60db mode, Deepgram's own TTS audio must not be queued."""
        client = self._client()
        import asyncio
        asyncio.run(client._handle_binary_audio(b"\xff" * 160))
        assert client._audio_queue.empty()

    def test_greeting_not_sent_to_deepgram(self) -> None:
        """Greeting is spoken by 60db, so it must not be in the Deepgram Settings."""
        from app.ai.deepgram_agent import DeepgramAgentClient

        sent: list[str] = []

        class _FakeWS:
            async def send(self, msg: str) -> None:
                sent.append(msg)

        client = DeepgramAgentClient(
            api_key="dg",
            speak_provider="60db",
            sixtydb_api_key="sk_live_test",
            greeting="Hello there!",
        )
        client._ws = _FakeWS()  # type: ignore[assignment]
        import asyncio
        asyncio.run(client._send_session_config())
        agent_cfg = json.loads(sent[0])["agent"]
        assert "greeting" not in agent_cfg

    def test_sixtydb_audio_enqueued_as_pcm16(self) -> None:
        """60db's mu-law chunks are converted to PCM16 and queued for the bridge."""
        client = self._client()
        raw_ulaw = b"\xff\x7f\x00\x10"
        import asyncio
        asyncio.run(client._enqueue_agent_audio(raw_ulaw))
        queued = client._audio_queue.get_nowait()
        assert queued == Codec.ulaw_to_pcm16(raw_ulaw)
        assert client._received_first_audio is True
        assert client._agent_speaking is True
