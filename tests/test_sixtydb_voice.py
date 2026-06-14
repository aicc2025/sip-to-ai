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

    def test_invalid_speak_provider_raises(self) -> None:
        """An unknown SPEAK_PROVIDER must fail fast, not silently use Deepgram."""
        with patch.dict(os.environ, {"SPEAK_PROVIDER": "elevenlabs"}, clear=False):
            from app import config as cfg_module
            with pytest.raises(ValueError, match="SPEAK_PROVIDER"):
                reload(cfg_module)

    def test_speak_provider_is_case_insensitive(self) -> None:
        """Valid values are accepted regardless of case (e.g. '60DB')."""
        with patch.dict(os.environ, {"SPEAK_PROVIDER": "60DB"}, clear=False):
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.speak_provider == "60db"


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

    def test_speak_sends_text_then_flush(self) -> None:
        """speak() emits send_text followed by flush_context for the context."""
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        sent: list[str] = []

        class _FakeWS:
            async def send(self, msg: str) -> None:
                sent.append(msg)

        client = SixtyDBTTSClient(api_key="k")
        client._ws = _FakeWS()  # type: ignore[assignment]
        client._connected = True

        import asyncio
        asyncio.run(client.speak("Hello world"))

        assert len(sent) == 2
        first = json.loads(sent[0])
        assert first["send_text"]["text"] == "Hello world"
        assert first["send_text"]["context_id"] == client._context_id
        second = json.loads(sent[1])
        assert second["flush_context"]["context_id"] == client._context_id

    def test_speak_ignores_empty_text(self) -> None:
        """Empty/whitespace text must not produce any 60db traffic."""
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        sent: list[str] = []

        class _FakeWS:
            async def send(self, msg: str) -> None:
                sent.append(msg)

        client = SixtyDBTTSClient(api_key="k")
        client._ws = _FakeWS()  # type: ignore[assignment]
        client._connected = True

        import asyncio
        asyncio.run(client.speak("   "))
        assert sent == []

    def test_speak_when_not_connected_is_noop(self) -> None:
        """speak() before connect() must not raise and must send nothing."""
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        client = SixtyDBTTSClient(api_key="k")
        import asyncio
        asyncio.run(client.speak("hi"))  # _ws is None, _connected False

    def test_flush_completed_invokes_callback(self) -> None:
        """flush_completed triggers the on_flush_complete callback (turn handoff)."""
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        flushed: list[bool] = []

        async def _on_flush() -> None:
            flushed.append(True)

        client = SixtyDBTTSClient(api_key="k", on_flush_complete=_on_flush)
        import asyncio
        asyncio.run(client._handle_message(json.dumps({"flush_completed": {"context_id": "c"}})))
        assert flushed == [True]

    def test_non_json_message_is_ignored(self) -> None:
        """A non-JSON frame must be tolerated, not crash the receive loop."""
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        client = SixtyDBTTSClient(api_key="k")
        import asyncio
        asyncio.run(client._handle_message("not json"))  # must not raise

    def test_ws_url_override(self) -> None:
        """Injectable ws_url lets tests/fakes point at a local endpoint."""
        from app.ai.sixtydb_tts import SixtyDBTTSClient

        client = SixtyDBTTSClient(api_key="k", ws_url="ws://localhost:9/ws")
        assert client._ws_url == "ws://localhost:9/ws"

        default = SixtyDBTTSClient(api_key="k")
        assert default._ws_url == SixtyDBTTSClient.WS_URL


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

    def test_assistant_conversation_text_routed_to_60db(self) -> None:
        """Deepgram's assistant ConversationText is synthesized by 60db."""
        client = self._client()

        spoken: list[str] = []

        class _FakeTTS:
            async def speak(self, text: str) -> None:
                spoken.append(text)

        client._sixtydb = _FakeTTS()  # type: ignore[assignment]
        msg = json.dumps(
            {"type": "ConversationText", "role": "assistant", "content": "Hi there!"}
        )
        import asyncio
        asyncio.run(client._handle_json_message(msg))
        assert spoken == ["Hi there!"]

    def test_user_conversation_text_not_spoken(self) -> None:
        """The caller's own transcript must never be sent back to 60db TTS."""
        client = self._client()

        spoken: list[str] = []

        class _FakeTTS:
            async def speak(self, text: str) -> None:
                spoken.append(text)

        client._sixtydb = _FakeTTS()  # type: ignore[assignment]
        msg = json.dumps(
            {"type": "ConversationText", "role": "user", "content": "what time is it"}
        )
        import asyncio
        asyncio.run(client._handle_json_message(msg))
        assert spoken == []

    def test_flush_done_clears_agent_speaking(self) -> None:
        """60db finishing an utterance re-opens the caller's audio path."""
        client = self._client()
        client._agent_speaking = True
        import asyncio
        asyncio.run(client._on_sixtydb_flush_done())
        assert client._agent_speaking is False

    def test_agent_audio_done_does_not_clear_speaking_in_60db(self) -> None:
        """Deepgram's AgentAudioDone must not end the turn — 60db owns the voice."""
        client = self._client()
        client._agent_speaking = True
        import asyncio
        asyncio.run(client._handle_json_message(json.dumps({"type": "AgentAudioDone"})))
        assert client._agent_speaking is True


class TestDefaultDeepgramPath:
    """Default behavior (SPEAK_PROVIDER unset/deepgram) must be unchanged."""

    def _client(self, **kwargs):
        from app.ai.deepgram_agent import DeepgramAgentClient
        return DeepgramAgentClient(api_key="dg", **kwargs)

    def test_no_sixtydb_client_created(self) -> None:
        assert self._client()._sixtydb is None

    def test_binary_audio_is_enqueued(self) -> None:
        """Deepgram's own TTS audio is played in the default path."""
        client = self._client()
        import asyncio
        asyncio.run(client._handle_binary_audio(b"\xff\x7f\x00\x10"))
        assert not client._audio_queue.empty()

    def test_agent_audio_done_clears_speaking(self) -> None:
        client = self._client()
        client._agent_speaking = True
        import asyncio
        asyncio.run(client._handle_json_message(json.dumps({"type": "AgentAudioDone"})))
        assert client._agent_speaking is False

    def test_conversation_text_is_noop(self) -> None:
        """Without 60db, ConversationText must not raise or queue audio."""
        client = self._client()
        msg = json.dumps(
            {"type": "ConversationText", "role": "assistant", "content": "Hi"}
        )
        import asyncio
        asyncio.run(client._handle_json_message(msg))
        assert client._audio_queue.empty()

    def test_greeting_sent_to_deepgram(self) -> None:
        """In the default path the greeting is rendered by Deepgram's own TTS."""
        sent: list[str] = []

        class _FakeWS:
            async def send(self, msg: str) -> None:
                sent.append(msg)

        client = self._client(greeting="Welcome!")
        client._ws = _FakeWS()  # type: ignore[assignment]
        import asyncio
        asyncio.run(client._send_session_config())
        agent_cfg = json.loads(sent[0])["agent"]
        assert agent_cfg["greeting"] == "Welcome!"
