"""Unit tests for GrokVoiceClient and Grok config wiring."""

import asyncio
import base64
import json
import os
from unittest.mock import patch

import pytest
import websockets
import websockets.exceptions


class TestGrokConfig:
    """Tests for Grok-related fields in AIConfig."""

    def test_grok_vendor_is_accepted(self) -> None:
        """AI_VENDOR=grok should be accepted (not coerced to mock)."""
        with patch.dict(os.environ, {"AI_VENDOR": "grok", "XAI_API_KEY": "x"}, clear=False):
            # Re-import to pick up the new env
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.vendor == "grok"

    def test_grok_defaults(self) -> None:
        """Grok config defaults match the spec."""
        with patch.dict(os.environ, {"AI_VENDOR": "grok", "XAI_API_KEY": "x"}, clear=False):
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.grok_model == "grok-voice-think-fast-2.0"
            assert cfg_module.config.ai.grok_voice == "eve"
            assert cfg_module.config.ai.grok_ws_endpoint == "wss://api.x.ai/v1/realtime"

    def test_grok_env_overrides(self) -> None:
        """GROK_MODEL / GROK_VOICE / GROK_WS_ENDPOINT env vars override defaults."""
        env = {
            "AI_VENDOR": "grok",
            "XAI_API_KEY": "x",
            "GROK_MODEL": "grok-voice-fast-1.0",
            "GROK_VOICE": "rex",
            "GROK_WS_ENDPOINT": "wss://example.invalid/realtime",
        }
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            assert cfg_module.config.ai.grok_model == "grok-voice-fast-1.0"
            assert cfg_module.config.ai.grok_voice == "rex"
            assert cfg_module.config.ai.grok_ws_endpoint == "wss://example.invalid/realtime"


class TestGrokConstructor:
    """Tests for GrokVoiceClient.__init__."""

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor raises ValueError if no api_key arg and XAI_API_KEY not set."""
        from app.ai.grok_voice import GrokVoiceClient

        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Grok API key"):
            GrokVoiceClient(api_key=None)

    def test_constructor_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor reads XAI_API_KEY from env when api_key arg is None."""
        from app.ai.grok_voice import GrokVoiceClient

        monkeypatch.setenv("XAI_API_KEY", "env-key")
        client = GrokVoiceClient(api_key=None)
        assert client._api_key == "env-key"

    def test_constructor_defaults(self) -> None:
        """Default model, voice, endpoint match spec."""
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        assert client._model == "grok-voice-think-fast-2.0"
        assert client._voice == "eve"
        assert client._ws_url == "wss://api.x.ai/v1/realtime"
        assert client._sample_rate == 8000
        assert client._frame_ms == 20


class _FakeWebSocket:
    """Minimal stand-in for websockets.WebSocketClientProtocol used in tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    async def recv(self) -> str:  # pragma: no cover - overridden per test as needed
        await asyncio.Event().wait()
        return ""


class TestGrokUplink:
    """Tests for send_pcm16_8k."""

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_sends_input_audio_buffer_append(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]
        client._connected = True

        # 320 bytes of PCM16 silence = 160 mu-law bytes
        await client.send_pcm16_8k(b"\x00" * 320)

        assert len(ws.sent) == 1
        msg = json.loads(ws.sent[0])
        assert msg["type"] == "input_audio_buffer.append"
        decoded = base64.b64decode(msg["audio"])
        assert len(decoded) == 160  # mu-law @ 8kHz, 20ms = 160 bytes

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_validates_frame_size(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        client._ws = _FakeWebSocket()  # type: ignore[assignment]
        client._connected = True

        with pytest.raises(ValueError, match="320"):
            await client.send_pcm16_8k(b"\x00" * 100)

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_raises_when_not_connected(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        with pytest.raises(ConnectionError):
            await client.send_pcm16_8k(b"\x00" * 320)


class TestGrokMessageProcessing:
    """Tests for _process_message → event/audio queue dispatch."""

    @pytest.mark.asyncio
    async def test_session_created_sets_event_and_emits_connected(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({"type": "session.created", "session": {"id": "s1"}})

        assert client._session_created_event.is_set()
        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.CONNECTED

    @pytest.mark.asyncio
    async def test_conversation_created_sets_event_and_emits_connected(self) -> None:
        # The live xAI realtime API uses conversation.created as the connection-ready
        # signal, not session.created. Verified against api.x.ai 2026-05-04.
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({"type": "conversation.created", "conversation": {"id": "c1"}})

        assert client._session_created_event.is_set()
        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.CONNECTED

    @pytest.mark.asyncio
    async def test_ping_is_handled_quietly(self) -> None:
        # xAI server emits application-level "ping" as keepalive; no response needed.
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        await client._process_message({"type": "ping"})

        # No event queued, no audio queued, no exception raised.
        assert client._event_queue.empty()
        assert client._audio_queue.empty()
        assert not client._session_created_event.is_set()

    @pytest.mark.asyncio
    async def test_session_updated_sets_event_and_emits(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({"type": "session.updated", "session": {}})

        assert client._session_updated_event.is_set()
        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.SESSION_UPDATED

    @pytest.mark.asyncio
    async def test_audio_delta_decoded_to_pcm16(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        # 160 bytes mu-law silence (0x7F is ~zero in mu-law)
        ulaw = bytes([0x7F] * 160)
        await client._process_message({
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(ulaw).decode("utf-8"),
        })

        chunk = client._audio_queue.get_nowait()
        assert len(chunk) == 320  # PCM16 = 2x mu-law

    @pytest.mark.asyncio
    async def test_speech_started_emits_partial_event(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({"type": "input_audio_buffer.speech_started"})

        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.TRANSCRIPT_PARTIAL
        assert evt.data == {"event": "speech_started"}

    @pytest.mark.asyncio
    async def test_transcription_completed_emits_final(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hello world",
        })

        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.TRANSCRIPT_FINAL
        assert evt.data == {"text": "hello world"}

    @pytest.mark.asyncio
    async def test_error_event_emitted(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient
        from app.ai.duplex_base import AiEventType

        client = GrokVoiceClient(api_key="k")
        await client._process_message({
            "type": "error",
            "error": {"message": "bad thing"},
        })

        evt = client._event_queue.get_nowait()
        assert evt.type == AiEventType.ERROR
        assert evt.error == "bad thing"

    @pytest.mark.asyncio
    async def test_unknown_event_does_not_raise(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        # Should be silently logged, no exception, no queue entries
        await client._process_message({"type": "some.unknown.event"})

        assert client._event_queue.empty()
        assert client._audio_queue.empty()


class TestGrokSessionConfig:
    """Tests for _configure_session and _send_greeting."""

    @pytest.mark.asyncio
    async def test_configure_session_payload(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(
            api_key="k",
            model="grok-voice-think-fast-2.0",
            voice="eve",
            instructions="be terse",
        )
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]

        await client._configure_session()

        assert len(ws.sent) == 1
        msg = json.loads(ws.sent[0])
        assert msg["type"] == "session.update"
        sess = msg["session"]
        assert sess["model"] == "grok-voice-think-fast-2.0"
        assert sess["voice"] == "eve"
        # xAI's session schema uses "instructions" (NOT "system_prompt")
        assert sess["instructions"] == "be terse"
        # xAI uses session.audio.{input,output}.format.{type,rate} with
        # "audio/pcmu" for G.711 μ-law and explicit rate 8000 for telephony
        # (default is 24000).
        assert sess["audio"]["input"]["format"]["type"] == "audio/pcmu"
        assert sess["audio"]["input"]["format"]["rate"] == 8000
        assert sess["audio"]["output"]["format"]["type"] == "audio/pcmu"
        assert sess["audio"]["output"]["format"]["rate"] == 8000
        assert sess["turn_detection"]["type"] == "server_vad"

    @pytest.mark.asyncio
    async def test_send_greeting_sends_response_create(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k", greeting="hi there")
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]

        await client._send_greeting()

        assert len(ws.sent) == 1
        msg = json.loads(ws.sent[0])
        assert msg["type"] == "response.create"
        assert msg["response"]["instructions"] == "hi there"
        assert msg["response"]["metadata"]["response_purpose"] == "greeting"
        # xAI requires a non-empty client_event_id in metadata
        client_event_id = msg["response"]["metadata"]["client_event_id"]
        assert isinstance(client_event_id, str) and len(client_event_id) > 0

    @pytest.mark.asyncio
    async def test_send_greeting_noop_when_unset(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k", greeting=None)
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]

        await client._send_greeting()

        assert ws.sent == []


class _RecvControlledFakeWS(_FakeWebSocket):
    """Fake WS where recv() yields scripted messages, then blocks."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        for m in messages:
            self._queue.put_nowait(m)
        self._closed_event = asyncio.Event()

    async def recv(self) -> str:
        if not self._queue.empty():
            return await self._queue.get()
        # After scripted messages exhausted, block until close
        await self._closed_event.wait()
        raise websockets.exceptions.ConnectionClosed(None, None)  # type: ignore[arg-type]

    async def close(self) -> None:
        await super().close()
        self._closed_event.set()


class TestGrokLifecycle:
    """Tests for connect/close/message-handler integration."""

    @pytest.mark.asyncio
    async def test_connect_completes_after_session_created_and_updated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.ai import grok_voice
        from app.ai.grok_voice import GrokVoiceClient

        scripted = [
            json.dumps({"type": "session.created", "session": {"id": "s"}}),
            json.dumps({"type": "session.updated", "session": {}}),
        ]
        fake_ws = _RecvControlledFakeWS(scripted)

        async def fake_connect(*args: object, **kwargs: object) -> _RecvControlledFakeWS:
            return fake_ws

        monkeypatch.setattr(grok_voice.websockets, "connect", fake_connect)

        client = GrokVoiceClient(api_key="k")
        await client.connect()

        assert client._connected is True
        assert client._session_created_event.is_set()
        assert client._session_updated_event.is_set()
        # session.update was sent during connect
        types = [json.loads(m)["type"] for m in fake_ws.sent]
        assert "session.update" in types

        await client.close()

    @pytest.mark.asyncio
    async def test_connect_sends_greeting_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.ai import grok_voice
        from app.ai.grok_voice import GrokVoiceClient

        scripted = [
            json.dumps({"type": "session.created", "session": {}}),
            json.dumps({"type": "session.updated", "session": {}}),
        ]
        fake_ws = _RecvControlledFakeWS(scripted)

        async def fake_connect(*args: object, **kwargs: object) -> _RecvControlledFakeWS:
            return fake_ws

        monkeypatch.setattr(grok_voice.websockets, "connect", fake_connect)

        client = GrokVoiceClient(api_key="k", greeting="welcome")
        await client.connect()

        types = [json.loads(m)["type"] for m in fake_ws.sent]
        assert "response.create" in types
        await client.close()

    @pytest.mark.asyncio
    async def test_close_unblocks_receive_chunks(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        client._connected = True

        # close() should put a sentinel and flip _connected
        await client.close()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_receive_chunks_yields_then_stops(self) -> None:
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        client._connected = True
        await client._audio_queue.put(b"\x00" * 320)

        gen = client.receive_chunks()
        first = await gen.__anext__()
        assert first == b"\x00" * 320

        # Simulate close: flip flag and push sentinel
        client._connected = False
        await client._audio_queue.put(b"")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    @pytest.mark.asyncio
    async def test_events_yields_queued_event(self) -> None:
        from app.ai.duplex_base import AiEvent, AiEventType
        from app.ai.grok_voice import GrokVoiceClient

        client = GrokVoiceClient(api_key="k")
        client._connected = True
        await client._event_queue.put(AiEvent(type=AiEventType.CONNECTED))

        gen = client.events()
        evt = await gen.__anext__()
        assert evt.type == AiEventType.CONNECTED


class TestGrokFactory:
    """Tests for create_ai_client() vendor=grok branch."""

    def test_create_ai_client_returns_grok_client(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # Build minimal agent_prompt.yaml
        prompt = tmp_path / "agent_prompt.yaml"
        prompt.write_text("instructions: be helpful\ngreeting: hi\n")

        env = {
            "AI_VENDOR": "grok",
            "XAI_API_KEY": "k",
            "AGENT_PROMPT_FILE": str(prompt),
        }
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            # Re-import main so it picks up the reloaded config singleton
            from app import main as main_module
            reload(main_module)

            client = main_module.create_ai_client()

        from app.ai.grok_voice import GrokVoiceClient
        assert isinstance(client, GrokVoiceClient)
        assert client._model == "grok-voice-think-fast-2.0"
        assert client._voice == "eve"
        assert client._greeting == "hi"

    def test_create_ai_client_grok_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force XAI_API_KEY to empty string. Using delenv alone is not robust because
        # config.py calls load_dotenv() at module load time, which would re-inject
        # XAI_API_KEY from a real .env file. Setting an empty string in os.environ
        # overrides the .env file value (load_dotenv doesn't replace existing env vars).
        env = {"AI_VENDOR": "grok", "XAI_API_KEY": ""}
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            from app import config as cfg_module
            reload(cfg_module)
            from app import main as main_module
            reload(main_module)

            with pytest.raises(ValueError, match="Grok API key"):
                main_module.create_ai_client()
