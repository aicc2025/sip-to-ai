"""Tests for the OpenAI Realtime client audio formats (pcmu passthrough and pcm16 @ 24kHz)."""

import base64
import json

import numpy as np
import pytest

from app.ai.openai_realtime import OpenAIRealtimeClient


class _FakeWebSocket:
    """Minimal stand-in for a websockets client connection that records sends."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    @property
    def events(self) -> list[dict]:
        return [json.loads(m) for m in self.sent]


def _connected_client(ws: _FakeWebSocket, **kwargs) -> OpenAIRealtimeClient:
    client = OpenAIRealtimeClient(api_key="sk-test", **kwargs)
    client._ws = ws  # type: ignore[assignment]
    client._connected = True
    return client


def _delta_event(audio: bytes) -> dict:
    return {"type": "response.output_audio.delta", "delta": base64.b64encode(audio).decode()}


class TestSessionConfig:
    def test_pcmu_session_config_is_unchanged(self) -> None:
        """Default mode keeps the original OpenAI session.update payload."""
        client = OpenAIRealtimeClient(api_key="sk-test")
        session = client._build_session_config()["session"]

        assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
        assert session["audio"]["output"]["format"] == {"type": "audio/pcmu"}
        assert session["audio"]["input"]["noise_reduction"] == {"type": "near_field"}
        assert session["audio"]["output"]["voice"] == "marin"

    def test_pcm16_session_config_uses_audio_pcm_24k(self) -> None:
        client = OpenAIRealtimeClient(
            api_key="sk-test", audio_format="pcm16", noise_reduction="none", voice=""
        )
        config = client._build_session_config()
        session = config["session"]

        assert config["type"] == "session.update"
        assert session["type"] == "realtime"
        assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
        # cascade accepts only a null noise_reduction
        assert session["audio"]["input"]["noise_reduction"] is None
        # empty voice is omitted so the server default applies
        assert "voice" not in session["audio"]["output"]

    async def test_configure_session_sends_payload(self) -> None:
        ws = _FakeWebSocket()
        client = _connected_client(ws, audio_format="pcm16")

        await client._configure_session()

        assert ws.events[0]["session"]["audio"]["input"]["format"]["type"] == "audio/pcm"


class TestUplink:
    async def test_pcmu_uplink_sends_160_bytes(self) -> None:
        ws = _FakeWebSocket()
        client = _connected_client(ws)

        await client.send_pcm16_8k(b"\x00" * 320)

        event = ws.events[0]
        assert event["type"] == "input_audio_buffer.append"
        assert len(base64.b64decode(event["audio"])) == 160

    async def test_pcm16_uplink_resamples_to_24k(self) -> None:
        ws = _FakeWebSocket()
        client = _connected_client(ws, audio_format="pcm16")
        tone = (np.sin(np.arange(160) * 2 * np.pi * 440 / 8000) * 8000).astype(np.int16)

        await client.send_pcm16_8k(tone.tobytes())

        event = ws.events[0]
        assert event["type"] == "input_audio_buffer.append"
        payload = base64.b64decode(event["audio"])
        # 160 samples @ 8kHz -> 480 samples @ 24kHz
        assert len(payload) == 960
        assert np.abs(np.frombuffer(payload, dtype=np.int16)).max() > 0

    async def test_pcm16_uplink_validates_frame_size(self) -> None:
        client = _connected_client(_FakeWebSocket(), audio_format="pcm16")

        with pytest.raises(ValueError, match="320"):
            await client.send_pcm16_8k(b"\x00" * 100)


class TestDownlink:
    async def test_pcmu_downlink_decodes_ulaw(self) -> None:
        client = _connected_client(_FakeWebSocket())

        await client._process_message(_delta_event(b"\xff" * 160))

        assert len(client._audio_queue.get_nowait()) == 320

    async def test_pcm16_downlink_resamples_to_8k(self) -> None:
        client = _connected_client(_FakeWebSocket(), audio_format="pcm16")

        # 480 samples @ 24kHz (20ms) -> 160 samples @ 8kHz
        await client._process_message(_delta_event(b"\x10\x00" * 480))

        assert len(client._audio_queue.get_nowait()) == 320

    async def test_pcm16_downlink_carries_partial_samples(self) -> None:
        """Chunks not aligned to 3 samples (or split mid-sample) lose no audio."""
        client = _connected_client(_FakeWebSocket(), audio_format="pcm16")
        audio = b"\x10\x00" * 600  # 600 samples @ 24kHz -> 200 samples @ 8kHz
        chunks = [audio[:7], audio[7:8], audio[8:501], audio[501:]]

        for chunk in chunks:
            await client._process_message(_delta_event(chunk))

        total = 0
        while not client._audio_queue.empty():
            out = client._audio_queue.get_nowait()
            assert len(out) % 2 == 0
            total += len(out)
        assert total == 400
        assert client._downlink_pcm24k_pending == b""


class TestGreetingFallback:
    async def test_rejected_out_of_band_greeting_is_resent_in_conversation(self) -> None:
        ws = _FakeWebSocket()
        client = _connected_client(ws, audio_format="pcm16", greeting="Hello")
        client._session_updated_event.set()

        await client._send_greeting()
        assert ws.events[0]["response"]["conversation"] == "none"

        await client._process_message(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "message": "unsupported value",
                    "param": "response.conversation",
                },
            }
        )

        assert len(ws.events) == 2
        retry = ws.events[1]
        assert retry["type"] == "response.create"
        assert "conversation" not in retry["response"]
        assert retry["response"]["instructions"] == "Hello"
        assert client._event_queue.empty()

    async def test_greeting_fallback_not_used_after_response_created(self) -> None:
        ws = _FakeWebSocket()
        client = _connected_client(ws, greeting="Hello")
        client._session_updated_event.set()

        await client._send_greeting()
        await client._process_message({"type": "response.created", "response": {}})
        await client._process_message(
            {"type": "error", "error": {"message": "x", "param": "response.conversation"}}
        )

        assert len(ws.events) == 1
        assert not client._event_queue.empty()
