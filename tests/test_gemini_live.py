"""Unit tests for GeminiLiveClient uplink message format."""

import base64
import json

import pytest


class _FakeWebSocket:
    """Minimal stand-in for websockets client used in tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True


class TestGeminiUplink:
    """Tests for send_pcm16_8k message format."""

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_uses_audio_blob_not_media_chunks(self) -> None:
        """Uplink must use realtimeInput.audio Blob; mediaChunks is deprecated
        and rejected by newer Live API models with WebSocket close 1007."""
        from app.ai.gemini_live import GeminiLiveClient

        client = GeminiLiveClient(api_key="k")
        ws = _FakeWebSocket()
        client._ws = ws  # type: ignore[assignment]
        client._connected = True

        # 320 bytes PCM16 silence @ 8kHz = 20ms
        await client.send_pcm16_8k(b"\x00" * 320)

        assert len(ws.sent) == 1
        ri = json.loads(ws.sent[0])["realtimeInput"]
        assert "mediaChunks" not in ri
        assert ri["audio"]["mimeType"] == "audio/pcm;rate=16000"
        # 8kHz -> 16kHz doubles samples: 320 bytes -> 640 bytes
        decoded = base64.b64decode(ri["audio"]["data"])
        assert len(decoded) == 640

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_validates_frame_size(self) -> None:
        from app.ai.gemini_live import GeminiLiveClient

        client = GeminiLiveClient(api_key="k")
        client._ws = _FakeWebSocket()  # type: ignore[assignment]
        client._connected = True

        with pytest.raises(ValueError, match="320"):
            await client.send_pcm16_8k(b"\x00" * 100)

    @pytest.mark.asyncio
    async def test_send_pcm16_8k_raises_when_not_connected(self) -> None:
        from app.ai.gemini_live import GeminiLiveClient

        client = GeminiLiveClient(api_key="k")
        with pytest.raises(ConnectionError):
            await client.send_pcm16_8k(b"\x00" * 320)
