"""AI audio never blocks the WebSocket reader; barge-in, health check and uplink failure handling."""

import asyncio
import base64
import json

import pytest
import structlog

from app.ai.duplex_base import AudioChunkQueue, describe_error
from app.ai.openai_realtime import OpenAIRealtimeClient
from app.bridge import AudioAdapter, CallSession
from app.bridge.call_session import AiHealthCheckFailed
from tests.mock_ai_client import MockDuplexClient

FRAME = b"\x01\x00" * 160


class TestAudioChunkQueue:
    async def test_put_never_blocks_and_get_waits(self) -> None:
        queue = AudioChunkQueue()
        for _ in range(10_000):
            queue.put_nowait(FRAME)  # far beyond the old maxsize=100
        assert queue.qsize() == 10_000

        empty = AudioChunkQueue()
        getter = asyncio.create_task(empty.get())
        await asyncio.sleep(0.01)
        assert not getter.done()
        empty.put_nowait(b"abc")
        assert await asyncio.wait_for(getter, 1.0) == b"abc"

    async def test_duration_limit_drops_oldest(self) -> None:
        queue = AudioChunkQueue(max_seconds=0.1)  # 1600 bytes
        for i in range(10):
            queue.put_nowait(bytes([i]) * 320)
        assert queue.buffered_bytes <= 1600
        assert queue.get_nowait()[0] == 5  # oldest five dropped
        assert queue.dropped_chunks == 5

    async def test_clear_and_cancelled_getter(self) -> None:
        queue = AudioChunkQueue()
        getter = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        getter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await getter
        queue.put_nowait(FRAME)
        queue.put_nowait(FRAME)
        assert queue.clear() == 2
        assert queue.empty() and queue.buffered_bytes == 0


def _pcm16_delta(seconds: float) -> dict:
    audio_24k = b"\x10\x00" * int(24000 * seconds)
    return {"type": "response.output_audio.delta", "delta": base64.b64encode(audio_24k).decode()}


class _ScriptedWs:
    """WebSocket stand-in that yields scripted server events."""

    def __init__(self, events: list[dict]) -> None:
        self._events = [json.dumps(e) for e in events]
        self.sent: list[str] = []

    async def recv(self) -> str:
        if self._events:
            return self._events.pop(0)
        await asyncio.sleep(3600)
        return ""

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        pass


class TestReaderNeverBlocks:
    async def test_speech_started_processed_behind_long_audio_backlog(self) -> None:
        # 200 x 0.5s = 100s of audio arrives before speech_started, and nobody
        # consumes the audio: the reader must still deliver the barge-in event
        events = [_pcm16_delta(0.5) for _ in range(200)] + [{"type": "input_audio_buffer.speech_started"}]
        client = OpenAIRealtimeClient(api_key="sk-test", audio_format="pcm16")
        client._ws = _ScriptedWs(events)  # type: ignore[assignment]
        client._connected = True

        handler = asyncio.create_task(client._message_handler())
        try:
            event = await asyncio.wait_for(client._event_queue.get(), 2.0)
        finally:
            handler.cancel()
        assert event.data == {"event": "speech_started"}
        assert client._audio_queue.qsize() == 200
        assert client.last_receive_time > 0

        assert client.clear_audio_queue() == 200
        assert client._audio_queue.empty()


class TestBargeInDuringFeed:
    async def test_clear_during_blocked_feed_drops_rest_of_chunk(self) -> None:
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=2)
        feed = asyncio.create_task(adapter.feed_ai_audio(FRAME * 10))
        await asyncio.sleep(0.01)
        assert not feed.done()  # downlink full: feed waits

        adapter.clear_downlink()
        await asyncio.sleep(0.01)
        # The waiting put may complete once, then the rest is dropped
        await asyncio.wait_for(feed, 1.0)
        assert adapter.downlink_stream._queue.qsize() == 0


class _DeadPingClient(MockDuplexClient):
    def __init__(self) -> None:
        super().__init__(sample_rate=8000)
        self.pings = 0

    async def ping(self) -> bool:
        self.pings += 1
        return False


class TestHealthCheck:
    async def test_recent_messages_skip_ping(self) -> None:
        client = _DeadPingClient()
        await client.connect()
        session = CallSession(AudioAdapter(), client)
        session.HEALTH_CHECK_INTERVAL = 0.05
        session._running = True

        async def keep_receiving() -> None:
            while True:
                client._mark_received()
                await asyncio.sleep(0.01)

        feeder = asyncio.create_task(keep_receiving())
        health = asyncio.create_task(session._health_safe())
        await asyncio.sleep(0.3)
        assert not health.done()
        assert client.pings == 0
        health.cancel()
        feeder.cancel()
        await asyncio.gather(health, feeder, return_exceptions=True)

    async def test_dead_connection_ends_session_without_reconnect(self) -> None:
        client = _DeadPingClient()
        reconnects: list[bool] = []

        async def reconnect() -> None:
            reconnects.append(True)

        client.reconnect = reconnect  # type: ignore[method-assign]
        session = CallSession(AudioAdapter(), client)
        session.HEALTH_CHECK_INTERVAL = 0.02
        await session.start()
        try:
            await asyncio.wait_for(session.wait_ended(), 2.0)
        finally:
            await session.stop()
        assert client.pings == CallSession.HEALTH_PING_ATTEMPTS
        assert reconnects == []

    async def test_health_task_raises_on_failed_pings(self) -> None:
        client = _DeadPingClient()
        session = CallSession(AudioAdapter(), client)
        session.HEALTH_CHECK_INTERVAL = 0.01
        session._running = True
        with pytest.raises(AiHealthCheckFailed):
            await asyncio.wait_for(session._health_safe(), 1.0)


class _BrokenSendClient(MockDuplexClient):
    def __init__(self) -> None:
        super().__init__(sample_rate=8000)
        self.sends = 0

    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        self.sends += 1
        raise ConnectionError("Not connected")


class TestUplinkAfterDisconnect:
    async def test_uplink_stops_after_connection_error_and_logs_once(self) -> None:
        adapter = AudioAdapter(uplink_capacity=100, downlink_capacity=10)
        client = _BrokenSendClient()
        session = CallSession(adapter, client)
        session._running = True
        for _ in range(50):
            adapter.on_rx_pcm16_8k(FRAME)

        with structlog.testing.capture_logs() as logs:
            await asyncio.wait_for(session._uplink_safe(), 1.0)

        assert client.sends == 1
        assert [e for e in logs if e["log_level"] == "error"] == []
        assert len([e for e in logs if e["event"] == "AI connection lost - stopping uplink"]) == 1


def test_describe_error_never_empty() -> None:
    assert describe_error(TimeoutError()) == "TimeoutError: timed out"
    assert describe_error(OSError("refused")) == "OSError: refused"
    assert describe_error(RuntimeError()) == "RuntimeError"
