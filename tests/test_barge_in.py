"""Barge-in: a caller speech-started event clears all queued downlink audio."""

import asyncio

from app.ai.duplex_base import AiEvent, AiEventType
from app.bridge import AudioAdapter, CallSession
from app.sip_async.audio_bridge import RTPAudioBridge
from app.sip_async.rtp_session import RTPSession
from tests.mock_ai_client import MockDuplexClient

FRAME = b"\x10\x00" * 160


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class TestAdapterClearDownlink:
    async def test_clear_downlink_drops_stream_pending_and_listeners(self) -> None:
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=10)
        rtp = RTPSession(local_port=40000, remote_addr=("127.0.0.1", 5004))
        RTPAudioBridge(rtp, adapter)  # registers the RTP tx queue listener

        await adapter.feed_ai_audio(FRAME * 3 + b"\x01\x00")  # 3 frames + partial
        for _ in range(4):
            rtp.tx_queue.put_nowait(FRAME)

        dropped = adapter.clear_downlink()

        assert dropped == 7
        assert adapter._pending_bytes == b""
        assert rtp.tx_queue.empty()
        assert adapter.get_tx_pcm16_8k_nowait() == b"\x00" * 320  # silence, nothing queued


class TestCallSessionBargeIn:
    async def test_speech_started_event_clears_queued_audio(self) -> None:
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=50)
        rtp = RTPSession(local_port=40001, remote_addr=("127.0.0.1", 5004))
        RTPAudioBridge(rtp, adapter)
        client = MockDuplexClient(sample_rate=8000)
        session = CallSession(audio_adapter=adapter, ai_client=client)

        await session.start()
        try:
            # Queued AI audio at every stage of the downlink path
            for _ in range(5):
                await adapter.feed_ai_audio(FRAME)
                rtp.tx_queue.put_nowait(FRAME)

            client._event_queue.put_nowait(
                AiEvent(type=AiEventType.TRANSCRIPT_PARTIAL, data={"event": "speech_started"})
            )

            await _wait_until(lambda: rtp.tx_queue.empty() and adapter._downlink_stream._queue.empty())
        finally:
            await session.stop()

    async def test_other_events_do_not_clear_audio(self) -> None:
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=50)
        client = MockDuplexClient(sample_rate=8000)
        session = CallSession(audio_adapter=adapter, ai_client=client)

        await session.start()
        try:
            for _ in range(5):
                await adapter.feed_ai_audio(FRAME)
            client._event_queue.put_nowait(
                AiEvent(type=AiEventType.TRANSCRIPT_PARTIAL, data={"event": "speech_stopped"})
            )
            await _wait_until(lambda: client._event_queue.empty())
            await asyncio.sleep(0.05)
            assert adapter._downlink_stream._queue.qsize() == 5
        finally:
            await session.stop()

    async def test_barge_in_clears_ai_client_audio_queue(self) -> None:
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=50)
        client = MockDuplexClient(sample_rate=8000)
        client._connected = True
        for _ in range(3):
            client._audio_queue.put_nowait(FRAME)
        session = CallSession(audio_adapter=adapter, ai_client=client)

        session._handle_barge_in()

        assert client._audio_queue.empty()
