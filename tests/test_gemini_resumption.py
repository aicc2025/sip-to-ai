"""GeminiLiveClient session resumption against a local fake Live API server.

The fake server speaks the subset of the BidiGenerateContent protocol the
client uses: setup -> setupComplete + sessionResumptionUpdate, goAway,
serverContent audio, and rejecting resume attempts with close 1011.
Like the real server (verified against gemini-3.8-live), it rejects a resume
while an earlier socket of the session is still open.
"""

import asyncio
import base64
import json
from typing import Any, AsyncIterator, Optional

import pytest
import websockets
from websockets.asyncio.server import ServerConnection
from websockets.protocol import State

from app.ai.duplex_base import AiEvent, AiEventType
from app.ai.gemini_live import GeminiLiveClient

# 20ms of PCM16 @ 24kHz -> 160 bytes after resampling to 8kHz
_AUDIO_24K = b"\x01\x00" * 480


def _audio_message() -> str:
    return json.dumps({
        "serverContent": {
            "modelTurn": {
                "parts": [{
                    "inlineData": {
                        "mimeType": "audio/pcm;rate=24000",
                        "data": base64.b64encode(_AUDIO_24K).decode(),
                    }
                }]
            }
        }
    })


class FakeGeminiServer:
    """Minimal Gemini Live server on 127.0.0.1."""

    def __init__(self) -> None:
        self.setups: list[dict[str, Any]] = []
        self.conns: list[ServerConnection] = []
        self.received: list[tuple[int, dict[str, Any]]] = []
        # Close the next N resume attempts (setup with a handle) with 1011
        self.reject_resume = 0
        # Close every setup with 1011 (resume and fresh)
        self.reject_all = False
        # Resumes rejected because an earlier socket was still open
        self.rejected_concurrent = 0
        self._server: Any = None
        self.url = ""

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/ws"

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def _handler(self, ws: ServerConnection) -> None:
        setup = json.loads(await ws.recv())["setup"]
        self.setups.append(setup)
        resume = bool(setup.get("sessionResumption", {}).get("handle"))
        if resume and any(c.state is State.OPEN for c in self.conns):
            self.rejected_concurrent += 1
            await ws.close(1011, "Internal error encountered")
            return
        if self.reject_all or (resume and self.reject_resume > 0):
            if resume and self.reject_resume > 0:
                self.reject_resume -= 1
            await ws.close(1011, "Internal error encountered")
            return

        index = len(self.conns)
        self.conns.append(ws)
        await ws.send(json.dumps({"setupComplete": {}}))
        await ws.send(json.dumps({
            "sessionResumptionUpdate": {"newHandle": f"handle-{index}", "resumable": True}
        }))
        try:
            async for raw in ws:
                self.received.append((index, json.loads(raw)))
        except websockets.exceptions.ConnectionClosed:
            pass

    @property
    def current(self) -> ServerConnection:
        return self.conns[-1]


@pytest.fixture
async def server() -> AsyncIterator[FakeGeminiServer]:
    srv = FakeGeminiServer()
    await srv.start()
    try:
        yield srv
    finally:
        await srv.stop()


def _client(server: FakeGeminiServer, **kwargs: Any) -> GeminiLiveClient:
    client = GeminiLiveClient(
        api_key="k",
        ws_url=server.url,
        resume_budget_sec=kwargs.pop("resume_budget_sec", 1.0),
        resume_initial_backoff_sec=0.05,
        resume_max_backoff_sec=0.2,
        **kwargs,
    )
    client.FRESH_SESSION_RETRY_DELAY_SEC = 0.01  # type: ignore[misc]
    client.SETUP_TIMEOUT_SEC = 2.0  # type: ignore[misc]
    return client


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class _Consumers:
    """Runs receive_chunks() and events() like the bridge does."""

    def __init__(self, client: GeminiLiveClient) -> None:
        self.chunks: list[bytes] = []
        self.events: list[AiEvent] = []
        self.audio_done = False
        self.events_done = False
        self._tasks = [
            asyncio.create_task(self._audio(client)),
            asyncio.create_task(self._events(client)),
        ]

    async def _audio(self, client: GeminiLiveClient) -> None:
        async for chunk in client.receive_chunks():
            self.chunks.append(chunk)
        self.audio_done = True

    async def _events(self, client: GeminiLiveClient) -> None:
        async for event in client.events():
            self.events.append(event)
        self.events_done = True

    def event_types(self) -> list[AiEventType]:
        return [e.type for e in self.events]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


def _resume_handle(setup: dict[str, Any]) -> Optional[str]:
    return setup.get("sessionResumption", {}).get("handle")


class TestSetupMessage:
    async def test_setup_requests_resumption_and_compression(self, server: FakeGeminiServer) -> None:
        client = _client(server)
        await client.connect()
        try:
            setup = server.setups[0]
            assert setup["sessionResumption"] == {}
            assert setup["contextWindowCompression"] == {"slidingWindow": {}}
        finally:
            await client.close()

    def test_setup_with_handle(self) -> None:
        client = GeminiLiveClient(api_key="k")
        setup = client._build_setup_message("abc")["setup"]
        assert setup["sessionResumption"] == {"handle": "abc"}

    async def test_connect_disables_library_keepalive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def spy(*args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            raise OSError("stop")

        monkeypatch.setattr("app.ai.gemini_live.websockets.connect", spy)
        client = GeminiLiveClient(api_key="k")
        with pytest.raises(ConnectionError):
            await client.connect()
        assert "ping_interval" in captured and captured["ping_interval"] is None

    async def test_resumption_update_stores_handle_and_is_not_unknown(self) -> None:
        client = GeminiLiveClient(api_key="k")
        await client._process_message({"sessionResumptionUpdate": {"newHandle": "h1", "resumable": True}})
        assert client._resume_handle == "h1"
        # Not resumable: keep the previous handle
        await client._process_message({"sessionResumptionUpdate": {"newHandle": "h2", "resumable": False}})
        await client._process_message({"sessionResumptionUpdate": {}})
        await client._process_message({})
        assert client._resume_handle == "h1"


class TestResumption:
    async def test_handle_stored_and_sent_on_reconnect(self, server: FakeGeminiServer) -> None:
        client = _client(server)
        await client.connect()
        consumers = _Consumers(client)
        try:
            await _wait_for(lambda: client._resume_handle == "handle-0")

            server.current.transport.abort()
            await _wait_for(lambda: len(server.conns) == 2)
            await _wait_for(lambda: not client.is_recovering)

            assert _resume_handle(server.setups[1]) == "handle-0"
            assert client.is_connected
            await _wait_for(lambda: client._resume_handle == "handle-1")
        finally:
            await consumers.stop()
            await client.close()

    async def test_go_away_swaps_socket_without_ending_streams(self, server: FakeGeminiServer) -> None:
        client = _client(server)
        await client.connect()
        consumers = _Consumers(client)
        try:
            await _wait_for(lambda: client._resume_handle == "handle-0")
            old_ws = client._ws

            await server.current.send(json.dumps({"goAway": {"timeLeft": "10s"}}))
            await _wait_for(lambda: len(server.conns) == 2)
            await _wait_for(lambda: not client.is_recovering)

            # Break-before-make: the old socket was closed before the resume
            assert _resume_handle(server.setups[1]) == "handle-0"
            assert server.rejected_concurrent == 0
            assert client._ws is not old_ws
            assert server.conns[0].close_code is not None
            # Closing the replaced socket must not start another recovery
            await asyncio.sleep(0.1)
            assert len(server.setups) == 2
            assert client._recoveries == 1

            # Streams keep running on the new socket
            await server.current.send(_audio_message())
            await _wait_for(lambda: len(consumers.chunks) == 1)
            await client.send_pcm16_8k(b"\x00" * 320)
            await _wait_for(lambda: any(i == 1 and "realtimeInput" in m for i, m in server.received))

            assert AiEventType.DISCONNECTED not in consumers.event_types()
            assert not consumers.audio_done and not consumers.events_done
            assert client.is_connected
        finally:
            await consumers.stop()
            await client.close()

    async def test_abrupt_close_recovers_with_handle(self, server: FakeGeminiServer) -> None:
        client = _client(server)
        await client.connect()
        consumers = _Consumers(client)
        try:
            await _wait_for(lambda: client._resume_handle == "handle-0")
            client._user_transcript_buffer = "partial"

            server.current.transport.abort()
            await _wait_for(lambda: client._ws_lost or len(server.conns) == 2)
            # Uplink during recovery: dropped silently, no exception
            for _ in range(5):
                await client.send_pcm16_8k(b"\x00" * 320)
            await _wait_for(lambda: len(server.conns) == 2)
            await _wait_for(lambda: not client.is_recovering)

            assert _resume_handle(server.setups[1]) == "handle-0"
            assert client._user_transcript_buffer == ""
            assert client._audio_queue.empty()  # no stale end-of-stream marker
            await server.current.send(_audio_message())
            await _wait_for(lambda: len(consumers.chunks) == 1)
            assert consumers.chunks[0] != b""
            assert AiEventType.DISCONNECTED not in consumers.event_types()
            assert not consumers.audio_done and not consumers.events_done
        finally:
            await consumers.stop()
            await client.close()

    async def test_resume_rejected_then_succeeds(self, server: FakeGeminiServer) -> None:
        client = _client(server, resume_budget_sec=5.0)
        await client.connect()
        consumers = _Consumers(client)
        try:
            await _wait_for(lambda: client._resume_handle == "handle-0")
            server.reject_resume = 3

            server.current.transport.abort()
            await _wait_for(lambda: len(server.conns) == 2)
            await _wait_for(lambda: not client.is_recovering)

            # initial + 3 rejected resumes + 1 successful resume
            assert len(server.setups) == 5
            assert all(_resume_handle(s) == "handle-0" for s in server.setups[1:])
            assert client.is_connected
            assert AiEventType.DISCONNECTED not in consumers.event_types()
        finally:
            await consumers.stop()
            await client.close()

    async def test_budget_exhausted_falls_back_to_fresh_session(self, server: FakeGeminiServer) -> None:
        client = _client(server, resume_budget_sec=0.5, greeting="Hello there")
        await client.connect()
        consumers = _Consumers(client)
        try:
            await _wait_for(lambda: client._resume_handle == "handle-0")
            server.reject_resume = 10_000

            server.current.transport.abort()
            await _wait_for(lambda: len(server.conns) == 2, timeout=10.0)
            await _wait_for(lambda: not client.is_recovering)

            resume_attempts = [s for s in server.setups[1:] if _resume_handle(s)]
            assert len(resume_attempts) >= 2
            assert server.setups[-1]["sessionResumption"] == {}
            assert client.is_connected
            # Fresh session: no greeting is repeated
            await client.send_pcm16_8k(b"\x00" * 320)
            await _wait_for(lambda: any(i == 1 for i, _ in server.received))
            assert not any(i == 1 and "clientContent" in m for i, m in server.received)
            # New session's handle replaces the dead one
            await _wait_for(lambda: client._resume_handle == "handle-1")
            assert AiEventType.DISCONNECTED not in consumers.event_types()
            assert not consumers.audio_done
        finally:
            await consumers.stop()
            await client.close()

    async def test_fresh_session_failure_ends_streams(self, server: FakeGeminiServer) -> None:
        client = _client(server, resume_budget_sec=0.3)
        await client.connect()
        consumers = _Consumers(client)
        try:
            await _wait_for(lambda: client._resume_handle == "handle-0")
            server.reject_all = True

            server.current.transport.abort()
            await _wait_for(lambda: consumers.audio_done and consumers.events_done, timeout=10.0)

            assert not client.is_connected
            assert AiEventType.DISCONNECTED in consumers.event_types()
            fresh = [s for s in server.setups[1:] if not _resume_handle(s)]
            assert len(fresh) == client.FRESH_SESSION_ATTEMPTS
            with pytest.raises(ConnectionError):
                await client.send_pcm16_8k(b"\x00" * 320)
        finally:
            await consumers.stop()
            await client.close()

    async def test_close_during_recovery(self, server: FakeGeminiServer) -> None:
        client = _client(server, resume_budget_sec=30.0)
        await client.connect()
        consumers = _Consumers(client)
        try:
            await _wait_for(lambda: client._resume_handle == "handle-0")
            server.reject_all = True

            server.current.transport.abort()
            await _wait_for(lambda: client.is_recovering and len(server.setups) >= 2)
            recovery = client._recovery_task

            await asyncio.wait_for(client.close(), timeout=3.0)

            assert recovery is not None and recovery.done()
            assert not client._reader_tasks
            assert not client.is_connected
            await _wait_for(lambda: consumers.audio_done)
            # close() is not a failure: no DISCONNECTED event from recovery
            assert AiEventType.DISCONNECTED not in consumers.event_types()
            setups = len(server.setups)
            await asyncio.sleep(0.3)
            assert len(server.setups) == setups  # no retries after close()
        finally:
            await consumers.stop()
            await client.close()

    async def test_reconnect_keeps_streams(self, server: FakeGeminiServer) -> None:
        client = _client(server)
        await client.connect()
        consumers = _Consumers(client)
        try:
            await _wait_for(lambda: client._resume_handle == "handle-0")
            await client.reconnect()
            assert len(server.conns) == 2
            assert _resume_handle(server.setups[1]) == "handle-0"
            assert server.rejected_concurrent == 0
            assert client.is_connected and not consumers.audio_done
        finally:
            await consumers.stop()
            await client.close()

    async def test_ping_healthy_during_recovery(self, server: FakeGeminiServer) -> None:
        client = _client(server, resume_budget_sec=30.0)
        await client.connect()
        try:
            assert await client.ping()
            server.reject_all = True
            server.current.transport.abort()
            await _wait_for(lambda: client.is_recovering)
            assert await client.ping()
        finally:
            await client.close()
