"""Minimal in-memory fake of a ``websockets`` client connection.

Used to drive both the Deepgram Voice Agent client and the 60db TTS client
through their real receive loops without any network. The fake records every
outgoing frame (``sent``) and lets a test push incoming frames (``push``).

A ``responder`` callable may be supplied to script server replies: it is invoked
with each outgoing frame and may return an iterable of frames to push back,
mimicking a real server (e.g. reply ``SettingsApplied`` after ``Settings``).
"""

import asyncio
from typing import Any, Callable, Iterable, List, Optional

# Sentinel pushed to terminate the async iteration (simulates a server close).
_CLOSE = object()


class FakeWebSocket:
    """Async-iterable stand-in for ``websockets.WebSocketClientProtocol``."""

    def __init__(
        self,
        responder: Optional[Callable[[Any], Optional[Iterable[Any]]]] = None,
    ) -> None:
        self.sent: List[Any] = []
        self.closed = False
        self._incoming: asyncio.Queue = asyncio.Queue()
        self._responder = responder

    # --- outgoing (client -> server) ------------------------------------
    async def send(self, message: Any) -> None:
        if self.closed:
            raise ConnectionError("send() on closed FakeWebSocket")
        self.sent.append(message)
        if self._responder is not None:
            for reply in self._responder(message) or ():
                await self._incoming.put(reply)

    # --- incoming (server -> client), via async iteration ----------------
    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> Any:
        message = await self._incoming.get()
        if message is _CLOSE:
            raise StopAsyncIteration
        return message

    async def push(self, message: Any) -> None:
        """Inject a server -> client frame (str JSON or bytes audio)."""
        await self._incoming.put(message)

    async def close(self) -> None:
        self.closed = True
        await self._incoming.put(_CLOSE)

    async def ping(self) -> "asyncio.Future":
        loop = asyncio.get_event_loop()
        fut: "asyncio.Future" = loop.create_future()
        fut.set_result(None)
        return fut

    # --- convenience views ----------------------------------------------
    @property
    def sent_text(self) -> List[str]:
        return [m for m in self.sent if isinstance(m, str)]

    @property
    def sent_binary(self) -> List[bytes]:
        return [bytes(m) for m in self.sent if isinstance(m, (bytes, bytearray))]


def patch_connect(monkeypatch, module_path: str, fake: FakeWebSocket) -> None:
    """Patch ``<module_path>.websockets.connect`` to return ``fake``."""

    async def _fake_connect(*_args, **_kwargs) -> FakeWebSocket:
        return fake

    monkeypatch.setattr(f"{module_path}.websockets.connect", _fake_connect)
