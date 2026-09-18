"""Base protocol and types for AI duplex communication."""

import asyncio
import time
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, AsyncIterator, Dict, Optional, Protocol, runtime_checkable

import structlog

# Event markers (AiEvent.data["event"]) that mean the caller started speaking.
# Vendors report them as TRANSCRIPT_PARTIAL events; the bridge uses them to
# drop queued downlink audio so the caller can interrupt the AI (barge-in).
BARGE_IN_EVENTS = frozenset({
    "speech_started",  # OpenAI Realtime / Grok: input_audio_buffer.speech_started
    "user_started_speaking",  # Deepgram: UserStartedSpeaking
    "interrupted",  # Gemini Live: serverContent.interrupted
})


class AiEventType(Enum):
    """AI event types (simplified for connection management only)."""

    CONNECTED = auto()
    DISCONNECTED = auto()
    ERROR = auto()
    SESSION_UPDATED = auto()

    # Optional debug/logging events
    TRANSCRIPT_PARTIAL = auto()
    TRANSCRIPT_FINAL = auto()


@dataclass
class AiEvent:
    """AI event data."""

    type: AiEventType
    data: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    error: Optional[str] = None


def is_barge_in_event(event: AiEvent) -> bool:
    """Return True if the event reports that the caller started speaking.

    Args:
        event: AI event

    Returns:
        True for barge-in events (user speech started / response interrupted)
    """
    if event.type != AiEventType.TRANSCRIPT_PARTIAL or not event.data:
        return False
    return event.data.get("event") in BARGE_IN_EVENTS


def describe_error(error: BaseException) -> str:
    """Return a non-empty, human-readable description of an exception.

    Some exceptions (notably TimeoutError from asyncio.timeout) have an empty
    message; the exception type is used so logs never show a blank reason.

    Args:
        error: Exception to describe

    Returns:
        Description such as "TimeoutError: timed out" or "OSError: refused"
    """
    if isinstance(error, TimeoutError) and not str(error):
        return "TimeoutError: timed out"
    message = str(error)
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message}"


class AudioChunkQueue:
    """Non-blocking FIFO for received AI audio (PCM16 @ 8kHz chunks).

    The WebSocket read loop must never wait on audio consumers: a blocked
    reader stops processing barge-in events and WebSocket pongs. Chunks are
    therefore queued without a count limit (put_nowait never raises).

    Memory is bounded by buffered audio duration instead: AI replies arrive
    faster than real time, so a long reply can be buffered here while it is
    played out at 1x. When more than max_seconds of audio is buffered the
    oldest audio is dropped (a warning is logged by the caller). At PCM16 @ 8kHz
    the default of 300s is about 4.8 MB per call. Barge-in clears the queue.

    The API mirrors the asyncio.Queue subset used by the AI clients.
    """

    BYTES_PER_SECOND = 16000  # PCM16 @ 8kHz
    DEFAULT_MAX_SECONDS = 300.0

    def __init__(self, max_seconds: float = DEFAULT_MAX_SECONDS) -> None:
        """Initialize the queue.

        Args:
            max_seconds: Maximum buffered audio duration before the oldest
                chunks are dropped
        """
        self._chunks: deque[bytes] = deque()
        self._bytes = 0
        self._max_bytes = int(max_seconds * self.BYTES_PER_SECOND)
        self._waiters: deque[asyncio.Future[None]] = deque()
        self.dropped_chunks = 0

    @property
    def buffered_bytes(self) -> int:
        """Total audio bytes currently queued."""
        return self._bytes

    @property
    def buffered_seconds(self) -> float:
        """Audio duration currently queued (seconds)."""
        return self._bytes / self.BYTES_PER_SECOND

    def qsize(self) -> int:
        """Number of queued chunks."""
        return len(self._chunks)

    def empty(self) -> bool:
        """True if no chunk is queued."""
        return not self._chunks

    def put_nowait(self, chunk: bytes) -> int:
        """Queue a chunk without blocking.

        Args:
            chunk: PCM16 chunk (b"" is the end-of-stream marker)

        Returns:
            Number of old chunks dropped to respect the duration limit
        """
        self._chunks.append(chunk)
        self._bytes += len(chunk)
        dropped = 0
        while self._bytes > self._max_bytes and len(self._chunks) > 1:
            old = self._chunks.popleft()
            self._bytes -= len(old)
            dropped += 1
        self.dropped_chunks += dropped
        self._wake_one()
        return dropped

    async def put(self, chunk: bytes) -> None:
        """Queue a chunk (never waits; async for asyncio.Queue compatibility)."""
        self.put_nowait(chunk)

    def get_nowait(self) -> bytes:
        """Remove and return the oldest chunk.

        Raises:
            asyncio.QueueEmpty: If no chunk is queued
        """
        if not self._chunks:
            raise asyncio.QueueEmpty
        chunk = self._chunks.popleft()
        self._bytes -= len(chunk)
        return chunk

    async def get(self) -> bytes:
        """Wait for and return the oldest chunk."""
        while not self._chunks:
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
            try:
                await waiter
            except BaseException:
                waiter.cancel()
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
                # Pass the wake-up on if this waiter was woken but cancelled
                if self._chunks:
                    self._wake_one()
                raise
        return self.get_nowait()

    def clear(self) -> int:
        """Drop all queued chunks.

        Returns:
            Number of chunks dropped
        """
        dropped = len(self._chunks)
        self._chunks.clear()
        self._bytes = 0
        return dropped

    def _wake_one(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return


@runtime_checkable
class AiDuplexClient(Protocol):
    """Protocol for AI duplex client implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to AI service.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connection to AI service."""
        ...

    @abstractmethod
    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to AI.

        Each implementation converts to its required format:
        - OpenAI: PCM16 8kHz → PCM16 24kHz
        - Deepgram: PCM16 8kHz → mulaw 8kHz
        - Mock: PCM16 8kHz (passthrough)

        Args:
            frame_20ms: 20ms PCM16 frame @ 8kHz (320 bytes)

        Raises:
            ConnectionError: If not connected
            ValueError: If frame size is invalid
        """
        ...

    @abstractmethod
    def receive_chunks(self) -> AsyncIterator[bytes]:
        """Iterate over received audio chunks from AI.

        Yields:
            PCM16 audio chunks @ 8kHz (variable size)
            - OpenAI Realtime: Converts G.711 → PCM16 internally (typically 1600-4000 bytes)
            - Deepgram Agent: Returns PCM16 directly (variable size)
            - Mock: Returns PCM16 (320 bytes/20ms frames)

        Raises:
            ConnectionError: If connection is lost
        """
        ...

    @abstractmethod
    def events(self) -> AsyncIterator[AiEvent]:
        """Iterate over events from AI.

        Yields:
            AI events

        Raises:
            ConnectionError: If connection is lost
        """
        ...

    @abstractmethod
    async def update_session(self, config: Dict[str, Any]) -> None:
        """Update session configuration.

        Args:
            config: Session configuration dictionary

        Raises:
            ConnectionError: If not connected
            ValueError: If configuration is invalid
        """
        ...

    @abstractmethod
    async def ping(self) -> bool:
        """Check connection health.

        Returns:
            True if connection is healthy

        Raises:
            ConnectionError: If connection check fails
        """
        ...

    @abstractmethod
    async def reconnect(self) -> None:
        """Reconnect to AI service.

        Raises:
            ConnectionError: If reconnection fails
        """
        ...


@dataclass
class SessionConfig:
    """Common session configuration."""

    # Audio configuration
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm16"

    # Voice configuration
    voice: Optional[str] = None
    language: str = "en-US"

    # Interaction configuration (VAD/barge-in handled by AI service)
    enable_vad: bool = True  # Informational only - AI service controls VAD
    silence_threshold_ms: int = 500

    # Model configuration
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # Custom instructions
    system_prompt: Optional[str] = None
    initial_context: Optional[str] = None


class AiDuplexBase:
    """Base class for AI duplex clients with common functionality."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20) -> None:
        """Initialize base client.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_ms: Frame duration in milliseconds
        """
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._frame_size = (sample_rate * frame_ms * 2) // 1000  # PCM16 = 2 bytes per sample
        self._connected = False
        self._events_dropped = 0
        # Monotonic time of the last message received from the AI service
        self._last_receive_time = 0.0

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        """Get expected frame size in bytes."""
        return self._frame_size

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def validate_frame(self, frame: bytes) -> None:
        """Validate audio frame size.

        Args:
            frame: Audio frame to validate

        Raises:
            ValueError: If frame size is invalid
        """
        if len(frame) != self._frame_size:
            raise ValueError(
                f"Invalid frame size: expected {self._frame_size}, got {len(frame)}"
            )

    def create_session_config(self, **kwargs: Any) -> SessionConfig:
        """Create session configuration.

        Args:
            **kwargs: Configuration parameters

        Returns:
            Session configuration object
        """
        return SessionConfig(
            sample_rate=self._sample_rate,
            **kwargs
        )

    @property
    def last_receive_time(self) -> float:
        """time.monotonic() of the last message received from the AI service (0 if none)."""
        return getattr(self, "_last_receive_time", 0.0)

    def _mark_received(self) -> None:
        """Record that a message was received (used by the health check)."""
        self._last_receive_time = time.monotonic()

    def _queue_audio(self, chunk: bytes) -> None:
        """Queue received AI audio without ever blocking the read loop.

        Args:
            chunk: PCM16 @ 8kHz chunk
        """
        queue = getattr(self, "_audio_queue", None)
        if queue is None:
            return
        if isinstance(queue, AudioChunkQueue):
            dropped = queue.put_nowait(chunk)
            if dropped:
                structlog.get_logger(__name__).warning(
                    "AI audio backlog over limit, dropped oldest audio",
                    dropped_chunks=dropped,
                    dropped_total=queue.dropped_chunks,
                    buffered_seconds=round(queue.buffered_seconds, 1)
                )
            return
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(chunk)

    def _emit_event(self, event: AiEvent) -> None:
        """Queue an event without blocking the WebSocket read loop.

        When the event queue is full the oldest event is dropped, so a slow (or
        missing) events() consumer can never stall audio delivery.

        Args:
            event: Event to queue
        """
        queue: Optional[asyncio.Queue[AiEvent]] = getattr(self, "_event_queue", None)
        if queue is None:
            return

        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        self._events_dropped = getattr(self, "_events_dropped", 0) + 1
        if self._events_dropped == 1 or self._events_dropped % 100 == 0:
            structlog.get_logger(__name__).warning(
                "AI event queue full, dropped oldest event",
                dropped_total=self._events_dropped,
                queued_type=event.type.name
            )
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def clear_audio_queue(self) -> int:
        """Drop received AI audio that has not been consumed yet (barge-in).

        Only drains while connected, so the b"" end-of-stream marker queued on
        disconnect is never removed.

        Returns:
            Number of chunks dropped
        """
        queue = getattr(self, "_audio_queue", None)
        if queue is None or not self._connected:
            return 0
        if isinstance(queue, AudioChunkQueue):
            return queue.clear()

        dropped = 0
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            dropped += 1
        return dropped
