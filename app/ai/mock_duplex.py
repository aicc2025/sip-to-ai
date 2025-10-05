"""Mock AI duplex client for testing."""

import asyncio
import time
from typing import AsyncIterator, Dict, Optional

import numpy as np
import structlog

from app.ai.duplex_base import AiDuplexBase, AiEvent, AiEventType


class MockDuplexClient(AiDuplexBase):
    """Mock AI duplex client for testing without external dependencies."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        echo_delay_ms: int = 500
    ) -> None:
        """Initialize mock client.

        Args:
            sample_rate: Audio sample rate
            frame_ms: Frame duration
            echo_delay_ms: Delay before echoing audio
        """
        super().__init__(sample_rate, frame_ms)
        self._echo_delay_ms = echo_delay_ms

        # Internal state
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue()
        self._echo_buffer: list[bytes] = []

        # Control
        self._stop_event = asyncio.Event()
        self._echo_task: Optional[asyncio.Task[None]] = None

        self._logger = structlog.get_logger(__name__)

    async def connect(self) -> None:
        """Connect to mock service."""
        if self._connected:
            return

        self._connected = True
        self._stop_event = asyncio.Event()

        # Start background echo task
        self._echo_task = asyncio.create_task(self._run_echo_task())

        # Send connected event
        await self._event_queue.put(
            AiEvent(
                type=AiEventType.CONNECTED,
                timestamp=time.time()
            )
        )

        self._logger.info("Mock AI client connected")

    async def close(self) -> None:
        """Close mock connection."""
        if not self._connected:
            return

        self._connected = False
        self._stop_event.set()

        # Send disconnected event
        await self._event_queue.put(
            AiEvent(
                type=AiEventType.DISCONNECTED,
                timestamp=time.time()
            )
        )

        # Cancel echo task
        if self._echo_task:
            self._echo_task.cancel()
            try:
                await self._echo_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Mock AI client disconnected")

    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to mock service.

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes/20ms)
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        self.validate_frame(frame_20ms)

        # Add to echo buffer (passthrough PCM16)
        self._echo_buffer.append(frame_20ms)

        # Simulate processing
        await asyncio.sleep(0.001)

    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Iterate over audio chunks from mock service.

        Yields:
            PCM16 audio chunks
        """
        while self._connected:
            try:
                # Wait for audio with timeout
                chunk = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=1.0
                )
                yield chunk
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error("Error in audio stream", error=str(e))
                break

    async def events(self) -> AsyncIterator[AiEvent]:
        """Iterate over events from mock service.

        Yields:
            Mock AI events
        """
        while self._connected:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                yield event
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error("Error in event stream", error=str(e))
                break

    async def update_session(self, config: Dict) -> None:
        """Update session configuration.

        Args:
            config: Session configuration
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        # Simulate configuration update
        await self._event_queue.put(
            AiEvent(
                type=AiEventType.SESSION_UPDATED,
                data=config,
                timestamp=time.time()
            )
        )

        self._logger.info("Session updated", config=config)

    async def ping(self) -> bool:
        """Check connection health.

        Returns:
            Always True for mock
        """
        return self._connected

    async def reconnect(self) -> None:
        """Reconnect to mock service."""
        await self.close()
        await asyncio.sleep(0.1)
        await self.connect()

    async def _run_echo_task(self) -> None:
        """Background task to echo audio with delay."""
        echo_delay_frames = self._echo_delay_ms // self._frame_ms

        while not self._stop_event.is_set():
            try:
                # Wait for echo delay
                await asyncio.sleep(self._frame_ms / 1000.0)

                # Check if we have enough buffered frames
                if len(self._echo_buffer) >= echo_delay_frames:
                    # Get frame from delay buffer
                    frame = self._echo_buffer.pop(0)

                    # Generate response audio (simple echo with modification)
                    samples = np.frombuffer(frame, dtype=np.int16)

                    # Apply simple effects
                    # 1. Reduce amplitude
                    samples = (samples * 0.7).astype(np.int16)

                    # 2. Add slight frequency shift (simulated)
                    t = np.arange(len(samples)) / self._sample_rate
                    shift = np.sin(2 * np.pi * 50 * t) * 100
                    samples = np.clip(samples + shift.astype(np.int16), -32768, 32767).astype(np.int16)

                    # Put in audio queue
                    await self._audio_queue.put(samples.tobytes())

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Echo task error", error=str(e))
                await asyncio.sleep(0.1)

class MockWebSocketDuplex(MockDuplexClient):
    """Mock duplex using WebSocket for more realistic testing.

    Note: This class is deprecated and may be removed in future versions.
    Use MockDuplexClient instead.
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8080/mock",
        sample_rate: int = 16000,
        frame_ms: int = 20
    ) -> None:
        """Initialize WebSocket mock client.

        Args:
            ws_url: WebSocket URL (ignored, uses in-memory queue)
            sample_rate: Audio sample rate
            frame_ms: Frame duration
        """
        super().__init__(sample_rate, frame_ms)
        self._ws_url = ws_url

    async def connect(self) -> None:
        """Connect via mock WebSocket."""
        await super().connect()
        self._logger.info("Mock WebSocket connected", url=self._ws_url)