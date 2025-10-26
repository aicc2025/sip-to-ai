"""Mock AI client for testing."""

import asyncio
from typing import AsyncIterator

import structlog

from app.ai.duplex_base import AiDuplexBase, AiEvent, AiEventType


class MockDuplexClient(AiDuplexBase):
    """Mock AI client that echoes audio back with delay."""

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_ms: int = 20,
        echo_delay_ms: int = 100
    ) -> None:
        """Initialize mock client.

        Args:
            sample_rate: Audio sample rate
            frame_ms: Frame duration in milliseconds
            echo_delay_ms: Echo delay in milliseconds
        """
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms)

        self._echo_delay_ms = echo_delay_ms
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue(maxsize=100)

        self._logger = structlog.get_logger(__name__)

    async def connect(self) -> None:
        """Connect to mock AI service."""
        if self._connected:
            return

        self._connected = True
        self._logger.info("Mock AI client connected")

        await self._event_queue.put(AiEvent(
            type=AiEventType.CONNECTED,
            data={"status": "connected"}
        ))

    async def close(self) -> None:
        """Close mock AI connection."""
        if not self._connected:
            return

        self._connected = False
        self._logger.info("Mock AI client closed")

        await self._event_queue.put(AiEvent(
            type=AiEventType.DISCONNECTED,
            data={"status": "disconnected"}
        ))

    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send audio to mock AI (echo back after delay).

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes/20ms)
        """
        if not self._connected:
            return

        # Echo back after delay
        await asyncio.sleep(self._echo_delay_ms / 1000.0)
        await self._audio_queue.put(frame_20ms)

    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Receive audio chunks from mock AI.

        Yields:
            PCM16 audio chunks @ 8kHz (320 bytes/20ms frames)
        """
        while self._connected:
            try:
                chunk = await self._audio_queue.get()
                yield chunk
            except Exception as e:
                self._logger.error("Audio stream error", error=str(e))
                break

    async def events(self) -> AsyncIterator[AiEvent]:
        """Iterate over events from mock AI.

        Yields:
            AI events
        """
        while self._connected:
            try:
                event = await self._event_queue.get()
                yield event
            except Exception as e:
                self._logger.error("Event stream error", error=str(e))
                break

    async def update_session(self, config: dict) -> None:
        """Update session configuration (no-op for mock).

        Args:
            config: Session configuration
        """
        self._logger.info("Session update (mock)", config=config)

    async def ping(self) -> bool:
        """Check connection health.

        Returns:
            True if healthy
        """
        return self._connected

    async def reconnect(self) -> None:
        """Reconnect to service."""
        await self.close()
        await asyncio.sleep(0.1)
        await self.connect()
