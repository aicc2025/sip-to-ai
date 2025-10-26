"""Audio adapter between SIP and AI services."""

import asyncio
from typing import Optional

import structlog

from app.utils.constants import AudioConstants
from app.utils.ring_buffer import StreamBuffer


class AudioAdapter:
    """Audio format adapter for SIP ↔ AI audio streaming.

    Simplified data flow (PCM16 passthrough):
    - PJSUA2 provides PCM16 @ 8kHz (onFrameReceived)
    - Pass PCM16 directly to AI (AI clients handle their own conversions)
    - AI returns PCM16 @ 8kHz (all AI clients normalize to this)
    - Pass through to PJSUA2 (onFrameRequested)
    """

    def __init__(
        self,
        uplink_capacity: int = 100,
        downlink_capacity: int = 200
    ) -> None:
        """Initialize audio adapter.

        Args:
            uplink_capacity: Uplink buffer capacity in frames
            downlink_capacity: Downlink buffer capacity in frames
        """
        # Stream buffers for PCM16 @ 8kHz passthrough
        self._uplink_stream = StreamBuffer(uplink_capacity)
        self._downlink_stream = StreamBuffer(downlink_capacity)

        # Accumulation buffer for downlink to avoid padding with zeros
        self._pending_bytes = b''

        # Stats
        self._frames_received = 0
        self._frames_sent = 0

        self._logger = structlog.get_logger(__name__)
        self._logger.info("AudioAdapter initialized (PCM16 passthrough mode)")

    def _log_periodic(self, counter: int, interval: int, message: str, **kwargs) -> None:
        """Log message periodically based on counter.

        Args:
            counter: Current counter value
            interval: Log every N counts
            message: Log message
            **kwargs: Additional structured log fields
        """
        if counter % interval == 0:
            self._logger.info(message, count=counter, **kwargs)

    @property
    def uplink_stream(self) -> StreamBuffer:
        """Get uplink stream (SIP -> AI)."""
        return self._uplink_stream

    @property
    def downlink_stream(self) -> StreamBuffer:
        """Get downlink stream (AI -> SIP)."""
        return self._downlink_stream

    def on_rx_pcm16_8k(self, pcm16_frame: bytes) -> None:
        """Handle received PCM16 frame from PJSUA2 (called from PJSUA2 thread).

        Args:
            pcm16_frame: 20ms PCM16 frame at 8kHz (320 bytes)
        """
        try:
            # Pass through PCM16 directly to uplink stream
            self._uplink_stream.send_nowait(pcm16_frame)

            self._frames_received += 1
            self._log_periodic(
                self._frames_received,
                AudioConstants.LOG_INTERVAL_FRAMES,
                "🎙️ Received frames from SIP"
            )

        except asyncio.QueueFull:
            # Buffer full, drop frame
            self._logger.debug("Uplink buffer full, dropping frame")
        except Exception as e:
            self._logger.error(f"Error processing RX frame: {e}")

    def get_tx_pcm16_8k_nowait(self) -> bytes:
        """Get next 20ms PCM16 frame for PJSUA2 output (non-blocking).

        This is a synchronous non-blocking method for use in PJSUA2 callbacks.

        Returns:
            20ms PCM16 frame at 8kHz (320 bytes), or silence if no data available
        """
        try:
            # Receive PCM16 frame from downlink stream (non-blocking)
            pcm16_frame = self._downlink_stream.receive_nowait()
            self._frames_sent += 1
            return pcm16_frame

        except asyncio.QueueEmpty:
            # No data available, return silence
            return AudioConstants.SILENCE_FRAME
        except Exception as e:
            self._logger.error(f"Error generating TX frame: {e}")
            # Return silence on error
            return AudioConstants.SILENCE_FRAME

    async def feed_ai_audio(self, audio_chunk: bytes) -> None:
        """Feed audio from AI to downlink with accumulation buffer.

        Accumulates variable-size chunks and splits into fixed 320-byte frames.
        Incomplete frames are kept in buffer until next chunk arrives.

        Args:
            audio_chunk: Audio chunk from AI (PCM16 @ 8kHz, variable size from AI clients)
        """
        try:
            # Append to pending buffer
            self._pending_bytes += audio_chunk

            # Split into complete frames
            offset = 0
            frames_sent = 0
            while offset + AudioConstants.PCM16_FRAME_SIZE <= len(self._pending_bytes):
                frame = self._pending_bytes[offset:offset + AudioConstants.PCM16_FRAME_SIZE]
                await self._downlink_stream.send(frame)
                offset += AudioConstants.PCM16_FRAME_SIZE
                frames_sent += 1

            # Keep incomplete part for next call (no padding)
            # This avoids inserting silence between chunks
            self._pending_bytes = self._pending_bytes[offset:]

            # Log frame processing
            if frames_sent > 0:
                self._logger.debug(
                    f"AI audio processed",
                    chunk_size=len(audio_chunk),
                    frames_sent=frames_sent,
                    pending=len(self._pending_bytes)
                )

        except Exception as e:
            self._logger.error(f"Error feeding AI audio: {e}")

    async def get_uplink_audio(self) -> bytes:
        """Get audio from uplink for AI (SIP → AI).

        Returns:
            PCM16 audio frame from SIP (320 bytes @ 8kHz)
        """
        return await self._uplink_stream.receive()

    async def get_downlink_audio(self) -> bytes:
        """Get audio from downlink for SIP (AI → SIP).

        Returns:
            PCM16 audio frame from AI (320 bytes @ 8kHz)
        """
        return await self._downlink_stream.receive()

    def get_stats(self) -> dict:
        """Get bridge statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "frames_received": self._frames_received,
            "frames_sent": self._frames_sent,
            "mode": "pcm16_passthrough"
        }

    async def close(self) -> None:
        """Close the audio adapter.

        Flushes any pending bytes by padding the final incomplete frame.
        """
        # Flush pending bytes if any
        if len(self._pending_bytes) > 0:
            if len(self._pending_bytes) < AudioConstants.PCM16_FRAME_SIZE:
                # Pad final incomplete frame with silence
                padding_size = AudioConstants.PCM16_FRAME_SIZE - len(self._pending_bytes)
                padded_frame = self._pending_bytes + b'\x00' * padding_size
                await self._downlink_stream.send(padded_frame)
                self._logger.debug(
                    "Flushed final incomplete frame",
                    original=len(self._pending_bytes),
                    padded_to=AudioConstants.PCM16_FRAME_SIZE
                )
            self._pending_bytes = b''

        await self._uplink_stream.close()
        await self._downlink_stream.close()
        self._logger.info("AudioAdapter closed", stats=self.get_stats())
