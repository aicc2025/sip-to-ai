"""Audio adapter between SIP and AI services."""

import asyncio
from typing import Callable, Optional

import structlog

from app.core.constants import AudioConstants
from app.core.ring_buffer import StreamBuffer


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
        """Get audio from uplink for AI.

        Returns:
            Audio frame for AI (G.711 or PCM16 depending on config)
        """
        return await self._uplink_stream.receive()

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


class CallSession:
    """Manages call session lifecycle: AI connection + audio transport tasks.

    Coordinates AudioAdapter and AI client for a single call.
    """

    def __init__(
        self,
        audio_adapter: AudioAdapter,
        ai_client: any
    ) -> None:
        """Initialize call session.

        Args:
            audio_adapter: AudioAdapter instance
            ai_client: AI duplex client
        """
        self._media = audio_adapter
        self._ai = ai_client

        self._running = False
        self._task_group_task: Optional[asyncio.Task[None]] = None

        self._logger = structlog.get_logger(__name__)

    async def start(self) -> None:
        """Start the call session using asyncio.TaskGroup.

        This starts background tasks and returns immediately without blocking.
        Call stop() to terminate the session.
        """
        if self._running:
            return

        self._running = True

        try:
            # Connect AI client
            await self._ai.connect()
            self._logger.info("AI client connected")

            # Define session runner with TaskGroup
            async def _run_session() -> None:
                self._logger.info("🏁 Session runner starting with TaskGroup...")
                try:
                    async with asyncio.TaskGroup() as tg:
                        self._logger.info("📋 Starting uplink task...")
                        tg.create_task(self._uplink_safe())

                        self._logger.info("📋 Starting AI receive task...")
                        tg.create_task(self._ai_recv_safe())

                        self._logger.info("📋 Starting health task...")
                        tg.create_task(self._health_safe())

                        self._logger.info("✅ All call session tasks started")
                    # TaskGroup exits when all tasks complete or on exception
                    self._logger.info("TaskGroup exited")
                except* Exception as eg:
                    self._logger.error(f"TaskGroup exceptions: {eg.exceptions}")
                    for e in eg.exceptions:
                        self._logger.error(f"  - {type(e).__name__}: {e}")
                finally:
                    self._running = False
                    self._logger.info("Session runner finished")

            # Create background task and store reference
            self._task_group_task = asyncio.create_task(_run_session())
            self._logger.info("Call session started - TaskGroup launched")

        except Exception as e:
            self._logger.error(f"Failed to start session: {e}", exc_info=True)
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop the call session."""
        if not self._running:
            return

        self._logger.info("Stopping call session...")

        self._running = False

        # Cancel the TaskGroup background task
        if self._task_group_task:
            self._task_group_task.cancel()
            try:
                await self._task_group_task
            except asyncio.CancelledError:
                self._logger.info("TaskGroup task cancelled")
            except Exception as e:
                self._logger.error(f"Error during TaskGroup cancellation: {e}")

        # Disconnect AI
        try:
            await self._ai.close()
            self._logger.info("AI client closed")
        except Exception as e:
            self._logger.error(f"Error closing AI client: {e}")

        # Close audio adapter
        try:
            await self._media.close()
            self._logger.info("Audio adapter closed")
        except Exception as e:
            self._logger.error(f"Error closing audio adapter: {e}")

        self._logger.info("Call session stopped")

    async def _uplink_safe(self) -> None:
        """Safe uplink with proper exception handling and cleanup."""
        frames_processed = 0
        self._logger.info("🚀 Uplink task STARTED")
        try:
            while self._running:
                try:
                    # Timeout per frame to prevent hang
                    async with asyncio.timeout(0.05):
                        frame = await self._media.get_uplink_audio()
                        await self._ai.send_pcm16_8k(frame)

                    frames_processed += 1
                    if frames_processed == 1:
                        self._logger.info("🔊 First frame sent to AI!")

                    if frames_processed % AudioConstants.LOG_INTERVAL_FRAMES == 0:
                        self._logger.info(
                            "🔊 Uplink processed frames",
                            count=frames_processed,
                            direction="SIP → AI"
                        )

                except TimeoutError:
                    # No data available, continue
                    await asyncio.sleep(0.01)
                except Exception as e:
                    self._logger.error(f"Uplink frame error: {e}", exc_info=True)
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            self._logger.info(f"Uplink task cancelled after {frames_processed} frames")
            raise  # Propagate cancellation
        except Exception as e:
            self._logger.error(f"Uplink task fatal error: {e}", exc_info=True)
        finally:
            self._logger.info(f"🛑 Uplink task STOPPED (processed {frames_processed} frames)")

    async def _ai_recv_safe(self) -> None:
        """Safe AI receive with proper exception handling and cleanup."""
        chunks_received = 0
        self._logger.info("🎧 AI receive task STARTED")
        try:
            async for chunk in self._ai.receive_chunks():
                if not self._running:
                    break

                # Direct passthrough to downlink stream
                await self._media.feed_ai_audio(chunk)

                chunks_received += 1
                if chunks_received % AudioConstants.LOG_INTERVAL_FRAMES == 0:
                    self._logger.info("📢 Received chunks from AI", count=chunks_received)

        except asyncio.CancelledError:
            self._logger.info(f"AI receive task cancelled after {chunks_received} chunks")
            raise  # Propagate cancellation
        except Exception as e:
            self._logger.error(f"AI receive fatal error: {e}", exc_info=True)
        finally:
            self._logger.info(f"🛑 AI receive task STOPPED (received {chunks_received} chunks)")

    async def _health_safe(self) -> None:
        """Safe health monitoring with proper exception handling."""
        reconnect_attempts = 0
        max_attempts = 3
        health_checks = 0

        self._logger.info("🏥 Health task STARTED")
        try:
            while self._running:
                try:
                    await asyncio.sleep(30)  # Health check interval
                    health_checks += 1

                    # Check AI connection
                    if not await self._ai.ping():
                        self._logger.warning(f"AI connection unhealthy (check #{health_checks})")

                        if reconnect_attempts < max_attempts:
                            self._logger.info(f"Attempting reconnect ({reconnect_attempts + 1}/{max_attempts})")
                            await self._ai.reconnect()
                            reconnect_attempts += 1
                        else:
                            self._logger.error("Max reconnection attempts reached, stopping session")
                            await self.stop()
                            break
                    else:
                        reconnect_attempts = 0  # Reset on success

                except asyncio.CancelledError:
                    raise  # Propagate cancellation
                except Exception as e:
                    self._logger.error(f"Health check error: {e}", exc_info=True)
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            self._logger.info(f"Health task cancelled after {health_checks} checks")
            raise  # Propagate cancellation
        except Exception as e:
            self._logger.error(f"Health task fatal error: {e}", exc_info=True)
        finally:
            self._logger.info(f"🛑 Health task STOPPED (performed {health_checks} checks)")