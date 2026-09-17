"""Call session lifecycle management: AI connection + audio transport."""

import asyncio
import time
from typing import Optional

import structlog

from app.ai.duplex_base import AiEventType, describe_error, is_barge_in_event
from app.bridge.audio_adapter import AudioAdapter
from app.utils.constants import AudioConstants


class AiStreamEnded(Exception):
    """Raised when the AI audio stream ends while the session is running."""


class AiHealthCheckFailed(Exception):
    """Raised when the AI connection is unresponsive; ends the session (and call)."""


class CallSession:
    """Manages call session lifecycle: AI connection + audio transport tasks.

    Coordinates AudioAdapter and AI client for a single call.
    """

    # Health check interval (seconds)
    HEALTH_CHECK_INTERVAL = 30.0
    # Pings attempted before the connection is declared dead
    HEALTH_PING_ATTEMPTS = 2

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
        self._stopped = False
        self._task_group_task: Optional[asyncio.Task[None]] = None
        # Set once the session has ended (runner finished, stop() or start failure)
        self._ended_event = asyncio.Event()

        self._logger = structlog.get_logger(__name__)

    async def start(self) -> None:
        """Start the call session using asyncio.TaskGroup.

        This starts background tasks and returns immediately without blocking.
        Call stop() to terminate the session.
        """
        if self._running or self._stopped:
            return

        self._running = True
        self._ended_event.clear()

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
                        tg.create_task(
                            self._uplink_safe(),
                            name="session-uplink"
                        )

                        self._logger.info("📋 Starting AI receive task...")
                        tg.create_task(
                            self._ai_recv_safe(),
                            name="session-ai-recv"
                        )

                        self._logger.info("📋 Starting AI events task...")
                        tg.create_task(
                            self._events_safe(),
                            name="session-ai-events"
                        )

                        self._logger.info("📋 Starting health task...")
                        tg.create_task(
                            self._health_safe(),
                            name="session-health"
                        )

                        self._logger.info("✅ All call session tasks started")
                    # TaskGroup exits when all tasks complete or on exception
                    self._logger.info("TaskGroup exited")

                except* asyncio.CancelledError:
                    # Normal cancellation during shutdown
                    self._logger.debug("Call session tasks cancelled (normal shutdown)")

                except* AiStreamEnded:
                    self._logger.warning("AI connection closed - ending call session")

                except* AiHealthCheckFailed:
                    self._logger.error("AI connection health check failed - ending call session")

                except* Exception as eg:
                    # Unexpected exceptions
                    self._logger.error(
                        "CallSession TaskGroup exceptions",
                        count=len(eg.exceptions)
                    )
                    for exc in eg.exceptions:
                        self._logger.error(
                            f"Exception: {type(exc).__name__}: {exc}",
                            exc_info=exc
                        )

                finally:
                    self._running = False
                    self._ended_event.set()
                    self._logger.info("Session runner finished")

            # Create background task and store reference
            self._task_group_task = asyncio.create_task(
                _run_session(),
                name="call-session-runner"
            )
            self._logger.info("Call session started - TaskGroup launched")

        except BaseException as e:
            if not isinstance(e, asyncio.CancelledError):
                # Connection failures are expected operational errors: no traceback
                expected = isinstance(e, (ConnectionError, TimeoutError))
                self._logger.error(f"Failed to start session: {describe_error(e)}", exc_info=not expected)
            self._running = False
            self._ended_event.set()
            raise

    async def wait_ended(self) -> None:
        """Wait until the session has ended (AI disconnect, failure or stop())."""
        await self._ended_event.wait()

    async def stop(self) -> None:
        """Stop the call session.

        Idempotent. Also releases the AI client and audio adapter when the
        session runner already finished on its own (e.g. AI disconnect).
        """
        if self._stopped:
            return

        self._stopped = True
        self._logger.info("Stopping call session...")

        self._running = False

        # Cancel the TaskGroup background task
        if self._task_group_task and not self._task_group_task.done():
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

        self._ended_event.set()
        self._logger.info("Call session stopped")

    async def _uplink_safe(self) -> None:
        """Safe uplink with proper exception handling and cleanup."""
        frames_processed = 0
        frame_errors = 0
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
                except ConnectionError as e:
                    # AI connection is gone: stop sending (the receive task ends the session)
                    self._logger.warning(
                        "AI connection lost - stopping uplink",
                        error=describe_error(e),
                        frames_sent=frames_processed
                    )
                    return
                except Exception as e:
                    frame_errors += 1
                    if not getattr(self._ai, "is_connected", True):
                        self._logger.warning(
                            "AI connection lost - stopping uplink",
                            error=describe_error(e),
                            frames_sent=frames_processed
                        )
                        return
                    # Log the first error with traceback, then only a periodic count
                    if frame_errors == 1:
                        self._logger.error(f"Uplink frame error: {describe_error(e)}", exc_info=True)
                    elif frame_errors % 500 == 0:
                        self._logger.error("Uplink frame errors continue", errors=frame_errors, last_error=describe_error(e))
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

        if self._running:
            # The AI audio stream only ends when the AI connection is gone
            raise AiStreamEnded()

    async def _events_safe(self) -> None:
        """Consume AI events: barge-in handling and logging.

        On a barge-in event (caller speech started) all queued downlink audio is
        dropped so the caller's speech interrupts the AI immediately.
        """
        events_received = 0
        self._logger.info("📨 AI events task STARTED")
        try:
            async for event in self._ai.events():
                events_received += 1

                if is_barge_in_event(event):
                    self._handle_barge_in()
                elif event.type == AiEventType.ERROR:
                    self._logger.warning("AI error event", error=event.error or event.data)
                elif event.type == AiEventType.DISCONNECTED:
                    self._logger.info("AI disconnected event received")

        except asyncio.CancelledError:
            self._logger.info(f"AI events task cancelled after {events_received} events")
            raise  # Propagate cancellation
        except Exception as e:
            self._logger.error(f"AI events fatal error: {e}", exc_info=True)
        finally:
            self._logger.info(f"🛑 AI events task STOPPED (received {events_received} events)")

    def _handle_barge_in(self) -> None:
        """Drop queued AI audio (client queue, adapter downlink, transport)."""
        ai_chunks = 0
        clear_audio_queue = getattr(self._ai, "clear_audio_queue", None)
        if callable(clear_audio_queue):
            ai_chunks = clear_audio_queue()

        downlink_frames = self._media.clear_downlink()

        self._logger.info(
            "Barge-in: caller speech started, cleared queued AI audio",
            ai_chunks_dropped=ai_chunks,
            downlink_frames_dropped=downlink_frames
        )

    async def _health_safe(self) -> None:
        """Monitor the AI connection and end the session if it is dead.

        A connection that delivered any message during the last interval is
        healthy without a ping (the WebSocket reader never blocks on audio, so
        received messages prove the connection works). Otherwise a ping is
        sent; if HEALTH_PING_ATTEMPTS pings fail the session ends with
        AiHealthCheckFailed, which ends the call (BYE). There is no silent
        reconnect: a new AI session would lose the conversation and the
        running audio streams.
        """
        health_checks = 0

        self._logger.info("🏥 Health task STARTED")
        try:
            while self._running:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                health_checks += 1

                last_receive = getattr(self._ai, "last_receive_time", 0.0) or 0.0
                if last_receive and time.monotonic() - last_receive < self.HEALTH_CHECK_INTERVAL:
                    continue

                healthy = False
                for attempt in range(1, self.HEALTH_PING_ATTEMPTS + 1):
                    try:
                        healthy = await self._ai.ping()
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self._logger.warning("AI health ping error", error=describe_error(e), attempt=attempt)
                        healthy = False
                    if healthy:
                        break
                    self._logger.warning(
                        "AI health ping failed",
                        check=health_checks,
                        attempt=attempt,
                        max_attempts=self.HEALTH_PING_ATTEMPTS
                    )

                if not healthy:
                    raise AiHealthCheckFailed()

        except asyncio.CancelledError:
            self._logger.info(f"Health task cancelled after {health_checks} checks")
            raise  # Propagate cancellation
        finally:
            self._logger.info(f"🛑 Health task STOPPED (performed {health_checks} checks)")
