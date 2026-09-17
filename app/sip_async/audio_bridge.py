"""Audio bridge between RTP session and AudioAdapter.

Bridges RTP audio (G.711) with AudioAdapter (PCM16) using TaskGroup.
"""

import asyncio
from typing import TYPE_CHECKING

import structlog

from app.utils.constants import AudioConstants

if TYPE_CHECKING:
    from app.bridge.audio_adapter import AudioAdapter
    from app.sip_async.rtp_session import RTPSession

logger = structlog.get_logger(__name__)


class RTPAudioBridge:
    """Bridge RTPSession and AudioAdapter using TaskGroup.

    Data flow:
    - Uplink: RTP → decode G.711 → PCM16 → AudioAdapter → AI
    - Downlink: AI → AudioAdapter → PCM16 → encode G.711 → RTP

    When no RTP arrives from the caller (hold, stopped media), the uplink keeps
    feeding 20ms silence frames so the AI connection stays alive.
    """

    # Frame interval (seconds)
    FRAME_INTERVAL = AudioConstants.FRAME_MS / 1000.0
    # Gap without caller RTP before silence is generated (tolerates jitter)
    SILENCE_AFTER = 3 * FRAME_INTERVAL

    def __init__(self, rtp_session: 'RTPSession', audio_adapter: 'AudioAdapter'):
        """Initialize audio bridge.

        Args:
            rtp_session: RTP session for network audio
            audio_adapter: Audio adapter for AI integration
        """
        self.rtp = rtp_session
        self.adapter = audio_adapter
        self._running = False
        self._stopped = False
        self._tasks: list[asyncio.Task[None]] = []

        # Barge-in: clearing the adapter downlink also clears the RTP send queue
        self.adapter.add_downlink_clear_listener(self.rtp.clear_tx_queue)

        # Statistics
        self._uplink_frames = 0
        self._downlink_frames = 0
        self._uplink_silence_frames = 0

    async def run(self) -> None:
        """Run bidirectional audio bridge with TaskGroup."""
        if self._stopped:
            return

        self._running = True

        logger.info("AudioBridge starting")

        try:
            async with asyncio.TaskGroup() as tg:
                # Uplink: RTP → AudioAdapter
                self._tasks.append(tg.create_task(
                    self._uplink_task(),
                    name="audiobridge-uplink"
                ))

                # Downlink: AudioAdapter → RTP
                self._tasks.append(tg.create_task(
                    self._downlink_task(),
                    name="audiobridge-downlink"
                ))

                logger.info("AudioBridge TaskGroup started")

        except* asyncio.CancelledError:
            # Normal cancellation during shutdown
            logger.debug("AudioBridge tasks cancelled (normal shutdown)")

        except* Exception as eg:
            # Unexpected exceptions
            logger.error(
                "AudioBridge TaskGroup exceptions",
                count=len(eg.exceptions)
            )
            for exc in eg.exceptions:
                logger.error(
                    f"Exception: {type(exc).__name__}: {exc}",
                    exc_info=exc
                )
        finally:
            self._running = False
            self._tasks.clear()
            logger.info(
                "AudioBridge stopped",
                uplink_frames=self._uplink_frames,
                uplink_silence_frames=self._uplink_silence_frames,
                downlink_frames=self._downlink_frames
            )

    async def _uplink_task(self) -> None:
        """Uplink: RTP → AudioAdapter → AI.

        Reads PCM16 audio from RTP and feeds to AudioAdapter. If no RTP
        arrives for SILENCE_AFTER seconds, a 20ms silence frame is fed every
        frame interval until RTP resumes, so the AI keeps receiving audio.

        Note:
            Runs continuously until cancelled.
        """
        logger.info("AudioBridge uplink task started")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.SILENCE_AFTER
        in_silence = False

        try:
            while True:
                pcm_data = await self.rtp.receive_frame(
                    timeout=max(0.0, deadline - loop.time())
                )

                if pcm_data is None:
                    # No caller RTP in time: keep the AI fed with silence
                    if not in_silence:
                        in_silence = True
                        logger.info("No caller RTP - feeding silence to AI")
                    pcm_data = AudioConstants.SILENCE_FRAME
                    self._uplink_silence_frames += 1
                    deadline += self.FRAME_INTERVAL
                    # Resync if the loop fell behind (avoid a burst of silence)
                    if deadline < loop.time():
                        deadline = loop.time() + self.FRAME_INTERVAL
                else:
                    if in_silence:
                        in_silence = False
                        logger.info(
                            "Caller RTP resumed",
                            silence_frames=self._uplink_silence_frames
                        )
                    deadline = loop.time() + self.SILENCE_AFTER

                # Feed PCM16 @ 8kHz to AudioAdapter
                # AudioAdapter expects 320 bytes (160 samples * 2 bytes)
                self.adapter.on_rx_pcm16_8k(pcm_data)

                self._uplink_frames += 1

                # Periodic logging
                if self._uplink_frames % 500 == 0:
                    logger.debug(
                        "AudioBridge uplink stats",
                        frames=self._uplink_frames,
                        direction="RTP → AudioAdapter → AI"
                    )

        except asyncio.CancelledError:
            logger.info("AudioBridge uplink cancelled")
            raise
        except Exception as e:
            logger.error("AudioBridge uplink error", error=str(e), exc_info=True)
            raise
        finally:
            logger.info(
                "AudioBridge uplink stopped",
                frames_processed=self._uplink_frames
            )

    async def _downlink_task(self) -> None:
        """Downlink: AI → AudioAdapter → RTP.

        Gets PCM16 audio from AudioAdapter and sends to RTP.

        Note:
            Runs continuously until cancelled. Does not check _running flag
            to avoid race conditions during startup.
        """
        logger.info("AudioBridge downlink task started")

        try:
            while True:
                # Get PCM16 audio from AudioAdapter downlink (AI → SIP)
                # This blocks until data is available
                pcm_data = await self.adapter.get_downlink_audio()

                # Send to RTP
                await self.rtp.send_audio(pcm_data)

                self._downlink_frames += 1

                # Periodic logging
                if self._downlink_frames % 500 == 0:
                    logger.debug(
                        "AudioBridge downlink stats",
                        frames=self._downlink_frames,
                        direction="AI → AudioAdapter → RTP"
                    )

        except asyncio.CancelledError:
            logger.info("AudioBridge downlink cancelled")
            raise
        except Exception as e:
            logger.error("AudioBridge downlink error", error=str(e), exc_info=True)
            raise
        finally:
            logger.info(
                "AudioBridge downlink stopped",
                frames_sent=self._downlink_frames
            )

    async def stop(self) -> None:
        """Stop the audio bridge and wait for its tasks to finish."""
        self._running = False
        self._stopped = True
        logger.info("AudioBridge stop requested")

        current = asyncio.current_task()
        tasks = [t for t in self._tasks if not t.done() and t is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
