"""PJSUA2 endpoint wrapper for SIP functionality.

Note: This module requires pjsua2 to be installed manually.

Installation instructions:
1. From local PJPROJECT build (Recommended):
   cd /path/to/pjproject/pjsip-apps/src/swig/python
   uv pip install .

2. For macOS with Homebrew:
   brew install pjsip
   uv pip install pjsua2

3. For Ubuntu/Debian:
   apt-get install python3-pjsua2

Without pjsua2, the application will fail to start (fail-fast design).
"""

import asyncio
import sys
from typing import Callable, Optional

import structlog

# Import for creating per-call resources
from app.config import config
from app.core.constants import AudioConstants

try:
    import pjsua2 as pj
except ImportError as e:
    print("=" * 80, file=sys.stderr)
    print("ERROR: pjsua2 is not installed!", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("\nPJSUA2 is required for SIP functionality.", file=sys.stderr)
    print("\nInstallation instructions:", file=sys.stderr)
    print("  1. From local PJPROJECT build (Recommended):", file=sys.stderr)
    print("     cd /path/to/pjproject/pjsip-apps/src/swig/python", file=sys.stderr)
    print("     uv pip install .", file=sys.stderr)
    print("\n  2. For macOS with Homebrew:", file=sys.stderr)
    print("     brew install pjsip && uv pip install pjsua2", file=sys.stderr)
    print("\n  3. For Ubuntu/Debian:", file=sys.stderr)
    print("     sudo apt-get install python3-pjsua2", file=sys.stderr)
    print("\nSee README.md for detailed installation instructions.", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    sys.exit(1)


class PJSIPEndpoint:
    """PJSUA2 endpoint wrapper for userless account (receive-only mode)."""

    def __init__(
        self,
        domain: str = "localhost",
        transport_type: str = "udp",
        port: int = 6060,
        event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """Initialize PJSIP endpoint with userless account.

        Userless account mode: Only listens for incoming SIP INVITE requests.
        No registration to SIP server required.

        Args:
            domain: SIP domain or IP address for URI
            transport_type: Transport type (udp/tcp)
            port: SIP listening port
            event_loop: Asyncio event loop for run_coroutine_threadsafe calls
        """
        self._domain = domain
        self._transport_type = transport_type
        self._port = port
        self._event_loop = event_loop

        self._endpoint: Optional["pj.Endpoint"] = None
        self._account: Optional["pj.Account"] = None
        self._transport: Optional["pj.Transport"] = None

        self._logger = structlog.get_logger(__name__)

    def initialize(self) -> None:
        """Initialize PJSIP endpoint."""

        try:
            # Create endpoint
            self._endpoint = pj.Endpoint()
            self._endpoint.libCreate()

            # Configure endpoint
            ep_cfg = pj.EpConfig()

            # UA config
            ep_cfg.uaConfig.threadCnt = 1  # Single worker thread for SIP processing
            ep_cfg.uaConfig.userAgent = "SIP-to-AI/1.0"

            # Logging config
            ep_cfg.logConfig.level = 5  # Max debug level to see SIP messages
            ep_cfg.logConfig.consoleLevel = 5  # Show all details in console
            ep_cfg.logConfig.msgLogging = 1  # Enable SIP message logging

            # Media config - disable SRTP
            ep_cfg.medConfig.srtpSecureSignaling = 0

            self._endpoint.libInit(ep_cfg)

            # Set null audio device on Linux (we use custom MediaPort)
            if sys.platform.startswith('linux'):
                try:
                    self._endpoint.audDevManager().setNullDev()
                    self._logger.info("Set null audio device for Linux")
                except Exception as e:
                    self._logger.warning(f"Could not set null audio device: {e}")

            # Create transport
            self._create_transport()

            # Start endpoint
            self._endpoint.libStart()

            # Set codec priorities - prefer PCMU/PCMA
            try:
                self._endpoint.codecSetPriority("PCMU/8000", 255)  # G.711 μ-law
                self._endpoint.codecSetPriority("PCMA/8000", 254)  # G.711 A-law
                self._logger.info("Codec priorities set: PCMU (255), PCMA (254)")
            except Exception as e:
                self._logger.warning(f"Could not set codec priorities: {e}")

            # Create account
            self._create_account()

            self._logger.info(
                "PJSIP endpoint initialized (userless account)",
                domain=self._domain,
                transport=self._transport_type,
                port=self._port
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize PJSIP: {e}")
            raise

    def _create_transport(self) -> None:
        """Create SIP transport."""
        if not self._endpoint:
            return

        transport_cfg = pj.TransportConfig()
        transport_cfg.port = self._port
        transport_cfg.boundAddress = self._domain  # Always set bound address

        self._logger.info(
            "Creating SIP transport",
            type=self._transport_type,
            port=self._port,
            bind_address=self._domain
        )

        try:
            if self._transport_type == "udp":
                try:
                    transport_id = self._endpoint.transportCreate(
                        pj.PJSIP_TRANSPORT_UDP,
                        transport_cfg
                    )
                    self._logger.info(f"UDP transport created on {self._domain}:{self._port}, ID={transport_id}")
                except pj.Error as e:
                    self._logger.error(f"Failed to bind to {self._domain}:{self._port}: {e}")
                    # Retry with 0.0.0.0 if specific IP binding fails
                    if self._domain != "0.0.0.0":
                        self._logger.info("Retrying with 0.0.0.0...")
                        transport_cfg.boundAddress = "0.0.0.0"
                        transport_id = self._endpoint.transportCreate(
                            pj.PJSIP_TRANSPORT_UDP,
                            transport_cfg
                        )
                        self._logger.info(f"UDP transport created on 0.0.0.0:{self._port}, ID={transport_id}")
                    else:
                        raise
            elif self._transport_type == "tcp":
                transport_id = self._endpoint.transportCreate(
                    pj.PJSIP_TRANSPORT_TCP,
                    transport_cfg
                )
            else:
                raise ValueError(f"Unsupported transport type: {self._transport_type}")

            # Store transport ID (it's an integer, not an object)
            self._transport = transport_id

            self._logger.info(
                "SIP transport created successfully",
                type=self._transport_type,
                port=self._port,
                transport_id=transport_id
            )
        except Exception as e:
            self._logger.error(f"Failed to create transport: {e}")
            raise

    def _create_account(self) -> None:
        """Create userless SIP account (receive-only mode).

        Userless account does not register to any SIP server.
        It only listens for incoming SIP INVITE requests.
        """
        if not self._endpoint:
            return

        # Configure userless account - no registration
        acc_cfg = pj.AccountConfig()

        # Set local SIP URI
        acc_cfg.idUri = f"sip:uas@{self._domain}:{self._port}"

        # Disable registration - we're only a UAS (User Agent Server)
        acc_cfg.regConfig.registrarUri = ""  # Empty string disables registration
        acc_cfg.regConfig.registerOnAdd = False

        # Configure NAT settings
        acc_cfg.natConfig.iceEnabled = False  # Disable ICE
        acc_cfg.natConfig.turnEnabled = False  # Disable TURN

        # Configure call settings
        acc_cfg.callConfig.prackUse = pj.PJSUA_100REL_NOT_USED
        acc_cfg.callConfig.timerUse = pj.PJSUA_SIP_TIMER_OPTIONAL

        # Disable SRTP completely - use plain RTP only
        acc_cfg.mediaConfig.srtpUse = pj.PJMEDIA_SRTP_DISABLED
        acc_cfg.mediaConfig.srtpSecureSignaling = 0
        acc_cfg.mediaConfig.srtpOptionalInAnswer = False

        # Disable ICE/TURN in media config
        acc_cfg.mediaConfig.enableIce = False
        acc_cfg.mediaConfig.enableTurn = False

        # Create account
        self._account = PJSIPAccount(self)
        self._account.create(acc_cfg)

        self._logger.info(
            "Userless SIP account created (receive-only)",
            uri=acc_cfg.idUri
        )

    def shutdown(self) -> None:
        """Shutdown PJSIP endpoint."""
        try:
            if self._account:
                self._account.shutdown()

            if self._endpoint:
                self._endpoint.libDestroy()

            self._logger.info("PJSIP endpoint shutdown")

        except Exception as e:
            self._logger.error(f"Error during PJSIP shutdown: {e}")


class PJSIPAccount(pj.Account):
    """PJSIP account wrapper."""

    def __init__(self, endpoint: PJSIPEndpoint) -> None:
        """Initialize account.

        Args:
            endpoint: Parent endpoint
        """
        super().__init__()
        self._endpoint = endpoint
        self._logger = structlog.get_logger(__name__)
        # Keep references to active calls to prevent garbage collection
        self.active_calls: dict[int, "PJSIPCall"] = {}

    def onRegState(self, prm: "pj.OnRegStateParam") -> None:
        """Handle registration state change.

        Args:
            prm: Registration parameters
        """
        info = self.getInfo()
        self._logger.info(
            f"Registration state changed",
            status=info.regStatus,
            reason=info.regStatusText
        )

    def onIncomingCall(self, prm: "pj.OnIncomingCallParam") -> None:
        """Handle incoming call.

        Creates independent resources for each call:
        - MediaBridge for audio codec conversion
        - AI client (OpenAI/Deepgram) with its own WebSocket
        - CallBridge to manage the audio flow

        Args:
            prm: Incoming call parameters
        """
        self._logger.info(f"Incoming SIP INVITE received! Call ID: {prm.callId}")

        try:
            # Create call object immediately (lightweight)
            call = PJSIPCall(
                account=self,
                call_id=prm.callId,
                audio_adapter=None,  # Will be set in async task
                session=None  # Will be set in async task
            )

            # Store call reference to prevent garbage collection (use string key)
            self.active_calls[str(prm.callId)] = call
            call.account_ref = self  # Back reference for cleanup
            self._logger.info(f"Stored call reference, active calls: {len(self.active_calls)}")

            # Get call info
            call_info = call.getInfo()
            self._logger.info(
                "Call details",
                call_id=call_info.callIdString,
                from_uri=call_info.remoteUri,
                to_uri=call_info.localUri
            )

            # Schedule async resource creation and 200 OK
            loop = self._endpoint._event_loop

            async def setup_call_resources():
                """Create resources asynchronously and send 200 OK."""
                try:
                    # Import here to avoid circular dependency
                    from app.sip.audio_adapter import AudioAdapter, CallSession

                    # Create AudioAdapter (PCM16 passthrough mode)
                    # All AI clients handle their own format conversions
                    audio_adapter = AudioAdapter(
                        uplink_capacity=config.audio.uplink_buf_frames,
                        downlink_capacity=config.audio.downlink_buf_frames
                    )

                    # Create AI client using factory from main.py
                    from app.main import create_ai_client
                    ai_client = create_ai_client()

                    # Create CallSession
                    session = CallSession(
                        audio_adapter=audio_adapter,
                        ai_client=ai_client
                    )

                    # Set resources on call object
                    call._audio_adapter = audio_adapter
                    call._session = session

                    self._logger.info("Call resources created asynchronously")

                    # Now send 200 OK
                    await asyncio.sleep(0.1)  # Small delay
                    call_param_200 = pj.CallOpParam()
                    call_param_200.statusCode = 200
                    call_param_200.reason = "OK"
                    call_param_200.opt.audioCount = 1
                    call_param_200.opt.videoCount = 0
                    call.answer(call_param_200)
                    self._logger.info("Sent 200 OK")

                except Exception as e:
                    self._logger.error(f"Error in async resource setup: {e}", exc_info=True)
                    # Try to decline the call
                    try:
                        call_param_err = pj.CallOpParam()
                        call_param_err.statusCode = 500
                        call_param_err.reason = "Internal Server Error"
                        call.answer(call_param_err)
                    except:
                        pass

            # Schedule resource setup in background
            if loop:
                asyncio.run_coroutine_threadsafe(setup_call_resources(), loop)

        except Exception as e:
            self._logger.error(f"Failed to handle incoming call: {e}", exc_info=True)
            # Remove from active calls on failure (use string key)
            self.active_calls.pop(str(prm.callId), None)


class PJSIPCall(pj.Call):
    """PJSIP call wrapper."""

    def __init__(
        self,
        account: PJSIPAccount,
        call_id: int = -1,
        audio_adapter: Optional[any] = None,
        session: Optional[any] = None
    ) -> None:
        """Initialize call.

        Args:
            account: Parent account
            call_id: PJSUA call ID (use -1 or pj.PJSUA_INVALID_ID for new calls)
            audio_adapter: AudioAdapter instance for this call
            session: CallSession instance for this call
        """
        # Pass call_id to parent Call constructor
        super().__init__(account, call_id)
        self._account = account
        self._audio_adapter = audio_adapter
        self._session = session
        self._media: Optional["PJSIPMediaPort"] = None
        self._audio_task: Optional[any] = None
        self._session_started = False  # Track if session was already started
        self._logger = structlog.get_logger(__name__)


    def onCallState(self, prm: "pj.OnCallStateParam") -> None:
        """Handle call state change.

        When call is confirmed, start the bridge exactly once.
        When call is disconnected, stop the bridge and clean up resources.

        Args:
            prm: Call state parameters
        """
        info = self.getInfo()
        self._logger.info(
            "Call state changed",
            state=info.state,
            state_text=info.stateText,
            last_status=info.lastStatusCode,
            last_reason=info.lastReason,
            call_id=info.callIdString,
            role=info.role
        )

        # Log specific state transitions
        if info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self._logger.warning(
                "⚠️  DISCONNECTED event received",
                reason=info.lastReason,
                status=info.lastStatusCode,
                call_id=info.callIdString
            )

        # Start session when call is confirmed (exactly once at signal level)
        if info.state == pj.PJSIP_INV_STATE_CONFIRMED:
            if not self._session_started and self._session:
                self._logger.info("Call confirmed - starting CallSession")
                try:
                    # Get event loop from endpoint
                    loop = self._account._endpoint._event_loop

                    # Start session in async context (connects AI WebSocket)
                    # Use run_coroutine_threadsafe for non-blocking execution
                    asyncio.run_coroutine_threadsafe(self._session.start(), loop)
                    self._logger.info("CallSession start scheduled")

                    # Start audio processing task
                    asyncio.run_coroutine_threadsafe(self._start_audio_processing(), loop)
                    self._logger.info("Audio processing scheduled")

                    self._session_started = True

                except Exception as e:
                    self._logger.error(f"Failed to start session: {e}", exc_info=True)

        elif info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self._logger.info("Call disconnected", reason=info.lastReason)

            # CRITICAL: Use asyncio.run_coroutine_threadsafe() for non-blocking cleanup
            # This returns immediately without blocking PJSUA2 thread

            # Mark session as stopped
            self._session_started = False

            # Store references for cleanup
            session = self._session
            adapter = self._audio_adapter

            # Clear instance references immediately
            self._session = None
            self._media = None
            self._audio_task = None

            # Schedule async cleanup using run_coroutine_threadsafe (non-blocking)
            if session or adapter:
                loop = self._account._endpoint._event_loop

                async def _cleanup():
                    """Async cleanup task."""
                    try:
                        if session:
                            await session.stop()
                            self._logger.info("Session stopped")
                    except Exception as e:
                        self._logger.error(f"Error stopping session: {e}")

                    try:
                        if adapter:
                            await adapter.close()
                            self._logger.info("AudioAdapter closed")
                    except Exception as e:
                        self._logger.error(f"Error closing adapter: {e}")

                # Schedule cleanup in event loop (returns immediately)
                if loop:
                    asyncio.run_coroutine_threadsafe(_cleanup(), loop)

            # Remove from active calls (use string key to match onIncomingCall)
            call_id_str = str(info.callIdString)
            if hasattr(self, 'account_ref') and self.account_ref:
                if call_id_str in self.account_ref.active_calls:
                    del self.account_ref.active_calls[call_id_str]
                    self._logger.info(f"Removed call {call_id_str} from active calls")

            self._logger.info("Call cleanup scheduled - returning to PJSUA2")

    def onCallMediaState(self, prm: "pj.OnCallMediaStateParam") -> None:
        """Handle media state change.

        Creates media handler when audio becomes active.
        Note: Session start is handled in onCallState(CONFIRMED) to avoid
        duplicate triggers from re-INVITE, hold/unhold, etc.

        Args:
            prm: Media state parameters
        """
        info = self.getInfo()

        for media in info.media:
            if media.type == pj.PJMEDIA_TYPE_AUDIO:
                if media.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                    self._logger.info("Audio media active")

    async def _start_audio_processing(self) -> None:
        """Start audio processing loop for this call.

        This task handles bidirectional audio flow:
        - PJSIP → AudioAdapter → AI (uplink handled by CallSession)
        - AI → AudioAdapter → PJSIP (downlink handled by CallSession)

        Creates PJSIPMediaPort to connect PJSUA2's AudioMedia to AudioAdapter.
        """
        self._logger.info("Starting audio processing loop")

        if not self._audio_adapter:
            self._logger.warning("AudioAdapter not available, skipping audio processing")
            return

        media_port = None
        call_media = None

        try:
            # Create PJSIPMediaPort to bridge PJSUA2 and AudioAdapter
            media_port = PJSIPMediaPort(
                audio_adapter=self._audio_adapter,
                sample_rate=8000,
                frame_ms=20
            )
            self._logger.info("Created PJSIPMediaPort for audio integration")

            # Create MediaFormatAudio for the port
            fmt = pj.MediaFormatAudio()
            fmt.type = pj.PJMEDIA_TYPE_AUDIO
            fmt.clockRate = 8000
            fmt.channelCount = 1
            fmt.bitsPerSample = 16
            fmt.frameTimeUsec = 20000  # 20ms frame

            # Create the audio port
            media_port.createPort("sip_to_ai_port", fmt)
            self._logger.info("Created PJSUA2 audio port with MediaFormatAudio")

            # Get call's audio media and connect to our port
            # Note: getMedia(-1) gets the first active media
            call_info = self.getInfo()
            for i, media_info in enumerate(call_info.media):
                if media_info.type == pj.PJMEDIA_TYPE_AUDIO and media_info.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                    # Get AudioMedia from the call
                    call_media = pj.AudioMedia.typecastFromMedia(self.getMedia(media_info.index))

                    # Connect call's audio to our media port (RX path: SIP -> AudioAdapter)
                    call_media.startTransmit(media_port)

                    # Connect our media port to call's audio (TX path: AudioAdapter -> SIP)
                    media_port.startTransmit(call_media)

                    self._logger.info(
                        "Connected AudioMedia bidirectionally",
                        media_index=media_info.index,
                        rx_path="SIP->MediaPort->AudioAdapter",
                        tx_path="AudioAdapter->MediaPort->SIP"
                    )
                    break

            # Keep running while session is active
            # Use _session_started flag instead of polling getInfo() which can fail after disconnect
            while self._session_started:
                await asyncio.sleep(0.5)

            self._logger.info("Audio processing loop ended")

        except asyncio.CancelledError:
            self._logger.info("Audio processing cancelled")
        except Exception as e:
            self._logger.error(f"Audio processing error: {e}", exc_info=True)
        finally:
            # Note: PJSUA2 automatically cleans up media connections when call disconnects
            # We don't need to (and shouldn't) manually call stopTransmit()
            self._logger.info("Audio processing cleanup complete")


class PJSIPMediaPort(pj.AudioMediaPort):
    """Custom AudioMediaPort for PJSUA2 integration with AudioAdapter.

    This class bridges PJSUA2's AudioMedia system with our AudioAdapter:
    - onFrameReceived: PJSUA2 -> AudioAdapter (PCM16 @ 8kHz)
    - onFrameRequested: AudioAdapter -> PJSUA2 (PCM16 @ 8kHz)
    """

    def __init__(
        self,
        audio_adapter: "AudioAdapter",  # type: ignore
        sample_rate: int = 8000,
        frame_ms: int = 20
    ) -> None:
        """Initialize custom media port.

        Args:
            audio_adapter: AudioAdapter instance
            sample_rate: Sample rate (must be 8000 for SIP)
            frame_ms: Frame duration in milliseconds
        """
        # Initialize parent AudioMediaPort
        super().__init__()

        self._adapter = audio_adapter
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._frame_size = (sample_rate * frame_ms * 2) // 1000  # PCM16: 2 bytes/sample
        self._logger = structlog.get_logger(__name__)

        # Debug counters
        self._frames_received_count = 0
        self._frames_requested_count = 0
        self._silence_frames_count = 0

        self._logger.info(
            "PJSIPMediaPort initialized",
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            frame_size=self._frame_size
        )

    def onFrameReceived(self, frame: "pj.MediaFrame") -> None:  # type: ignore
        """Called by PJSUA2 when audio frame is received from SIP.

        Args:
            frame: PJSUA2 MediaFrame object containing PCM16 audio data
        """
        try:
            # Extract PCM16 data from PJSUA2 MediaFrame
            if frame.buf and frame.size > 0:
                # Convert frame buffer to bytes
                pcm16_data = bytes(frame.buf[:frame.size])

                # Send PCM16 frame to AudioAdapter
                self._adapter.on_rx_pcm16_8k(pcm16_data)

                self._frames_received_count += 1
                if self._frames_received_count % AudioConstants.LOG_INTERVAL_FRAMES == 0:
                    self._logger.info(
                        "📞 PJSIP MediaPort received frames",
                        count=self._frames_received_count
                    )
            else:
                self._logger.debug("Received empty frame from PJSUA2")

        except Exception as e:
            self._logger.error("Error in onFrameReceived", error=str(e), exc_info=True)

    def onFrameRequested(self, frame: "pj.MediaFrame") -> None:  # type: ignore
        """Called by PJSUA2 when it needs audio frame to send to SIP.

        Simplified: Just read from AudioAdapter and pass to PJSUA2.
        All padding/quality handling is done in AudioAdapter.feed_ai_audio().

        Args:
            frame: PJSUA2 MediaFrame object to fill with PCM16 audio data
        """
        try:
            # Get PCM16 frame from AudioAdapter (always returns 320 bytes)
            pcm16_data = self._adapter.get_tx_pcm16_8k_nowait()

            # Fill PJSUA2 MediaFrame
            frame.buf = pj.ByteVector(pcm16_data)
            frame.size = len(pcm16_data)
            frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO

            # Track stats for debugging
            self._frames_requested_count += 1
            if pcm16_data == AudioConstants.SILENCE_FRAME:
                self._silence_frames_count += 1

            # Periodic logging
            if self._frames_requested_count % AudioConstants.LOG_INTERVAL_STATS == 0:
                self._logger.debug(
                    "📊 TX stats",
                    total=self._frames_requested_count,
                    silence=self._silence_frames_count,
                    silence_ratio=f"{100*self._silence_frames_count/self._frames_requested_count:.1f}%"
                )

        except Exception as e:
            self._logger.error("Error in onFrameRequested", error=str(e), exc_info=True)
            # Return silence on error
            silence_bytes = b'\x00' * self._frame_size
            frame.buf = pj.ByteVector(silence_bytes)
            frame.size = self._frame_size
            frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO


def create_endpoint(
    domain: str = "localhost",
    transport_type: str = "udp",
    port: int = 6060,
    event_loop: Optional[asyncio.AbstractEventLoop] = None
) -> PJSIPEndpoint:
    """Factory function to create PJSIP endpoint with userless account.

    Creates a receive-only SIP endpoint that listens for incoming INVITE requests.
    No registration to SIP server required.

    Args:
        domain: SIP domain or IP address for URI
        transport_type: Transport type (udp/tcp)
        port: SIP listening port
        event_loop: Asyncio event loop for run_coroutine_threadsafe calls

    Returns:
        PJSIP endpoint instance

    Note:
        Requires pjsua2 to be installed. If not available, the module
        import will fail and the application will exit.
    """
    return PJSIPEndpoint(
        domain=domain,
        transport_type=transport_type,
        port=port,
        event_loop=event_loop
    )