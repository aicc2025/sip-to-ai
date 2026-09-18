"""Gemini Live API adapter.

Bidirectional audio streaming with Google's Gemini Live models (default: gemini-3.8-live):

1. WebSocket connection to Gemini Live API
2. Session configuration with voice settings
3. Audio streaming with resampling (8kHz SIP <-> 16kHz/24kHz Gemini)
4. Event handling for transcription and errors
5. Transparent session resumption (goAway and dropped connections)

Audio Flow:
- Input: PCM16 @ 8kHz -> resample to PCM16 @ 16kHz -> base64 -> Gemini
- Output: Gemini -> PCM16 @ 24kHz -> resample to PCM16 @ 8kHz

Session resumption:
- Every setup message requests session resumption; the server answers with
  sessionResumptionUpdate messages whose newHandle is stored.
- On goAway, or when the socket closes without close() being called, a
  background recovery opens a new socket with the stored handle (the model
  keeps the conversation context) and swaps it in. Consumers keep their
  receive_chunks()/events() streams; uplink frames are dropped while no usable
  socket exists.
- The server rejects a resume (close 1011) while the session's previous socket
  is still open, so a planned swap (goAway, reconnect()) is break-before-make:
  the old socket's close handshake completes first, then the new socket
  resumes (a few seconds without a usable socket).
- If resumption keeps failing for resume_budget_sec, a fresh session without
  context is opened so the call continues. Only if that fails too the streams
  end (DISCONNECTED event + end-of-stream marker), which ends the call.

Note: Gemini Live does not support G.711/mulaw natively, so resampling is required.
"""

import asyncio
import base64
import json
import os
import time
from typing import Any, AsyncIterator, Dict, Optional, Set

import structlog
import websockets
from websockets.asyncio.client import ClientConnection

from app.ai.duplex_base import AiDuplexBase, AiEvent, AiEventType, AudioChunkQueue, describe_error
from app.utils.codec import resample_pcm16


class GeminiLiveClient(AiDuplexBase):
    """Gemini Live API client for bidirectional audio streaming."""

    # Gemini Live API WebSocket endpoint
    WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

    # Audio sample rates
    SIP_SAMPLE_RATE = 8000      # SIP uses 8kHz
    GEMINI_INPUT_RATE = 16000   # Gemini expects 16kHz input
    GEMINI_OUTPUT_RATE = 24000  # Gemini outputs 24kHz

    # Pong wait for ping(). Gemini pongs were observed to take 8-30s, so a
    # short timeout reports a working connection as dead.
    PING_TIMEOUT_SEC = 30.0
    # Timeouts for opening a socket and for the setupComplete answer
    OPEN_TIMEOUT_SEC = 10.0
    SETUP_TIMEOUT_SEC = 10.0
    # Fresh-session (no handle) attempts after resumption gave up
    FRESH_SESSION_ATTEMPTS = 2
    FRESH_SESSION_RETRY_DELAY_SEC = 1.0
    # Max wait for the close handshake of a replaced or failed socket
    RETIRE_CLOSE_TIMEOUT_SEC = 2.0
    # Max wait for the close handshake before resuming on a planned swap
    # (goAway, reconnect()); Gemini acknowledged close frames after 1.4-3s
    PLANNED_CLOSE_TIMEOUT_SEC = 5.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.8-live",
        voice: str = "Puck",
        instructions: str = "You are a helpful assistant.",
        greeting: Optional[str] = None,
        ws_url: Optional[str] = None,
        resume_budget_sec: float = 30.0,
        resume_initial_backoff_sec: float = 0.5,
        resume_max_backoff_sec: float = 4.0,
    ) -> None:
        """Initialize Gemini Live client.

        Args:
            api_key: Google AI API key (falls back to GEMINI_API_KEY env var)
            model: Gemini model to use (must support Live API)
            voice: Voice for speech synthesis (Puck, Charon, Kore, Fenrir, Aoede)
            instructions: System instructions for the AI
            greeting: Optional greeting message to speak when session starts
            ws_url: WebSocket endpoint override (default: WS_URL; used by tests)
            resume_budget_sec: How long to retry resuming a session with the
                stored handle before falling back to a fresh session
            resume_initial_backoff_sec: First delay between resume attempts
                (doubled after each failure)
            resume_max_backoff_sec: Maximum delay between resume attempts
        """
        super().__init__(sample_rate=self.SIP_SAMPLE_RATE, frame_ms=20)

        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("Gemini API key not provided")

        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._greeting = greeting
        self._ws_url = ws_url or self.WS_URL
        self._resume_budget_sec = resume_budget_sec
        self._resume_initial_backoff_sec = resume_initial_backoff_sec
        self._resume_max_backoff_sec = resume_max_backoff_sec

        # Current socket. Replaced by the recovery task; every socket has its
        # own reader task, and readers of replaced sockets stop processing.
        self._ws: Optional[ClientConnection] = None
        # True when the current socket is known to be closed (uplink frames are
        # dropped until a new socket is installed)
        self._ws_lost = False
        self._reader_tasks: Set[asyncio.Task[None]] = set()

        # Session resumption
        self._resume_handle: Optional[str] = None
        self._recovery_task: Optional[asyncio.Task[None]] = None
        self._closing = False
        self._frames_dropped = 0
        self._recoveries = 0

        # Event queues
        # Audio never blocks the WebSocket reader (see AudioChunkQueue)
        self._audio_queue = AudioChunkQueue()
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue(maxsize=100)

        # Stats
        self._audio_frames_sent = 0
        self._audio_chunks_received = 0

        # Transcription buffers (accumulate before logging)
        self._user_transcript_buffer = ""
        self._ai_transcript_buffer = ""

        self._logger = structlog.get_logger(__name__)

    @property
    def is_recovering(self) -> bool:
        """True while a session swap or reconnect is in progress."""
        return self._recovery_task is not None and not self._recovery_task.done()

    async def connect(self) -> None:
        """Connect to Gemini Live API."""
        if self._connected:
            return

        self._closing = False
        self._resume_handle = None
        try:
            ws = await self._open_session(handle=None)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Gemini Live: {describe_error(e)}") from e

        self._connected = True
        self._install_socket(ws)
        self._emit_event(
            AiEvent(
                type=AiEventType.CONNECTED,
                data={"resumed": False},
                timestamp=time.time()
            )
        )
        self._logger.info(
            "Gemini Live connected",
            model=self._model,
            voice=self._voice
        )

        # Send greeting if configured
        if self._greeting:
            try:
                await self._send_greeting()
            except Exception as e:
                await self.close()
                raise ConnectionError(f"Failed to connect to Gemini Live: {describe_error(e)}") from e

    async def close(self) -> None:
        """Close connection.

        Cancels an in-flight session recovery and all socket reader tasks.
        """
        self._closing = True
        was_connected = self._connected
        self._connected = False
        if was_connected:
            # Wake receive_chunks() so it ends
            self._audio_queue.put_nowait(b"")

        recovery = self._recovery_task
        self._recovery_task = None
        if recovery and not recovery.done():
            recovery.cancel()
            try:
                await recovery
            except asyncio.CancelledError:
                self._logger.debug("Session recovery cancelled")
            except Exception as e:
                self._logger.debug("Session recovery ended with error", error=describe_error(e))

        await self._cancel_readers()

        ws = self._ws
        if ws is not None:
            try:
                async with asyncio.timeout(self.RETIRE_CLOSE_TIMEOUT_SEC):
                    await ws.close()
            except Exception as e:
                self._logger.debug("WebSocket close error", error=describe_error(e))

        if was_connected:
            self._logger.info("Gemini Live disconnected")

    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to Gemini.

        Resamples 8kHz to 16kHz before sending. While the socket is being
        replaced (no usable socket) frames are dropped silently.

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes)
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        # Validate input: 320 bytes = 160 samples @ 8kHz = 20ms
        if len(frame_20ms) != 320:
            raise ValueError(f"Expected 320 bytes PCM16 @ 8kHz, got {len(frame_20ms)}")

        if self._ws_lost:
            self._drop_frame()
            return

        # Resample 8kHz -> 16kHz (doubles the samples)
        pcm16_16k = resample_pcm16(frame_20ms, self.SIP_SAMPLE_RATE, self.GEMINI_INPUT_RATE)

        # Log first few frames for debugging
        if self._audio_frames_sent < 3:
            self._logger.info(
                f"Frame #{self._audio_frames_sent + 1}",
                input_size=len(frame_20ms),
                output_size=len(pcm16_16k),
                expected_output=640  # 320 samples * 2 bytes @ 16kHz
            )

        # Send realtime input message with base64-encoded audio.
        # Use the `audio` Blob field; `mediaChunks` is deprecated and rejected
        # (WebSocket close 1007) by newer Live API models.
        message = {
            "realtimeInput": {
                "audio": {
                    "mimeType": "audio/pcm;rate=16000",
                    "data": base64.b64encode(pcm16_16k).decode("utf-8")
                }
            }
        }

        try:
            await self._ws.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            # The socket's reader starts the recovery; the frame is lost
            self._drop_frame()
            return

        self._audio_frames_sent += 1
        if self._audio_frames_sent % 50 == 0:  # Log every 1 second
            self._logger.info(f"Sent {self._audio_frames_sent} audio frames to Gemini")

    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Receive audio chunks from Gemini.

        The stream continues across session swaps and recoveries; it only ends
        on close() or when recovery failed.

        Yields:
            PCM16 audio chunks @ 8kHz (resampled from 24kHz)
        """
        while self._connected:
            try:
                chunk = await self._audio_queue.get()
                if not self._connected and chunk == b"":
                    break
                if chunk == b"":
                    # Stale end-of-stream marker from an earlier session
                    continue
                yield chunk
            except Exception as e:
                self._logger.error("Audio stream error", error=str(e))
                break

    async def events(self) -> AsyncIterator[AiEvent]:
        """Iterate over events from Gemini.

        Yields:
            AI events (CONNECTED, DISCONNECTED, ERROR, etc.)
        """
        while self._connected:
            try:
                event = await self._event_queue.get()
                yield event
            except Exception as e:
                self._logger.error("Event stream error", error=str(e))
                break

    async def update_session(self, config: Dict) -> None:
        """Update session configuration.

        Note: Gemini Live has limited session update support.
        Model cannot be changed after setup.

        Args:
            config: Session configuration (instructions, voice, etc.)
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        # Gemini allows updating some parameters via setup message
        # But model cannot be changed after initial setup
        self._logger.warning(
            "Session update requested - Gemini has limited update support",
            config_keys=list(config.keys())
        )

    async def ping(self) -> bool:
        """Check connection health.

        While a session recovery is running the connection is reported healthy:
        the recovery has its own time budget and ends the streams on failure.

        Returns:
            True if healthy
        """
        if not self._connected or not self._ws:
            return False
        if self.is_recovering:
            return True

        try:
            pong_waiter = await self._ws.ping()
            await asyncio.wait_for(pong_waiter, timeout=self.PING_TIMEOUT_SEC)
            return True
        except (asyncio.TimeoutError, Exception):
            return False

    async def reconnect(self) -> None:
        """Reconnect to the service, keeping the conversation when possible.

        When connected, replaces the socket through the same path as goAway
        (resume with the stored handle, fresh session as fallback). When not
        connected, opens a new session.

        Raises:
            ConnectionError: If reconnection fails
        """
        if not self._connected:
            await self.connect()
            return

        self._start_recovery("manual reconnect")
        task = self._recovery_task
        if task is not None:
            await asyncio.shield(task)
        if not self._connected:
            raise ConnectionError("Gemini Live reconnect failed")

    def _build_setup_message(self, handle: Optional[str]) -> Dict[str, Any]:
        """Build the session setup message.

        Args:
            handle: Session resumption handle (None for a new session)

        Returns:
            Setup message
        """
        return {
            "setup": {
                "model": f"models/{self._model}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": self._voice
                            }
                        }
                    }
                },
                "systemInstruction": {
                    "parts": [{
                        "text": self._instructions
                    }]
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                # Request resumption handles; an empty object starts a new session
                "sessionResumption": {"handle": handle} if handle else {},
                # Removes the 15-minute audio session limit
                "contextWindowCompression": {"slidingWindow": {}}
            }
        }

    async def _send_setup(
        self,
        ws: Optional[ClientConnection] = None,
        handle: Optional[str] = None
    ) -> None:
        """Send the session setup message.

        Args:
            ws: Socket to send on (default: current socket)
            handle: Session resumption handle (None for a new session)
        """
        target = ws or self._ws
        if target is None:
            raise ConnectionError("Not connected")

        self._logger.info(
            "Sending Gemini setup",
            model=self._model,
            voice=self._voice,
            instructions_length=len(self._instructions),
            resume=handle is not None
        )

        await target.send(json.dumps(self._build_setup_message(handle)))

    async def _open_session(self, handle: Optional[str]) -> ClientConnection:
        """Open a socket, send setup and wait for setupComplete.

        The socket is not installed; the caller swaps it in. Messages received
        before setupComplete are handled here (resumption updates are stored).

        Args:
            handle: Session resumption handle (None for a new session)

        Returns:
            Socket with a completed setup

        Raises:
            Exception: If connecting, setup or the setupComplete wait fails
        """
        url = f"{self._ws_url}?key={self._api_key}"
        async with asyncio.timeout(self.OPEN_TIMEOUT_SEC):
            ws = await websockets.connect(
                url,
                open_timeout=self.OPEN_TIMEOUT_SEC,
                proxy=None,  # Direct connect; don't auto-use env SOCKS proxy
                # No library keepalive: Gemini pongs can take 8-30s, which trips
                # the default 20s ping_timeout (close 1011). Liveness is owned
                # by the bridge health check (ping()).
                ping_interval=None,
            )

        try:
            await self._send_setup(ws, handle)
            self._logger.info("Waiting for setup complete from Gemini...", resume=handle is not None)
            async with asyncio.timeout(self.SETUP_TIMEOUT_SEC):
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if "setupComplete" in data:
                        break
                    if "sessionResumptionUpdate" in data:
                        self._handle_resumption_update(data["sessionResumptionUpdate"])
                    elif data:
                        self._logger.debug("Message before setupComplete", keys=list(data.keys()))
        except BaseException:
            try:
                async with asyncio.timeout(self.RETIRE_CLOSE_TIMEOUT_SEC):
                    await ws.close()
            except BaseException:
                pass
            raise

        self._logger.info("Gemini setup complete", resume=handle is not None)
        return ws

    def _install_socket(self, ws: ClientConnection) -> Optional[ClientConnection]:
        """Make ws the current socket and start its reader task.

        Args:
            ws: Socket with a completed setup

        Returns:
            The previous socket (to be retired by the caller), if any
        """
        old = self._ws
        self._ws = ws
        self._ws_lost = False
        # Partial transcripts belong to the previous socket's turn
        self._user_transcript_buffer = ""
        self._ai_transcript_buffer = ""
        task = asyncio.create_task(self._reader(ws), name="gemini-message-handler")
        self._reader_tasks.add(task)
        task.add_done_callback(self._reader_tasks.discard)
        return old if old is not ws else None

    async def _retire_socket(self, ws: ClientConnection, timeout: Optional[float] = None) -> None:
        """Close a socket that is being replaced or failed.

        Waits for the close handshake, then aborts the transport on timeout.

        Args:
            ws: Socket to close
            timeout: Handshake wait (default: RETIRE_CLOSE_TIMEOUT_SEC)
        """
        try:
            async with asyncio.timeout(timeout or self.RETIRE_CLOSE_TIMEOUT_SEC):
                await ws.close()
        except Exception as e:
            self._logger.debug("Socket close handshake incomplete, aborting", error=describe_error(e))
            try:
                ws.transport.abort()
            except Exception:
                pass

    async def _cancel_readers(self) -> None:
        """Cancel all socket reader tasks and wait for them."""
        current = asyncio.current_task()
        tasks = [t for t in self._reader_tasks if t is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _handle_resumption_update(self, update: Dict[str, Any]) -> None:
        """Store a new session resumption handle.

        Args:
            update: sessionResumptionUpdate payload
        """
        new_handle = update.get("newHandle")
        if update.get("resumable") and new_handle:
            self._resume_handle = new_handle
            self._logger.debug("Session resumption handle updated", handle_prefix=new_handle[:12])

    def _drop_frame(self) -> None:
        """Count an uplink frame dropped while no usable socket exists."""
        self._frames_dropped += 1
        if self._frames_dropped == 1:
            self._logger.info("Dropping uplink audio while Gemini session reconnects")

    def _start_recovery(self, reason: str, detail: Optional[Any] = None) -> None:
        """Start the background session swap/recovery if not already running.

        Args:
            reason: Why the socket is replaced (goAway, connection lost, ...)
            detail: Extra data for the log (goAway payload, close code, ...)
        """
        if self._closing or not self._connected:
            return
        if self.is_recovering:
            self._logger.debug("Session recovery already running", reason=reason)
            return

        self._logger.warning(
            "Gemini session recovery started",
            reason=reason,
            detail=detail,
            has_handle=self._resume_handle is not None
        )
        self._frames_dropped = 0
        self._recovery_task = asyncio.create_task(
            self._recover(reason),
            name="gemini-session-recovery"
        )

    async def _recover(self, reason: str) -> None:
        """Replace the current socket, keeping the conversation when possible.

        0. If the current socket is still open (goAway, reconnect()), close it
           first: the server rejects the resume (1011) until it has completed
           the close handshake of the old socket. Uplink frames are dropped
           from here on.
        1. Resume with the stored handle, exponential backoff, until
           resume_budget_sec is used up.
        2. Fall back to a fresh session without handle (no context, no greeting).
        3. If that fails too, end the streams (DISCONNECTED + end marker).

        Args:
            reason: Why the socket is replaced
        """
        started = time.monotonic()
        old = self._ws
        if old is not None and not self._ws_lost:
            # Planned swap. _ws_lost also stops the old reader from treating
            # the close as a connection loss; it keeps delivering messages
            # until the close handshake completes.
            self._ws_lost = True
            await self._retire_socket(old, timeout=self.PLANNED_CLOSE_TIMEOUT_SEC)
            self._logger.info(
                "Old Gemini socket closed for session swap",
                elapsed_sec=round(time.monotonic() - started, 2)
            )
        await self._recover_steps(reason, started)

    async def _recover_steps(self, reason: str, started: float) -> None:
        """Resume / fresh-session / fail steps of _recover().

        Args:
            reason: Why the socket is replaced
            started: time.monotonic() when the recovery started
        """
        new_ws: Optional[ClientConnection] = None
        resumed = False

        handle = self._resume_handle
        if handle:
            delay = self._resume_initial_backoff_sec
            attempt = 0
            while True:
                attempt += 1
                try:
                    new_ws = await self._open_session(handle)
                    resumed = True
                    break
                except Exception as e:
                    elapsed = time.monotonic() - started
                    self._logger.warning(
                        "Gemini session resume attempt failed",
                        attempt=attempt,
                        elapsed_sec=round(elapsed, 1),
                        error=describe_error(e)
                    )
                remaining = self._resume_budget_sec - (time.monotonic() - started)
                if remaining <= 0:
                    break
                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 2, self._resume_max_backoff_sec)
                # A newer handle may have arrived on a still-open old socket
                handle = self._resume_handle or handle
        else:
            self._logger.warning("No session resumption handle available")

        if new_ws is None:
            self._logger.error(
                "Gemini session resumption failed - starting a fresh session; "
                "the conversation context is lost",
                reason=reason,
                elapsed_sec=round(time.monotonic() - started, 1)
            )
            self._resume_handle = None
            for attempt in range(1, self.FRESH_SESSION_ATTEMPTS + 1):
                try:
                    new_ws = await self._open_session(None)
                    break
                except Exception as e:
                    self._logger.warning(
                        "Gemini fresh session attempt failed",
                        attempt=attempt,
                        error=describe_error(e)
                    )
                    if attempt < self.FRESH_SESSION_ATTEMPTS:
                        await asyncio.sleep(self.FRESH_SESSION_RETRY_DELAY_SEC)

        if new_ws is None:
            self._logger.error(
                "Gemini session recovery failed - ending AI stream",
                reason=reason,
                elapsed_sec=round(time.monotonic() - started, 1)
            )
            await self._fail_session()
            return

        if self._closing:
            await self._retire_socket(new_ws)
            return

        old = self._install_socket(new_ws)
        self._recoveries += 1
        self._logger.info(
            "Gemini session recovered",
            reason=reason,
            resumed=resumed,
            elapsed_sec=round(time.monotonic() - started, 2),
            frames_dropped=self._frames_dropped,
            recoveries=self._recoveries
        )
        if old is not None:
            await self._retire_socket(old)

    async def _fail_session(self) -> None:
        """End the streams after recovery failed (the bridge ends the call)."""
        if self._closing or not self._connected:
            return
        self._connected = False
        self._emit_event(
            AiEvent(
                type=AiEventType.DISCONNECTED,
                timestamp=time.time()
            )
        )
        self._audio_queue.put_nowait(b"")
        ws = self._ws
        if ws is not None:
            await self._retire_socket(ws)

    async def _send_greeting(self) -> None:
        """Send greeting message to trigger initial response."""
        if not self._ws or not self._greeting:
            return

        # Send text content to trigger greeting response
        greeting_message = {
            "clientContent": {
                "turns": [{
                    "role": "user",
                    "parts": [{
                        "text": f"[System: Greet the caller with this message: {self._greeting}]"
                    }]
                }],
                "turnComplete": True
            }
        }

        await self._ws.send(json.dumps(greeting_message))
        self._logger.info("Greeting request sent", greeting_preview=self._greeting[:50])

    async def _reader(self, ws: ClientConnection) -> None:
        """Read messages from one socket until it closes or is replaced.

        Messages are only processed while ws is the current socket. A close of
        the current socket that was not requested via close() starts recovery.

        Args:
            ws: Socket to read from
        """
        while True:
            try:
                message = await ws.recv()
            except websockets.exceptions.ConnectionClosed as e:
                if ws is self._ws and not self._ws_lost and not self._closing and self._connected:
                    self._ws_lost = True
                    rcvd = e.rcvd
                    self._logger.warning(
                        "WebSocket connection closed unexpectedly",
                        code=rcvd.code if rcvd else None,
                        reason=rcvd.reason if rcvd else None
                    )
                    self._start_recovery(
                        "connection lost",
                        {"code": rcvd.code, "reason": rcvd.reason} if rcvd else None
                    )
                return
            except Exception as e:
                # Not expected from recv(); treat the socket as unusable
                self._logger.error("WebSocket receive error", error=describe_error(e))
                if ws is self._ws and not self._ws_lost and not self._closing and self._connected:
                    self._ws_lost = True
                    self._start_recovery("receive error", describe_error(e))
                return

            if ws is not self._ws:
                # Replaced socket: ignore anything it still delivers
                return

            try:
                self._mark_received()
                data = json.loads(message)
                await self._process_message(data)
            except json.JSONDecodeError as e:
                self._logger.error("Failed to decode message", error=str(e))
            except Exception as e:
                self._logger.error("Message handler error", error=str(e))

    async def _process_message(self, data: Dict) -> None:
        """Process WebSocket message from Gemini.

        Args:
            data: Parsed JSON message
        """
        # setupComplete is consumed by _open_session(); a late duplicate is ignored
        if "setupComplete" in data:
            self._logger.debug("Ignoring setupComplete outside session setup")
            return

        # Check for server content (model response)
        if "serverContent" in data:
            server_content = data["serverContent"]

            # Check for model turn with audio
            model_turn = server_content.get("modelTurn", {})
            parts = model_turn.get("parts", [])

            for part in parts:
                # Handle audio data
                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    mime_type = inline_data.get("mimeType", "")
                    audio_data = inline_data.get("data", "")

                    if "audio" in mime_type and audio_data:
                        await self._handle_audio(audio_data)

                # Handle text (transcription)
                if "text" in part:
                    text = part["text"]
                    self._logger.info(f"AI response text: {text}")
                    self._emit_event(
                        AiEvent(
                            type=AiEventType.TRANSCRIPT_FINAL,
                            data={"text": text, "role": "model"},
                            timestamp=time.time()
                        )
                    )

            # Check for input transcription (user speech) - accumulate
            if "inputTranscription" in server_content:
                input_text = server_content["inputTranscription"].get("text", "")
                if input_text:
                    self._user_transcript_buffer += input_text

            # Check for output transcription (model speech) - accumulate
            if "outputTranscription" in server_content:
                output_text = server_content["outputTranscription"].get("text", "")
                if output_text:
                    self._ai_transcript_buffer += output_text

            # Check for turn complete - log accumulated transcription
            if server_content.get("turnComplete"):
                # Log accumulated user transcription
                if self._user_transcript_buffer.strip():
                    self._logger.info(f"User: {self._user_transcript_buffer.strip()}")
                    self._emit_event(
                        AiEvent(
                            type=AiEventType.TRANSCRIPT_FINAL,
                            data={"text": self._user_transcript_buffer.strip(), "role": "user"},
                            timestamp=time.time()
                        )
                    )
                    self._user_transcript_buffer = ""

                # Log accumulated AI transcription
                if self._ai_transcript_buffer.strip():
                    self._logger.info(f"AI: {self._ai_transcript_buffer.strip()}")
                    self._ai_transcript_buffer = ""

                self._logger.debug("Model turn complete")

            # Check for interrupted - log what we have so far
            if server_content.get("interrupted"):
                if self._ai_transcript_buffer.strip():
                    self._logger.info(f"AI (interrupted): {self._ai_transcript_buffer.strip()}")
                    self._ai_transcript_buffer = ""
                self._logger.info("Model response interrupted (barge-in)")
                self._emit_event(
                    AiEvent(
                        type=AiEventType.TRANSCRIPT_PARTIAL,
                        data={"event": "interrupted"},
                        timestamp=time.time()
                    )
                )

            return

        # Check for tool calls
        if "toolCall" in data:
            self._logger.info("Tool call received", data=data["toolCall"])
            return

        # Go away: the server will close this socket soon (timeLeft). Swap to a
        # resumed session in the background; consumers see no disconnect.
        if "goAway" in data:
            self._logger.warning("Received goAway from Gemini", data=data["goAway"])
            self._start_recovery("goAway", data["goAway"])
            return

        # Session resumption handle updates
        if "sessionResumptionUpdate" in data:
            self._handle_resumption_update(data["sessionResumptionUpdate"])
            return

        # Check for usage metadata
        if "usageMetadata" in data:
            usage = data["usageMetadata"]
            self._logger.debug(
                "Usage metadata",
                prompt_tokens=usage.get("promptTokenCount"),
                response_tokens=usage.get("responseTokenCount")
            )
            return

        # Empty messages carry nothing
        if not data:
            return

        # Log unknown message types
        self._logger.debug("Unknown message type", keys=list(data.keys()))

    async def _handle_audio(self, audio_base64: str) -> None:
        """Handle incoming audio data from Gemini.

        Decodes base64, resamples from 24kHz to 8kHz, and queues for playback.

        Args:
            audio_base64: Base64-encoded PCM16 @ 24kHz audio
        """
        # Decode base64
        pcm16_24k = base64.b64decode(audio_base64)

        # Resample 24kHz -> 8kHz
        pcm16_8k = resample_pcm16(pcm16_24k, self.GEMINI_OUTPUT_RATE, self.SIP_SAMPLE_RATE)

        # Calculate durations for logging
        duration_ms = (len(pcm16_24k) / 2 / self.GEMINI_OUTPUT_RATE) * 1000

        # Queue for playback
        self._queue_audio(pcm16_8k)
        self._audio_chunks_received += 1

        if self._audio_chunks_received % 10 == 0:
            self._logger.info(
                f"Received {self._audio_chunks_received} audio chunks from Gemini",
                pcm16_24k=f"{len(pcm16_24k)}B",
                pcm16_8k=f"{len(pcm16_8k)}B",
                duration=f"{duration_ms:.1f}ms"
            )
        elif self._audio_chunks_received <= 5:
            self._logger.info(
                f"Chunk #{self._audio_chunks_received}",
                pcm16_24k=f"{len(pcm16_24k)}B",
                pcm16_8k=f"{len(pcm16_8k)}B",
                duration=f"{duration_ms:.1f}ms"
            )
