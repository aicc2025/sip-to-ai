"""OpenAI Realtime API adapter.

Default integration uses G.711 μ-law @ 8kHz (native OpenAI support):

1. WebSocket connection to wss://api.openai.com/v1/realtime
2. Session configuration with semantic VAD and barge-in
3. G.711 μ-law audio streaming (no resampling needed)
4. Event handling for transcription and errors

Audio Flow (audio_format="pcmu", default):
- Input: PCM16 @ 8kHz → G.711 μ-law @ 8kHz → OpenAI
- Output: OpenAI → G.711 μ-law @ 8kHz → PCM16 @ 8kHz

Audio Flow (audio_format="pcm16", for Realtime-compatible gateways that only
accept audio/pcm @ 24kHz, e.g. cascade-realtime-gateway):
- Input: PCM16 @ 8kHz → resample → PCM16 @ 24kHz → OpenAI
- Output: OpenAI → PCM16 @ 24kHz → resample → PCM16 @ 8kHz

Session configuration (new schema):
{
    "type": "session.update",
    "session": {
        "type": "realtime",
        "model": "gpt-realtime-2.1",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcmu"},
                "transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad"}
            },
            "output": {
                "format": {"type": "audio/pcmu"},
                "voice": "marin"
            }
        },
        "instructions": "You are a helpful assistant."
    }
}

Key Benefits:
- No resampling overhead (8kHz throughout)
- Better audio quality (no lossy resampling)
- Consistent with Deepgram architecture
- Lower latency and CPU usage
"""

import asyncio
import base64
import json
import os
import time
from typing import AsyncIterator, Dict, Optional

import structlog
import websockets
from websockets.client import WebSocketClientProtocol

from app.ai.duplex_base import AiDuplexBase, AiEvent, AiEventType
from app.utils.codec import Codec, resample_pcm16

# Supported values for the audio_format constructor argument (OPENAI_AUDIO_FORMAT)
AUDIO_FORMATS = ("pcmu", "pcm16")
# Supported values for the noise_reduction constructor argument (OPENAI_NOISE_REDUCTION)
NOISE_REDUCTION_MODES = ("near_field", "far_field", "none")


class OpenAIRealtimeClient(AiDuplexBase):
    """OpenAI Realtime API client."""

    PCM16_SAMPLE_RATE = 24000  # audio/pcm is always 24kHz in the Realtime GA API

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-realtime-2.1",
        voice: str = "marin",
        ws_endpoint: str = "wss://api.openai.com/v1/realtime",
        project: Optional[str] = None,
        organization: Optional[str] = None,
        instructions: str = "You are a helpful assistant.",
        greeting: Optional[str] = None,
        audio_format: str = "pcmu",
        noise_reduction: str = "near_field"
    ) -> None:
        """Initialize OpenAI Realtime client.

        Args:
            api_key: OpenAI API key
            model: Model to use
            voice: Voice for TTS
            ws_endpoint: OpenAI Realtime WebSocket endpoint
            project: Optional OpenAI project id for scoped model access
            organization: Optional OpenAI organization id for scoped model access
            instructions: System instructions/prompt for the AI
            greeting: Optional greeting message to play when call connects
            audio_format: "pcmu" (G.711 μ-law @ 8kHz, default) or "pcm16"
                (PCM16 @ 24kHz with 8kHz ↔ 24kHz resampling)
            noise_reduction: "near_field" (default), "far_field", or "none"
                (sends null, which disables input noise reduction)

        Note:
            - "pcmu" is a direct passthrough SIP → OpenAI → SIP (no resampling)
            - "pcm16" is required by gateways that reject audio/pcmu
        """
        if audio_format not in AUDIO_FORMATS:
            raise ValueError(
                f"audio_format must be one of {AUDIO_FORMATS}, got: {audio_format!r}"
            )
        if noise_reduction not in NOISE_REDUCTION_MODES:
            raise ValueError(
                f"noise_reduction must be one of {NOISE_REDUCTION_MODES}, got: {noise_reduction!r}"
            )

        # OpenAI Realtime API configuration
        self._sip_sample_rate = 8000  # SIP uses 8kHz
        self._pcm16_mode = audio_format == "pcm16"
        if self._pcm16_mode:
            self._openai_sample_rate = self.PCM16_SAMPLE_RATE
            self._audio_format = "audio/pcm"  # PCM16 little-endian @ 24kHz
        else:
            self._openai_sample_rate = 8000  # G.711 μ-law is 8kHz
            self._audio_format = "audio/pcmu"  # G.711 μ-law format
        self._noise_reduction = noise_reduction
        self._frame_ms = 20
        self._sample_rate = self._sip_sample_rate  # Base class expects this

        super().__init__(self._sample_rate, self._frame_ms)

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key not provided")

        self._model = model
        self._voice = voice
        self._project = project or os.getenv("OPENAI_PROJECT") or os.getenv("OPENAI_PROJECT_ID")
        self._organization = organization or os.getenv("OPENAI_ORGANIZATION") or os.getenv("OPENAI_ORG_ID")
        self._instructions = instructions
        self._greeting = greeting
        self._ws: Optional[WebSocketClientProtocol] = None
        self._ws_url = ws_endpoint.rstrip("/")

        # Event queues (using asyncio Queues)
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue(maxsize=100)

        # Control
        self._stop_event = asyncio.Event()
        self._session_created_event = asyncio.Event()
        self._session_updated_event = asyncio.Event()
        self._connect_error: Optional[str] = None
        self._message_handler_task: Optional[asyncio.Task[None]] = None

        # Stats for debugging
        self._audio_frames_sent = 0
        self._audio_chunks_received = 0

        # pcm16 mode: 24kHz bytes not yet resampled. Only whole 3-sample groups
        # (6 bytes) are resampled so 24kHz → 8kHz never drops fractional samples
        # between chunks and every emitted chunk is whole 16-bit samples.
        self._downlink_pcm24k_pending = b""
        # Greeting is first sent out-of-band; set while a fallback is still possible
        self._greeting_out_of_band_pending = False

        self._logger = structlog.get_logger(__name__)

    async def connect(self) -> None:
        """Connect to OpenAI Realtime API."""
        if self._connected:
            return

        try:
            # Connect WebSocket with auth headers and timeout
            # Note: Don't use "OpenAI-Beta: realtime=v1" - that connects to old API
            headers = self._build_headers()

            # Set connection timeout (10 seconds)
            async with asyncio.timeout(10.0):
                self._ws = await websockets.connect(
                    f"{self._ws_url}?model={self._model}",
                    additional_headers=headers,
                    open_timeout=10.0,  # WebSocket-level timeout
                    proxy=None,  # Direct connect; don't auto-use env SOCKS proxy
                )

            self._connected = True
            self._downlink_pcm24k_pending = b""
            self._stop_event.clear()
            self._session_created_event.clear()
            self._session_updated_event.clear()
            self._connect_error = None

            # Start message handler task first
            self._message_handler_task = asyncio.create_task(
                self._message_handler(),
                name="openai-message-handler"
            )

            # Wait for session.created from OpenAI (or fail fast on error event)
            self._logger.info("Waiting for session.created from OpenAI...")
            await self._wait_for_session_event(self._session_created_event, "session.created", 5.0)

            self._logger.info("Received session.created, now configuring session...")

            # Configure session after receiving session.created
            await self._configure_session()

            # Wait for first session.updated from OpenAI (or fail fast on error event)
            self._logger.info("Waiting for session.updated from OpenAI...")
            await self._wait_for_session_event(self._session_updated_event, "session.updated", 5.0)

            self._logger.info("Received session.updated")

            # Send greeting after first session.updated (only once)
            if self._greeting:
                await self._send_greeting()
                self._logger.info("Sent greeting after session.updated")

            self._logger.info(
                "OpenAI Realtime connected",
                model=self._model,
                voice=self._voice,
                has_project=bool(self._project),
                has_organization=bool(self._organization)
            )

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect: {e}")

    async def close(self) -> None:
        """Close connection."""
        if not self._connected:
            return

        self._connected = False
        self._stop_event.set()
        try:
            self._audio_queue.put_nowait(b"")
        except asyncio.QueueFull:
            pass

        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                self._logger.debug("Message handler task cancelled")
                # Expected during close(), no need to propagate

        if self._ws:
            await self._ws.close()

        self._logger.info("OpenAI Realtime disconnected")

    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to OpenAI.

        Converts PCM16 → G.711 μ-law @ 8kHz ("pcmu") or resamples
        PCM16 8kHz → 24kHz ("pcm16") before sending.

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes)
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        # Validate input: 320 bytes = 160 samples @ 8kHz = 20ms
        if len(frame_20ms) != 320:
            raise ValueError(f"Expected 320 bytes PCM16 @ 8kHz, got {len(frame_20ms)}")

        if self._pcm16_mode:
            # Resample PCM16 8kHz → 24kHz (320 bytes → 960 bytes)
            payload = resample_pcm16(frame_20ms, self._sip_sample_rate, self._openai_sample_rate)
            expected_output = 960
        else:
            # Convert PCM16 → G.711 μ-law (320 bytes → 160 bytes)
            payload = Codec.pcm16_to_ulaw(frame_20ms)
            expected_output = 160

        # Log first few frames for debugging
        if self._audio_frames_sent < 3:
            import numpy as np
            samples_pcm16 = np.frombuffer(frame_20ms, dtype=np.int16)
            self._logger.info(
                f"📤 Frame #{self._audio_frames_sent + 1}",
                audio_format=self._audio_format,
                input_size=len(frame_20ms),
                output_size=len(payload),
                expected_output=expected_output,
                pcm16_min=int(samples_pcm16.min()),
                pcm16_max=int(samples_pcm16.max())
            )

        # Send audio append message (base64 encoded as per OpenAI Realtime API spec)
        message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(payload).decode("utf-8")
        }

        await self._ws.send(json.dumps(message))

        self._audio_frames_sent += 1
        if self._audio_frames_sent % 50 == 0:  # Log every 1 second (50 frames * 20ms)
            self._logger.info(f"📤 Sent {self._audio_frames_sent} audio frames to OpenAI")

    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Receive audio chunks from OpenAI.

        Yields:
            PCM16 audio chunks @ 8kHz (variable size, typically 320-4000 bytes)
        """
        while self._connected:
            try:
                chunk = await self._audio_queue.get()
                if not self._connected and chunk == b"":
                    break
                yield chunk
            except Exception as e:
                self._logger.error("Audio stream error", error=str(e))
                break

    async def events(self) -> AsyncIterator[AiEvent]:
        """Iterate over events from OpenAI.

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

    async def update_session(self, config: Dict) -> None:
        """Update session configuration.

        Args:
            config: Session configuration
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        message = {
            "type": "session.update",
            "session": config
        }

        await self._ws.send(json.dumps(message))
        self._logger.info("Session updated")

    async def ping(self) -> bool:
        """Check connection health.

        Returns:
            True if healthy
        """
        if not self._connected or not self._ws:
            return False

        try:
            pong_waiter = await self._ws.ping()
            await asyncio.wait_for(pong_waiter, timeout=5.0)
            return True
        except (asyncio.TimeoutError, Exception):
            return False

    async def reconnect(self) -> None:
        """Reconnect to service."""
        await self.close()
        await asyncio.sleep(1.0)
        await self.connect()

    def _build_headers(self) -> Dict[str, str]:
        """Build OpenAI request headers, including optional project/org scope."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._project:
            headers["OpenAI-Project"] = self._project
        if self._organization:
            headers["OpenAI-Organization"] = self._organization
        return headers

    async def _wait_for_session_event(
        self, event: asyncio.Event, name: str, timeout: float
    ) -> None:
        """Wait for an asyncio.Event or raise with the OpenAI error if one arrives first."""
        async with asyncio.timeout(timeout):
            while True:
                if event.is_set():
                    return
                if self._connect_error is not None:
                    raise ConnectionError(f"OpenAI rejected {name}: {self._connect_error}")
                await asyncio.sleep(0.05)

    def _audio_format_payload(self) -> Dict:
        """Return the session audio format object for the configured audio format."""
        if self._pcm16_mode:
            return {"type": self._audio_format, "rate": self._openai_sample_rate}
        return {"type": self._audio_format}

    def _build_session_config(self) -> Dict:
        """Build the initial session.update event (GA schema)."""
        noise_reduction: Optional[Dict] = None
        if self._noise_reduction != "none":
            noise_reduction = {"type": self._noise_reduction}

        output: Dict = {"format": self._audio_format_payload()}
        # An empty voice leaves the server-side default voice in place
        if self._voice:
            output["voice"] = self._voice

        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self._model,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": self._audio_format_payload(),
                        "transcription": {
                            "model": "whisper-1"
                        },
                        "noise_reduction": noise_reduction,
                        "turn_detection": {
                            "type": "semantic_vad",
                            "create_response": True,
                            "eagerness": "medium"
                        }
                    },
                    "output": output
                },
                "instructions": self._instructions
            }
        }

    async def _configure_session(self) -> None:
        """Configure initial session using new schema."""
        config = self._build_session_config()

        self._logger.info(
            "Configuring OpenAI session (new schema)",
            audio_format=self._audio_format,
            input_sample_rate=self._openai_sample_rate,
            voice=self._voice,
            noise_reduction=self._noise_reduction,
            has_greeting=self._greeting is not None,
            instructions_length=len(self._instructions)
        )

        # Log the full config for debugging
        self._logger.debug(f"Session config: {json.dumps(config, indent=2)}")

        await self._ws.send(json.dumps(config))

    async def _send_greeting(self, out_of_band: bool = True) -> None:
        """Send greeting message to OpenAI.

        Args:
            out_of_band: Send with conversation "none" so the greeting is not
                added to the conversation. Gateways that only accept "auto"
                reject this; _process_message then resends with out_of_band=False.
        """
        if not self._ws or not self._greeting:
            return

        response: Dict = {
            "instructions": self._greeting,
            "output_modalities": ["audio"],
            "metadata": {
                "response_purpose": "greeting"
            }
        }
        if out_of_band:
            response["conversation"] = "none"

        greeting_request = {
            "type": "response.create",
            "response": response
        }

        self._greeting_out_of_band_pending = out_of_band
        await self._ws.send(json.dumps(greeting_request))
        self._logger.info(
            "Greeting request sent",
            out_of_band=out_of_band,
            greeting_preview=self._greeting[:50]
        )

    def _convert_downlink_audio(self, audio_bytes: bytes) -> bytes:
        """Convert a decoded output audio delta to PCM16 @ 8kHz.

        Returns b"" when a pcm16 chunk is too short to yield a whole 8kHz sample;
        the remainder is kept and prepended to the next chunk.
        """
        if not self._pcm16_mode:
            # G.711 μ-law @ 8kHz → PCM16 @ 8kHz
            return Codec.ulaw_to_pcm16(audio_bytes)

        data = self._downlink_pcm24k_pending + audio_bytes
        group = 2 * (self._openai_sample_rate // self._sip_sample_rate)  # 6 bytes
        usable = len(data) - (len(data) % group)
        self._downlink_pcm24k_pending = data[usable:]
        if usable == 0:
            return b""
        return resample_pcm16(data[:usable], self._openai_sample_rate, self._sip_sample_rate)

    async def _message_handler(self) -> None:
        """Handle WebSocket messages."""
        if not self._ws:
            return

        while not self._stop_event.is_set():
            try:
                message = await self._ws.recv()
                data = json.loads(message)

                await self._process_message(data)

            except websockets.exceptions.ConnectionClosed:
                self._logger.warning("WebSocket connection closed")
                self._connected = False
                self._stop_event.set()
                event = AiEvent(
                    type=AiEventType.DISCONNECTED,
                    timestamp=time.time()
                )
                try:
                    self._event_queue.put_nowait(event)
                except asyncio.QueueFull:
                    self._logger.debug("Event queue full, dropping disconnect event")
                try:
                    self._audio_queue.put_nowait(b"")
                except asyncio.QueueFull:
                    pass
                break
            except Exception as e:
                self._logger.error("Message handler error", error=str(e))

    async def _process_message(self, data: Dict) -> None:
        """Process WebSocket message.

        Args:
            data: Message data
        """
        msg_type = data.get("type")
        self._logger.debug(f"Received OpenAI event: {msg_type}")

        if msg_type == "session.created":
            # Signal that session is created
            self._session_created_event.set()

            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.CONNECTED,
                    data=data.get("session"),
                    timestamp=time.time()
                )
            )

        elif msg_type == "session.updated":
            session_data = data.get("session", {})
            # Log transcription config
            transcription = session_data.get("input_audio_transcription")
            self._logger.info(f"Session updated - input_audio_transcription: {transcription}")

            # Signal that session.updated received (for connect() to proceed)
            self._session_updated_event.set()

            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.SESSION_UPDATED,
                    data=session_data,
                    timestamp=time.time()
                )
            )

        elif msg_type == "input_audio_buffer.speech_started":
            # User started speaking - this is our barge-in signal
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"event": "speech_started"},
                    timestamp=time.time()
                )
            )

        elif msg_type == "input_audio_buffer.speech_stopped":
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"event": "speech_stopped"},
                    timestamp=time.time()
                )
            )

        elif msg_type == "conversation.item.input_audio_transcription.delta":
            # Incremental transcription results
            delta_text = data.get("delta")
            self._logger.info(f"🎤 Transcription delta: {delta_text}")
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"text": delta_text},
                    timestamp=time.time()
                )
            )

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # Final transcription result
            transcript = data.get("transcript")
            self._logger.info(f"✅ Transcription completed: {transcript}")
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_FINAL,
                    data={"text": transcript},
                    timestamp=time.time()
                )
            )

        elif msg_type == "response.audio_transcript.delta":
            # AI response transcript (what the AI is saying)
            delta_text = data.get("delta")
            self._logger.info(f"🤖 AI transcript delta: {delta_text}")

        elif msg_type == "response.audio_transcript.done":
            # AI response transcript completed
            transcript = data.get("transcript")
            self._logger.info(f"✅ AI transcript done: {transcript}")

        elif msg_type == "response.output_audio.delta":
            # Audio chunk from AI (base64 G.711 μ-law @ 8kHz or PCM16 @ 24kHz)
            audio_base64 = data.get("delta")
            if audio_base64:
                # Decode base64 to get audio bytes
                audio_bytes = base64.b64decode(audio_base64)

                # Convert to PCM16 @ 8kHz for the SIP side
                pcm16_8k = self._convert_downlink_audio(audio_bytes)
                if not pcm16_8k:
                    return

                # Log chunk sizes
                chunk_size_in = len(audio_bytes)
                chunk_size_pcm16 = len(pcm16_8k)
                duration_ms = (chunk_size_pcm16 / 2 / self._sip_sample_rate) * 1000

                await self._audio_queue.put(pcm16_8k)
                self._audio_chunks_received += 1

                if self._audio_chunks_received % 10 == 0:
                    self._logger.info(
                        f"📢 Received {self._audio_chunks_received} audio chunks ({self._audio_format})",
                        input_bytes=f"{chunk_size_in}B",
                        pcm16_8k=f"{chunk_size_pcm16}B",
                        duration=f"{duration_ms:.1f}ms"
                    )
                elif self._audio_chunks_received <= 5:
                    self._logger.info(
                        f"📢 Chunk #{self._audio_chunks_received} ({self._audio_format})",
                        input_bytes=f"{chunk_size_in}B",
                        pcm16_8k=f"{chunk_size_pcm16}B",
                        duration=f"{duration_ms:.1f}ms"
                    )

        elif msg_type == "response.created":
            # The greeting was accepted (or another response started): no fallback needed
            self._greeting_out_of_band_pending = False

        elif msg_type == "error":
            err = data.get("error", {}) or {}
            if (
                self._greeting_out_of_band_pending
                and err.get("param") == "response.conversation"
            ):
                # Gateway rejects out-of-band responses; resend the greeting in-conversation
                self._logger.warning(
                    "Out-of-band greeting rejected, resending in conversation",
                    error=err.get("message")
                )
                await self._send_greeting(out_of_band=False)
                return
            err_message = self._format_error_message(err)
            # Surface to connect() so it fails fast instead of timing out
            if not self._session_updated_event.is_set():
                self._connect_error = err_message
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.ERROR,
                    error=err_message,
                    timestamp=time.time()
                )
            )
            self._logger.error("OpenAI error", error=err)

        else:
            # Log unhandled events for debugging
            self._logger.debug(f"Unhandled event: {msg_type}", data=data)

    def _format_error_message(self, err: Dict) -> str:
        """Return a useful error string without logging credentials."""
        message = err.get("message") or "unknown error"
        code = err.get("code")
        err_type = err.get("type")

        parts = [message]
        details = ", ".join(str(value) for value in (err_type, code) if value)
        if details:
            parts.append(f"({details})")

        if code == "mismatched_project":
            parts.append(
                "OPENAI_PROJECT does not match the project that owns OPENAI_API_KEY. "
                "Use an API key created in that project, or set OPENAI_PROJECT to the key's project."
            )
        elif code == "model_not_found" or "does not exist or you do not have access" in message:
            parts.append(
                "Check that OPENAI_API_KEY belongs to the project with access to "
                f"{self._model}, or set OPENAI_PROJECT / OPENAI_ORGANIZATION for that scope."
            )

        return " ".join(parts)
