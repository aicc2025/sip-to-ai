"""OpenAI Realtime API adapter.

Simplified integration using G.711 μ-law @ 8kHz (native OpenAI support):

1. WebSocket connection to wss://api.openai.com/v1/realtime
2. Session configuration with semantic VAD and barge-in
3. G.711 μ-law audio streaming (no resampling needed)
4. Event handling for transcription and errors

Audio Flow:
- Input: PCM16 @ 8kHz → G.711 μ-law @ 8kHz → OpenAI
- Output: OpenAI → G.711 μ-law @ 8kHz → PCM16 @ 8kHz

Session configuration (new schema):
{
    "type": "session.update",
    "session": {
        "type": "realtime",
        "model": "gpt-realtime",
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
from app.utils.codec import Codec


class OpenAIRealtimeClient(AiDuplexBase):
    """OpenAI Realtime API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-realtime",
        voice: str = "marin",
        instructions: str = "You are a helpful assistant.",
        greeting: Optional[str] = None
    ) -> None:
        """Initialize OpenAI Realtime client.

        Args:
            api_key: OpenAI API key
            model: Model to use
            voice: Voice for TTS
            instructions: System instructions/prompt for the AI
            greeting: Optional greeting message to play when call connects

        Note:
            - Uses G.711 μ-law @ 8kHz (native OpenAI support)
            - No resampling needed (same as SIP/Deepgram)
            - Direct passthrough from SIP → OpenAI → SIP
        """
        # OpenAI Realtime API configuration
        # Using G.711 μ-law (audio/pcmu) @ 8kHz - native OpenAI support
        self._sip_sample_rate = 8000  # SIP uses 8kHz
        self._openai_sample_rate = 8000  # OpenAI Realtime uses 8kHz for G.711 μ-law
        self._audio_format = "audio/pcmu"  # G.711 μ-law format
        self._frame_ms = 20
        self._sample_rate = self._sip_sample_rate  # Base class expects this

        super().__init__(self._sample_rate, self._frame_ms)

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key not provided")

        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._greeting = greeting
        self._ws: Optional[WebSocketClientProtocol] = None
        self._ws_url = "wss://api.openai.com/v1/realtime"

        # Event queues (using asyncio Queues)
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue(maxsize=100)

        # Control
        self._stop_event = asyncio.Event()
        self._session_created_event = asyncio.Event()
        self._session_updated_event = asyncio.Event()
        self._message_handler_task: Optional[asyncio.Task[None]] = None

        # Stats for debugging
        self._audio_frames_sent = 0
        self._audio_chunks_received = 0

        self._logger = structlog.get_logger(__name__)

    async def connect(self) -> None:
        """Connect to OpenAI Realtime API."""
        if self._connected:
            return

        try:
            # Connect WebSocket with auth headers and timeout
            # Note: Don't use "OpenAI-Beta: realtime=v1" - that connects to old API
            headers = {
                "Authorization": f"Bearer {self._api_key}"
            }

            # Set connection timeout (10 seconds)
            async with asyncio.timeout(10.0):
                self._ws = await websockets.connect(
                    f"{self._ws_url}?model={self._model}",
                    additional_headers=headers,
                    open_timeout=10.0  # WebSocket-level timeout
                )

            self._connected = True
            self._stop_event.clear()
            self._session_created_event.clear()
            self._session_updated_event.clear()

            # Start message handler task first
            self._message_handler_task = asyncio.create_task(
                self._message_handler(),
                name="openai-message-handler"
            )

            # Wait for session.created from OpenAI
            self._logger.info("Waiting for session.created from OpenAI...")
            async with asyncio.timeout(5.0):
                await self._session_created_event.wait()

            self._logger.info("Received session.created, now configuring session...")

            # Configure session after receiving session.created
            await self._configure_session()

            # Wait for first session.updated from OpenAI
            self._logger.info("Waiting for session.updated from OpenAI...")
            async with asyncio.timeout(5.0):
                await self._session_updated_event.wait()

            self._logger.info("Received session.updated")

            # Send greeting after first session.updated (only once)
            if self._greeting:
                await self._send_greeting()
                self._logger.info("Sent greeting after session.updated")

            self._logger.info(
                "OpenAI Realtime connected",
                model=self._model,
                voice=self._voice
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

        Converts PCM16 → G.711 μ-law @ 8kHz before sending.

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes)
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        # Validate input: 320 bytes = 160 samples @ 8kHz = 20ms
        if len(frame_20ms) != 320:
            raise ValueError(f"Expected 320 bytes PCM16 @ 8kHz, got {len(frame_20ms)}")

        # Convert PCM16 → G.711 μ-law (320 bytes → 160 bytes)
        g711_ulaw = Codec.pcm16_to_ulaw(frame_20ms)

        # Log first few frames for debugging
        if self._audio_frames_sent < 3:
            import numpy as np
            samples_pcm16 = np.frombuffer(frame_20ms, dtype=np.int16)
            self._logger.info(
                f"📤 Frame #{self._audio_frames_sent + 1}",
                input_size=len(frame_20ms),
                output_size=len(g711_ulaw),
                expected_output=160,  # 160 bytes G.711
                pcm16_min=int(samples_pcm16.min()),
                pcm16_max=int(samples_pcm16.max())
            )

        # Send audio append message (base64 encoded as per OpenAI Realtime API spec)
        message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(g711_ulaw).decode("utf-8")
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

    async def _configure_session(self) -> None:
        """Configure initial session using new schema."""
        config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self._model,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {
                            "type": self._audio_format
                        },
                        "transcription": {
                            "model": "whisper-1"
                        },
                        "noise_reduction": {
                            "type": "near_field"
                        },
                        "turn_detection": {
                            "type": "semantic_vad",
                            "create_response": True,
                            "eagerness": "medium"
                        }
                    },
                    "output": {
                        "format": {
                            "type": self._audio_format
                        },
                        "voice": self._voice
                    }
                },
                "instructions": self._instructions
            }
        }

        self._logger.info(
            "Configuring OpenAI session (new schema)",
            audio_format=self._audio_format,
            input_sample_rate=self._openai_sample_rate,
            voice=self._voice,
            has_greeting=self._greeting is not None,
            instructions_length=len(self._instructions)
        )

        # Log the full config for debugging
        self._logger.debug(f"Session config: {json.dumps(config, indent=2)}")

        await self._ws.send(json.dumps(config))

    async def _send_greeting(self) -> None:
        """Send greeting message to OpenAI."""
        if not self._ws or not self._greeting:
            return

        greeting_request = {
            "type": "response.create",
            "response": {
                "instructions": self._greeting,
                "conversation": "none",
                "output_modalities": ["audio"],
                "metadata": {
                    "response_purpose": "greeting"
                }
            }
        }

        await self._ws.send(json.dumps(greeting_request))
        self._logger.info("Greeting request sent", greeting_preview=self._greeting[:50])

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
            # Audio chunk from AI (base64 encoded G.711 μ-law @ 8kHz)
            audio_base64 = data.get("delta")
            if audio_base64:
                # Decode base64 to get audio bytes
                audio_bytes = base64.b64decode(audio_base64)

                # Verify we're using the expected format
                if self._audio_format != "audio/pcmu":
                    self._logger.error(
                        "Unexpected audio format - expected audio/pcmu",
                        actual_format=self._audio_format,
                        chunk_size=len(audio_bytes)
                    )
                    return

                # G.711 μ-law @ 8kHz from OpenAI
                g711_ulaw = audio_bytes
                # Convert G.711 μ-law → PCM16 @ 8kHz for PJSUA2
                pcm16_8k = Codec.ulaw_to_pcm16(g711_ulaw)

                # Log chunk sizes
                chunk_size_g711 = len(g711_ulaw)
                chunk_size_pcm16 = len(pcm16_8k)
                duration_ms = (chunk_size_g711 / 8000) * 1000  # G.711 @ 8kHz: 1 byte = 1 sample

                await self._audio_queue.put(pcm16_8k)
                self._audio_chunks_received += 1

                if self._audio_chunks_received % 10 == 0:
                    self._logger.info(
                        f"📢 Received {self._audio_chunks_received} audio chunks (G.711 μ-law)",
                        g711_ulaw=f"{chunk_size_g711}B",
                        pcm16_8k=f"{chunk_size_pcm16}B",
                        duration=f"{duration_ms:.1f}ms"
                    )
                elif self._audio_chunks_received <= 5:
                    self._logger.info(
                        f"📢 Chunk #{self._audio_chunks_received} (G.711 μ-law)",
                        g711_ulaw=f"{chunk_size_g711}B",
                        pcm16_8k=f"{chunk_size_pcm16}B",
                        duration=f"{duration_ms:.1f}ms"
                    )

        elif msg_type == "error":
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.ERROR,
                    error=data.get("error", {}).get("message"),
                    timestamp=time.time()
                )
            )
            self._logger.error("OpenAI error", error=data.get("error"))

        else:
            # Log unhandled events for debugging
            self._logger.debug(f"Unhandled event: {msg_type}", data=data)
