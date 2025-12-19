"""Gemini Live API adapter.

Bidirectional audio streaming with Google's Gemini 2.5 Flash model:

1. WebSocket connection to Gemini Live API
2. Session configuration with voice settings
3. Audio streaming with resampling (8kHz SIP <-> 16kHz/24kHz Gemini)
4. Event handling for transcription and errors

Audio Flow:
- Input: PCM16 @ 8kHz -> resample to PCM16 @ 16kHz -> base64 -> Gemini
- Output: Gemini -> PCM16 @ 24kHz -> resample to PCM16 @ 8kHz

Note: Gemini Live does not support G.711/mulaw natively, so resampling is required.
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
from app.utils.codec import resample_pcm16


class GeminiLiveClient(AiDuplexBase):
    """Gemini Live API client for bidirectional audio streaming."""

    # Gemini Live API WebSocket endpoint
    WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

    # Audio sample rates
    SIP_SAMPLE_RATE = 8000      # SIP uses 8kHz
    GEMINI_INPUT_RATE = 16000   # Gemini expects 16kHz input
    GEMINI_OUTPUT_RATE = 24000  # Gemini outputs 24kHz

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-native-audio-preview-12-2025",
        voice: str = "Puck",
        instructions: str = "You are a helpful assistant.",
        greeting: Optional[str] = None
    ) -> None:
        """Initialize Gemini Live client.

        Args:
            api_key: Google AI API key (falls back to GEMINI_API_KEY env var)
            model: Gemini model to use (must support Live API)
            voice: Voice for speech synthesis (Puck, Charon, Kore, Fenrir, Aoede)
            instructions: System instructions for the AI
            greeting: Optional greeting message to speak when session starts
        """
        super().__init__(sample_rate=self.SIP_SAMPLE_RATE, frame_ms=20)

        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("Gemini API key not provided")

        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._greeting = greeting
        self._ws: Optional[WebSocketClientProtocol] = None

        # Event queues
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue(maxsize=100)

        # Control
        self._stop_event = asyncio.Event()
        self._setup_complete_event = asyncio.Event()
        self._message_handler_task: Optional[asyncio.Task[None]] = None

        # Stats
        self._audio_frames_sent = 0
        self._audio_chunks_received = 0

        # Transcription buffers (accumulate before logging)
        self._user_transcript_buffer = ""
        self._ai_transcript_buffer = ""

        self._logger = structlog.get_logger(__name__)

    async def connect(self) -> None:
        """Connect to Gemini Live API."""
        if self._connected:
            return

        try:
            # Build WebSocket URL with API key
            ws_url = f"{self.WS_URL}?key={self._api_key}"

            # Connect WebSocket with timeout
            async with asyncio.timeout(10.0):
                self._ws = await websockets.connect(
                    ws_url,
                    open_timeout=10.0
                )

            self._connected = True
            self._stop_event.clear()
            self._setup_complete_event.clear()

            # Start message handler task
            self._message_handler_task = asyncio.create_task(
                self._message_handler(),
                name="gemini-message-handler"
            )

            # Send session setup message
            await self._send_setup()

            # Wait for setup complete
            self._logger.info("Waiting for setup complete from Gemini...")
            async with asyncio.timeout(10.0):
                await self._setup_complete_event.wait()

            self._logger.info(
                "Gemini Live connected",
                model=self._model,
                voice=self._voice
            )

            # Send greeting if configured
            if self._greeting:
                await self._send_greeting()

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Gemini Live: {e}")

    async def close(self) -> None:
        """Close connection."""
        if not self._connected:
            return

        self._connected = False
        self._stop_event.set()

        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                self._logger.debug("Message handler task cancelled")

        if self._ws:
            await self._ws.close()

        self._logger.info("Gemini Live disconnected")

    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to Gemini.

        Resamples 8kHz to 16kHz before sending.

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes)
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        # Validate input: 320 bytes = 160 samples @ 8kHz = 20ms
        if len(frame_20ms) != 320:
            raise ValueError(f"Expected 320 bytes PCM16 @ 8kHz, got {len(frame_20ms)}")

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

        # Send realtime input message with base64-encoded audio
        message = {
            "realtimeInput": {
                "mediaChunks": [{
                    "mimeType": "audio/pcm;rate=16000",
                    "data": base64.b64encode(pcm16_16k).decode("utf-8")
                }]
            }
        }

        await self._ws.send(json.dumps(message))

        self._audio_frames_sent += 1
        if self._audio_frames_sent % 50 == 0:  # Log every 1 second
            self._logger.info(f"Sent {self._audio_frames_sent} audio frames to Gemini")

    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Receive audio chunks from Gemini.

        Yields:
            PCM16 audio chunks @ 8kHz (resampled from 24kHz)
        """
        while self._connected:
            try:
                chunk = await self._audio_queue.get()
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

        Returns:
            True if healthy
        """
        if not self._connected or not self._ws:
            return False

        try:
            await self._ws.ping()
            return True
        except Exception:
            return False

    async def reconnect(self) -> None:
        """Reconnect to service."""
        await self.close()
        await asyncio.sleep(1.0)
        await self.connect()

    async def _send_setup(self) -> None:
        """Send initial session setup message."""
        setup_message = {
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
                "outputAudioTranscription": {}
            }
        }

        self._logger.info(
            "Sending Gemini setup",
            model=self._model,
            voice=self._voice,
            instructions_length=len(self._instructions)
        )

        await self._ws.send(json.dumps(setup_message))

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

    async def _message_handler(self) -> None:
        """Handle WebSocket messages from Gemini."""
        if not self._ws:
            return

        while not self._stop_event.is_set():
            try:
                message = await self._ws.recv()
                data = json.loads(message)

                await self._process_message(data)

            except websockets.exceptions.ConnectionClosed:
                self._logger.warning("WebSocket connection closed")
                await self._event_queue.put(
                    AiEvent(
                        type=AiEventType.DISCONNECTED,
                        timestamp=time.time()
                    )
                )
                break
            except json.JSONDecodeError as e:
                self._logger.error("Failed to decode message", error=str(e))
            except Exception as e:
                self._logger.error("Message handler error", error=str(e))

    async def _process_message(self, data: Dict) -> None:
        """Process WebSocket message from Gemini.

        Args:
            data: Parsed JSON message
        """
        # Check for setup complete
        if "setupComplete" in data:
            self._setup_complete_event.set()
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.CONNECTED,
                    data=data.get("setupComplete"),
                    timestamp=time.time()
                )
            )
            self._logger.info("Gemini setup complete")
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
                    await self._event_queue.put(
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
                    await self._event_queue.put(
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

            return

        # Check for tool calls
        if "toolCall" in data:
            self._logger.info("Tool call received", data=data["toolCall"])
            return

        # Check for go away (disconnection notice)
        if "goAway" in data:
            self._logger.warning("Received goAway from Gemini", data=data["goAway"])
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.DISCONNECTED,
                    data=data["goAway"],
                    timestamp=time.time()
                )
            )
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
        await self._audio_queue.put(pcm16_8k)
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
