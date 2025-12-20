"""Deepgram Voice Agent client implementation.

WebSocket-only implementation using standard websockets library.
No Deepgram SDK dependency required.

API Documentation:
- WebSocket: wss://agent.deepgram.com/v1/agent/converse
- Authentication: Authorization: token <API_KEY>
- Audio: supports mulaw (μ-law) @ 8kHz
- Protocol: JSON + binary audio chunks
"""

import asyncio
import json
from typing import AsyncIterator, Dict, Optional

import structlog
import websockets
from websockets.client import WebSocketClientProtocol

from app.ai.duplex_base import AiDuplexBase, AiEvent, AiEventType


class DeepgramAgentClient(AiDuplexBase):
    """Deepgram Voice Agent client using WebSocket only."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 8000,
        frame_ms: int = 20,
        audio_format: str = "mulaw",
        listen_model: str = "nova-2",
        speak_model: str = "aura-asteria-en",
        llm_model: str = "gpt-4o-mini",
        instructions: str = "You are a helpful voice assistant.",
        greeting: Optional[str] = None
    ) -> None:
        """Initialize Deepgram Voice Agent client.

        Args:
            api_key: Deepgram API key
            sample_rate: Audio sample rate (must be 8000 for mulaw)
            frame_ms: Frame duration in milliseconds
            audio_format: Audio format (mulaw for μ-law encoding)
            listen_model: STT model (nova-2, nova-3)
            speak_model: TTS voice model
            llm_model: LLM model for agent
            instructions: Agent instructions/system prompt
            greeting: Optional greeting message spoken at call start
        """
        # Initialize base class (same pattern as OpenAI client)
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms)

        if not api_key:
            raise ValueError("Deepgram API key is required")

        self._api_key = api_key
        self._audio_format = audio_format
        self._listen_model = listen_model
        self._speak_model = speak_model
        self._llm_model = llm_model
        self._instructions = instructions
        self._greeting = greeting

        # Override frame size for mulaw (1 byte per sample, same as G.711)
        if audio_format == "mulaw":
            self._frame_size = (sample_rate * frame_ms) // 1000  # mulaw = 1 byte per sample

        self._ws_url = "wss://agent.deepgram.com/v1/agent/converse"
        self._ws: Optional[WebSocketClientProtocol] = None

        # Event queues (same pattern as OpenAI client)
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue(maxsize=100)

        # Background task for receiving messages
        self._receive_task: Optional[asyncio.Task] = None

        # Settings ready flag - wait for SettingsApplied before sending audio
        self._settings_ready = asyncio.Event()

        # Agent speaking flag - track when AI is speaking to prevent barge-in
        self._agent_speaking = False
        self._last_agent_audio_time = 0.0  # Timestamp of last agent audio
        self._received_first_audio = False  # Track if we've received any AI audio yet

        # KeepAlive task - send periodic KeepAlive messages
        self._keepalive_task: Optional[asyncio.Task] = None

        self._logger = structlog.get_logger(__name__)

        # Validate configuration
        if audio_format != "mulaw":
            raise ValueError(f"Deepgram only supports mulaw format, got: {audio_format}")
        if sample_rate != 8000:
            raise ValueError(f"Deepgram only supports 8kHz sample rate, got: {sample_rate}")

    async def connect(self) -> None:
        """Connect to Deepgram Voice Agent."""
        if self._connected:
            self._logger.warning("Already connected")
            return

        try:
            self._logger.info("Connecting to Deepgram Voice Agent", url=self._ws_url)

            # Connect with authentication header and timeout
            async with asyncio.timeout(10.0):
                self._ws = await websockets.connect(
                    self._ws_url,
                    additional_headers={"Authorization": f"token {self._api_key}"},
                    open_timeout=10.0  # WebSocket-level timeout
                )

            self._connected = True
            self._logger.info("Connected to Deepgram Voice Agent")

            # Send session configuration
            await self._send_session_config()

            # Start background task to receive messages
            self._receive_task = asyncio.create_task(
                self._receive_messages(),
                name="deepgram-receive"
            )
            self._logger.info("Started background message receiver task")

            # Start KeepAlive task to prevent connection timeout
            self._keepalive_task = asyncio.create_task(
                self._send_keepalive(),
                name="deepgram-keepalive"
            )
            self._logger.info("Started KeepAlive task")

            # Emit connected event
            await self._event_queue.put(AiEvent(
                type=AiEventType.CONNECTED,
                data={"status": "connected"}
            ))

        except Exception as e:
            self._connected = False
            self._logger.error("Failed to connect to Deepgram", error=str(e))
            await self._event_queue.put(AiEvent(
                type=AiEventType.ERROR,
                data={"error": str(e)}
            ))
            raise

    async def _send_session_config(self) -> None:
        """Send session configuration to Deepgram."""
        if not self._ws:
            return

        agent_config: Dict = {
            "language": "en",
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": self._listen_model
                }
            },
            "think": {
                "provider": {
                    "type": "open_ai",
                    "model": self._llm_model
                },
                "prompt": self._instructions
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": self._speak_model
                }
            }
        }

        if self._greeting:
            agent_config["greeting"] = self._greeting

        config = {
            "type": "Settings",
            "audio": {
                "input": {
                    "encoding": self._audio_format,
                    "sample_rate": self._sample_rate
                },
                "output": {
                    "encoding": self._audio_format,
                    "sample_rate": self._sample_rate,
                    "container": "none"
                }
            },
            "agent": agent_config
        }

        await self._ws.send(json.dumps(config))
        self._logger.info(
            "Sent Settings to Deepgram",
            listen_model=self._listen_model,
            speak_model=self._speak_model,
            llm_model=self._llm_model
        )

    async def close(self) -> None:
        """Close connection to Deepgram Voice Agent."""
        if self._ws is None:
            return

        self._connected = False
        try:
            self._audio_queue.put_nowait(b"")
        except asyncio.QueueFull:
            pass

        try:
            self._logger.info("Disconnecting from Deepgram")

            # Cancel background tasks
            if self._keepalive_task and not self._keepalive_task.done():
                self._keepalive_task.cancel()
                try:
                    await self._keepalive_task
                except asyncio.CancelledError:
                    pass

            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass

            await self._ws.close()
            self._ws = None

            await self._event_queue.put(AiEvent(
                type=AiEventType.DISCONNECTED,
                data={"status": "disconnected"}
            ))

        except Exception as e:
            self._logger.error("Error during disconnect", error=str(e))

    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to Deepgram.

        Converts PCM16 → mulaw before sending.

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes/20ms)
        """
        if not self._ws:
            self._logger.warning("Cannot send audio: not connected")
            return

        # Wait for SettingsApplied before sending audio
        if not self._settings_ready.is_set():
            try:
                await asyncio.wait_for(self._settings_ready.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._logger.error("Timeout waiting for SettingsApplied")
                return

        try:
            # Don't send any audio until we receive first AI audio
            # This prevents triggering Deepgram's VAD before greeting starts
            if not self._received_first_audio:
                return

            # Skip sending audio while agent is speaking (prevent barge-in)
            import time
            current_time = time.time()
            time_since_last_audio = current_time - self._last_agent_audio_time

            if self._agent_speaking or time_since_last_audio < 2.0:
                return

            # Convert PCM16 → mulaw
            from app.utils.codec import Codec
            mulaw_chunk = Codec.pcm16_to_ulaw(frame_20ms)

            # Send raw binary audio (μ-law bytes) directly to Deepgram
            await self._ws.send(mulaw_chunk)

        except Exception as e:
            self._logger.error("Failed to send audio", error=str(e))
            await self._event_queue.put(AiEvent(
                type=AiEventType.ERROR,
                data={"error": f"Send audio failed: {e}"}
            ))

    async def _send_keepalive(self) -> None:
        """Send periodic KeepAlive messages to prevent connection timeout.

        Deepgram requires audio or KeepAlive within 10 seconds.
        """
        if not self._ws:
            return

        try:
            while self._connected:
                if self._ws and self._connected:
                    keepalive_msg = {"type": "KeepAlive"}
                    await self._ws.send(json.dumps(keepalive_msg))

                await asyncio.sleep(5.0)

        except asyncio.CancelledError:
            self._logger.debug("KeepAlive task cancelled")
            # Expected during close(), no need to propagate
        except Exception as e:
            self._logger.error("Error in KeepAlive task", error=str(e))

    async def _receive_messages(self) -> None:
        """Receive messages from Deepgram WebSocket."""
        if not self._ws:
            return

        try:
            async for message in self._ws:
                if isinstance(message, str):
                    await self._handle_json_message(message)
                elif isinstance(message, bytes):
                    # Deepgram sends binary μ-law audio (typically 960 bytes = 120ms)
                    # Split into 20ms frames (160 bytes each) for AudioAdapter
                    await self._handle_binary_audio(message)

        except websockets.exceptions.ConnectionClosed:
            self._logger.info("Deepgram connection closed")
            self._connected = False
            event = AiEvent(
                type=AiEventType.DISCONNECTED,
                data={"status": "disconnected"}
            )
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                self._logger.debug("Event queue full, dropping disconnect event")
            try:
                self._audio_queue.put_nowait(b"")
            except asyncio.QueueFull:
                pass
        except Exception as e:
            self._logger.error("Error receiving messages", error=str(e))
            await self._event_queue.put(AiEvent(
                type=AiEventType.ERROR,
                data={"error": str(e)}
            ))

    async def _handle_binary_audio(self, audio_data: bytes) -> None:
        """Handle binary audio message from Deepgram.

        Deepgram sends binary μ-law audio in variable-size chunks (typically 960 bytes).
        Convert to PCM16 and yield as variable-size chunk.

        Frame splitting and padding is handled by AudioAdapter.feed_ai_audio().

        Args:
            audio_data: Binary μ-law audio data from Deepgram
        """
        import time
        from app.utils.codec import Codec

        # Mark that we've received first audio - now safe to send user audio
        if not self._received_first_audio:
            self._received_first_audio = True
            self._logger.info("Received first AI audio - enabling user audio")

        # Mark agent as speaking and update timestamp
        self._agent_speaking = True
        self._last_agent_audio_time = time.time()

        # Convert μ-law to PCM16 (variable-size chunk)
        pcm16_chunk = Codec.ulaw_to_pcm16(audio_data)

        # Send entire chunk to AudioAdapter (it will handle frame splitting)
        await self._audio_queue.put(pcm16_chunk)

    async def _handle_json_message(self, message: str) -> None:
        """Handle JSON message from Deepgram.

        Args:
            message: JSON message string
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "UserStartedSpeaking":
                await self._event_queue.put(AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"event": "user_started_speaking"}
                ))

            elif msg_type == "AgentStartedSpeaking":
                await self._event_queue.put(AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"event": "agent_started_speaking"}
                ))

            elif msg_type == "AgentAudioDone":
                self._agent_speaking = False
                await self._event_queue.put(AiEvent(
                    type=AiEventType.TRANSCRIPT_FINAL,
                    data={"event": "agent_audio_done"}
                ))

            elif msg_type == "Error":
                error_msg = data.get("message", "Unknown error")
                error_code = data.get("code", "unknown")
                self._logger.error(
                    "Deepgram error",
                    error=error_msg,
                    code=error_code
                )
                await self._event_queue.put(AiEvent(
                    type=AiEventType.ERROR,
                    data={"error": error_msg, "code": error_code}
                ))

            elif msg_type == "SettingsApplied":
                self._settings_ready.set()
                self._logger.info("Settings applied")

            elif msg_type == "Welcome":
                self._logger.info("Connected to Deepgram Voice Agent")

            elif msg_type == "ConversationText":
                # Log conversation for debugging (optional)
                pass

            elif msg_type == "History":
                # Conversation history (optional)
                pass

        except Exception as e:
            self._logger.error("Failed to handle JSON message", error=str(e))

    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Receive audio chunks from Deepgram.

        Yields:
            PCM16 audio chunks @ 8kHz (320 bytes/20ms frames)
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
        """Iterate over events from Deepgram.

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
            config: Session configuration dictionary (Deepgram agent settings)
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        message = {
            "type": "Settings",
            **config
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

    async def run(self) -> None:
        """Run the Deepgram client main loop."""
        await self.connect()
        await self._receive_messages()
