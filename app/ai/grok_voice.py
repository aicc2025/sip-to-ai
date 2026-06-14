"""xAI Grok Voice realtime API adapter.

Patterned on OpenAIRealtimeClient. The Grok Voice realtime protocol mirrors
OpenAI Realtime almost 1:1 (event names, handshake, audio buffer model), and
natively supports G.711 mu-law @ 8kHz so no resampling is needed.

Audio Flow:
- Input:  PCM16 @ 8kHz -> G.711 mu-law @ 8kHz -> Grok
- Output: Grok -> G.711 mu-law @ 8kHz -> PCM16 @ 8kHz

Endpoint: wss://api.x.ai/v1/realtime?model=<model>
Auth:     Authorization: Bearer <XAI_API_KEY>
"""

import asyncio
import base64
import json
import os
import time
import uuid
from typing import AsyncIterator, Dict, Optional

import structlog
import websockets
from websockets.client import WebSocketClientProtocol

from app.ai.duplex_base import AiDuplexBase, AiEvent, AiEventType
from app.utils.codec import Codec


class GrokVoiceClient(AiDuplexBase):
    """xAI Grok Voice realtime API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-voice-think-fast-1.0",
        voice: str = "eve",
        instructions: str = "You are a helpful assistant.",
        greeting: Optional[str] = None,
        ws_endpoint: str = "wss://api.x.ai/v1/realtime",
    ) -> None:
        """Initialize Grok Voice client.

        Args:
            api_key: xAI API key (falls back to XAI_API_KEY env var).
            model: Grok voice model (e.g. grok-voice-think-fast-1.0).
            voice: Built-in voice (eve, ara, leo, rex, sal) or custom 8-char id.
            instructions: System prompt for the agent.
            greeting: Optional greeting played when the call connects.
            ws_endpoint: WebSocket endpoint (overridable for testing).
        """
        self._sip_sample_rate = 8000
        self._grok_sample_rate = 8000
        self._frame_ms = 20
        self._sample_rate = self._sip_sample_rate

        super().__init__(self._sample_rate, self._frame_ms)

        self._api_key = api_key or os.getenv("XAI_API_KEY")
        if not self._api_key:
            raise ValueError("Grok API key not provided")

        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._greeting = greeting
        self._ws: Optional[WebSocketClientProtocol] = None
        self._ws_url = ws_endpoint

        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._event_queue: asyncio.Queue[AiEvent] = asyncio.Queue(maxsize=100)

        self._connected = False
        self._stop_event = asyncio.Event()
        self._session_created_event = asyncio.Event()
        self._session_updated_event = asyncio.Event()
        self._message_handler_task: Optional[asyncio.Task[None]] = None

        self._audio_frames_sent = 0
        self._audio_chunks_received = 0

        self._logger = structlog.get_logger(__name__)

    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to Grok.

        Converts PCM16 -> G.711 mu-law @ 8kHz before sending.

        Args:
            frame_20ms: PCM16 audio frame @ 8kHz (320 bytes).
        """
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")

        if len(frame_20ms) != 320:
            raise ValueError(f"Expected 320 bytes PCM16 @ 8kHz, got {len(frame_20ms)}")

        g711_ulaw = Codec.pcm16_to_ulaw(frame_20ms)

        message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(g711_ulaw).decode("utf-8"),
        }
        await self._ws.send(json.dumps(message))

        self._audio_frames_sent += 1
        if self._audio_frames_sent % 50 == 0:
            self._logger.info("📤 Sent audio frames to Grok", frames=self._audio_frames_sent)

    async def _process_message(self, data: Dict) -> None:
        """Dispatch a single Grok server event."""
        msg_type = data.get("type")
        self._logger.debug("Received Grok event", msg_type=msg_type)

        if msg_type in ("session.created", "conversation.created"):
            # xAI's live realtime API emits "conversation.created" as the
            # connection-ready signal. The doc summary listed "session.created"
            # too, but the live server only sends "conversation.created" in
            # current versions. Accept either as the ready signal so we tolerate
            # future protocol updates without a code change.
            self._session_created_event.set()
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.CONNECTED,
                    data=data.get("session") or data.get("conversation"),
                    timestamp=time.time(),
                )
            )

        elif msg_type == "ping":
            # Application-level keepalive from xAI server; no response needed.
            self._logger.debug("Grok ping")

        elif msg_type == "session.updated":
            self._session_updated_event.set()
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.SESSION_UPDATED,
                    data=data.get("session", {}),
                    timestamp=time.time(),
                )
            )

        elif msg_type == "input_audio_buffer.speech_started":
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"event": "speech_started"},
                    timestamp=time.time(),
                )
            )

        elif msg_type == "input_audio_buffer.speech_stopped":
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_PARTIAL,
                    data={"event": "speech_stopped"},
                    timestamp=time.time(),
                )
            )

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = data.get("transcript")
            self._logger.info("✅ User transcript", text=transcript)
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.TRANSCRIPT_FINAL,
                    data={"text": transcript},
                    timestamp=time.time(),
                )
            )

        elif msg_type == "response.output_audio.delta":
            audio_b64 = data.get("delta")
            if audio_b64:
                ulaw = base64.b64decode(audio_b64)
                pcm16 = Codec.ulaw_to_pcm16(ulaw)
                await self._audio_queue.put(pcm16)
                self._audio_chunks_received += 1
                if self._audio_chunks_received % 10 == 0:
                    self._logger.info(
                        "📢 Received audio chunks",
                        chunks=self._audio_chunks_received,
                        ulaw_bytes=len(ulaw),
                        pcm16_bytes=len(pcm16),
                    )

        elif msg_type in ("response.output_audio_transcript.delta", "response.output_audio_transcript.done"):
            self._logger.info("🤖 AI transcript", **{k: data.get(k) for k in ("delta", "transcript") if k in data})

        elif msg_type == "response.done":
            self._logger.debug("Grok response.done")

        elif msg_type == "error":
            err = data.get("error", {})
            await self._event_queue.put(
                AiEvent(
                    type=AiEventType.ERROR,
                    error=err.get("message"),
                    timestamp=time.time(),
                )
            )
            self._logger.error("Grok error", error=err)

        else:
            self._logger.debug("Unhandled Grok event", msg_type=msg_type, data=data)

    async def _configure_session(self) -> None:
        """Send the initial session.update payload.

        Schema verified against https://docs.x.ai/voice-realtime.ws.json on
        2026-05-04: audio is at session.audio.{input,output}.format.{type,rate};
        type "audio/pcmu" is G.711 μ-law; system prompt field is "instructions"
        (NOT "system_prompt"). Default rate is 24000 — explicit 8000 is required
        for telephony.
        """
        if not self._ws:
            return

        config = {
            "type": "session.update",
            "session": {
                "model": self._model,
                "voice": self._voice,
                "instructions": self._instructions,
                "audio": {
                    "input": {"format": {"type": "audio/pcmu", "rate": 8000}},
                    "output": {"format": {"type": "audio/pcmu", "rate": 8000}},
                },
                "turn_detection": {"type": "server_vad"},
            },
        }
        self._logger.info(
            "Configuring Grok session",
            model=self._model,
            voice=self._voice,
            has_greeting=self._greeting is not None,
            instructions_length=len(self._instructions),
        )
        await self._ws.send(json.dumps(config))

    async def _send_greeting(self) -> None:
        """Send greeting response.create after session.updated."""
        if not self._ws or not self._greeting:
            return

        # xAI's RealtimeClientEvent schema requires metadata.client_event_id on
        # response.create. Without it the server returns "invalid_event" and the
        # greeting never plays. uuid4 is sufficient — the value just needs to be
        # unique per request for client-side correlation.
        message = {
            "type": "response.create",
            "response": {
                "instructions": self._greeting,
                "conversation": "none",
                "output_modalities": ["audio"],
                "metadata": {
                    "client_event_id": str(uuid.uuid4()),
                    "response_purpose": "greeting",
                },
            },
        }
        await self._ws.send(json.dumps(message))
        self._logger.info("Greeting request sent", greeting_preview=self._greeting[:50])

    async def connect(self) -> None:
        """Connect to Grok Voice WebSocket and complete the handshake."""
        if self._connected:
            return

        try:
            headers = {"Authorization": f"Bearer {self._api_key}"}

            async with asyncio.timeout(10.0):
                self._ws = await websockets.connect(
                    f"{self._ws_url}?model={self._model}",
                    additional_headers=headers,
                    open_timeout=10.0,
                    proxy=None,  # Direct connect; don't auto-use env SOCKS proxy
                )

            self._connected = True
            self._stop_event.clear()
            self._session_created_event.clear()
            self._session_updated_event.clear()

            self._message_handler_task = asyncio.create_task(
                self._message_handler(),
                name="grok-message-handler",
            )

            self._logger.info("Waiting for session.created from Grok...")
            async with asyncio.timeout(5.0):
                await self._session_created_event.wait()

            self._logger.info("Configuring Grok session...")
            await self._configure_session()

            self._logger.info("Waiting for session.updated from Grok...")
            async with asyncio.timeout(5.0):
                await self._session_updated_event.wait()

            if self._greeting:
                await self._send_greeting()

            self._logger.info("Grok Voice connected", model=self._model, voice=self._voice)

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect: {e}") from e

    async def close(self) -> None:
        """Close the Grok WebSocket and cancel background tasks."""
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
                self._logger.debug("Message handler cancelled")

        if self._ws:
            await self._ws.close()

        self._logger.info("Grok Voice disconnected")

    async def _message_handler(self) -> None:
        """Read frames from the WebSocket and dispatch via _process_message."""
        if not self._ws:
            return

        while not self._stop_event.is_set():
            try:
                message = await self._ws.recv()
                data = json.loads(message)
                await self._process_message(data)
            except websockets.exceptions.ConnectionClosed:
                self._logger.warning("Grok WebSocket closed")
                self._connected = False
                self._stop_event.set()
                try:
                    self._event_queue.put_nowait(
                        AiEvent(type=AiEventType.DISCONNECTED, timestamp=time.time())
                    )
                except asyncio.QueueFull:
                    pass
                try:
                    self._audio_queue.put_nowait(b"")
                except asyncio.QueueFull:
                    pass
                break
            except Exception as e:
                self._logger.error("Grok message handler error", error=str(e))

    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM16 @ 8kHz audio chunks decoded from Grok."""
        while self._connected:
            try:
                chunk = await self._audio_queue.get()
                if not self._connected and chunk == b"":
                    break
                yield chunk
            except Exception as e:
                self._logger.error("Grok audio stream error", error=str(e))
                break

    async def events(self) -> AsyncIterator[AiEvent]:
        """Yield AI events (CONNECTED, ERROR, etc.)."""
        while self._connected:
            try:
                event = await self._event_queue.get()
                yield event
            except Exception as e:
                self._logger.error("Grok event stream error", error=str(e))
                break

    async def update_session(self, config: Dict) -> None:
        """Send a session.update with the given config payload."""
        if not self._connected or not self._ws:
            raise ConnectionError("Not connected")
        message = {"type": "session.update", "session": config}
        await self._ws.send(json.dumps(message))
        self._logger.info("Grok session updated")

    async def ping(self) -> bool:
        """Health check via WebSocket ping."""
        if not self._connected or not self._ws:
            return False
        try:
            pong_waiter = await self._ws.ping()
            await asyncio.wait_for(pong_waiter, timeout=5.0)
            return True
        except (asyncio.TimeoutError, Exception):
            return False

    async def reconnect(self) -> None:
        """Close and reconnect."""
        await self.close()
        await asyncio.sleep(1.0)
        await self.connect()
