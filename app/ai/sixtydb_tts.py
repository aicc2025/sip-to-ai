"""60db Text-to-Speech WebSocket client (voice-only "speak" engine).

This is NOT a standalone voice agent. 60db has no LLM, so it cannot listen,
think and reply on its own. Instead it is used as the *voice* of the Deepgram
Voice Agent:

    Deepgram (STT + LLM/think + VAD/turn-taking) --emits agent text-->
        ConversationText --> SixtyDBTTSClient.speak(text) --> mulaw audio
            --> bridge --> caller

So Deepgram is the brain and 60db is the mouth.

API Documentation:
- WebSocket: wss://api.60db.ai/ws/tts?apiKey=<API_KEY>
- Auth: ``apiKey`` query parameter (NOT an Authorization header)
- Audio output: MULAW @ 8kHz (telephony-native, same as Deepgram/G.711)
- Protocol: JSON control messages; audio returned as base64 in ``audio_chunk``

Connection sequence:
1. connect -> server sends ``connection_established``
2. client sends ``create_context`` -> server replies ``context_created``
3. client sends ``send_text`` then ``flush_context`` to synthesize
4. server streams ``audio_chunk`` messages, then a ``flush_completed``
"""

import asyncio
import base64
import json
from typing import Awaitable, Callable, Optional

import structlog
import websockets
from websockets.client import WebSocketClientProtocol


class SixtyDBTTSClient:
    """60db TTS WebSocket client: text -> mulaw @ 8kHz audio.

    Audio is delivered out-of-band via the ``on_audio`` callback as raw 8kHz
    mu-law bytes (decoded from the base64 ``audio_chunk`` field). ``on_flush_complete``
    fires when 60db signals an utterance has been fully synthesized.
    """

    WS_URL = "wss://api.60db.ai/ws/tts"
    # 60db's documented default voice; overridable via SIXTYDB_VOICE_ID.
    DEFAULT_VOICE_ID = "fbb75ed2-975a-40c7-9e06-38e30524a9a1"

    def __init__(
        self,
        api_key: str,
        voice_id: Optional[str] = None,
        sample_rate: int = 8000,
        speed: float = 1.0,
        stability: int = 50,
        similarity: int = 75,
        on_audio: Optional[Callable[[bytes], Awaitable[None]]] = None,
        on_flush_complete: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """Initialize the 60db TTS client.

        Args:
            api_key: 60db API key (``sk_live_...``)
            voice_id: 60db voice UUID (defaults to 60db's default voice)
            sample_rate: Output sample rate (must be 8000 for MULAW)
            speed: Speech rate, 0.5-2.0
            stability: 0-100 (lower = more expressive)
            similarity: 0-100 (voice matching fidelity)
            on_audio: Async callback invoked with raw mu-law bytes per chunk
            on_flush_complete: Async callback invoked when an utterance is done
        """
        if not api_key:
            raise ValueError("60db API key is required")
        if sample_rate != 8000:
            raise ValueError(f"60db TTS (MULAW) only supports 8kHz, got: {sample_rate}")

        self._api_key = api_key
        self._voice_id = voice_id or self.DEFAULT_VOICE_ID
        self._sample_rate = sample_rate
        self._speed = speed
        self._stability = stability
        self._similarity = similarity
        self._on_audio = on_audio
        self._on_flush_complete = on_flush_complete

        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        # Single persistent synthesis context for the whole call.
        self._context_id = "sip-to-ai"
        self._context_ready = asyncio.Event()
        self._receive_task: Optional[asyncio.Task] = None

        self._logger = structlog.get_logger(__name__)

    @property
    def is_connected(self) -> bool:
        """Whether the TTS WebSocket is connected and the context is ready."""
        return self._connected

    async def connect(self) -> None:
        """Open the WebSocket and initialize the synthesis context."""
        if self._connected:
            return

        url = f"{self.WS_URL}?apiKey={self._api_key}"
        self._logger.info("Connecting to 60db TTS", url=self.WS_URL, voice_id=self._voice_id)

        async with asyncio.timeout(10.0):
            self._ws = await websockets.connect(url, open_timeout=10.0)

        self._connected = True

        # Start receiver first so connection_established / context_created are caught.
        self._receive_task = asyncio.create_task(
            self._receive_messages(), name="sixtydb-tts-receive"
        )

        await self._send_create_context()

        try:
            await asyncio.wait_for(self._context_ready.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            self._logger.error("Timeout waiting for 60db context_created")
            raise

        self._logger.info("60db TTS ready", context_id=self._context_id)

    async def _send_create_context(self) -> None:
        """Send the create_context message that configures the voice/audio."""
        if not self._ws:
            return
        msg = {
            "create_context": {
                "context_id": self._context_id,
                "voice_id": self._voice_id,
                "audio_config": {
                    "audio_encoding": "MULAW",
                    "sample_rate_hertz": self._sample_rate,
                },
                "speed": self._speed,
                "stability": self._stability,
                "similarity": self._similarity,
            }
        }
        await self._ws.send(json.dumps(msg))

    async def speak(self, text: str) -> None:
        """Synthesize ``text`` to speech. Audio arrives via the on_audio callback.

        Sends ``send_text`` followed by ``flush_context`` so 60db starts
        streaming audio immediately for this utterance.
        """
        if not self._ws or not self._connected:
            self._logger.warning("Cannot speak: 60db TTS not connected")
            return
        if not text or not text.strip():
            return

        await self._ws.send(
            json.dumps({"send_text": {"context_id": self._context_id, "text": text}})
        )
        await self._ws.send(
            json.dumps({"flush_context": {"context_id": self._context_id}})
        )
        self._logger.info("Sent text to 60db TTS", chars=len(text))

    async def _receive_messages(self) -> None:
        """Receive and dispatch JSON messages from 60db."""
        if not self._ws:
            return
        try:
            async for message in self._ws:
                # 60db TTS communicates exclusively via JSON (audio is base64).
                if isinstance(message, str):
                    await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            self._logger.info("60db TTS connection closed")
            self._connected = False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.error("60db TTS receive error", error=str(e))

    async def _handle_message(self, message: str) -> None:
        """Parse one JSON message and route it to the appropriate handler."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            self._logger.warning("60db TTS sent non-JSON message")
            return

        if "connection_established" in data:
            info = data["connection_established"]
            self._logger.info(
                "60db TTS connection established",
                workspace=info.get("workspace"),
                credit_balance=info.get("credit_balance"),
            )
        elif "context_created" in data:
            self._context_ready.set()
        elif "audio_chunk" in data:
            b64 = data["audio_chunk"].get("audioContent", "")
            if b64 and self._on_audio:
                ulaw_bytes = base64.b64decode(b64)
                await self._on_audio(ulaw_bytes)
        elif "flush_completed" in data:
            if self._on_flush_complete:
                await self._on_flush_complete()
        elif "context_closed" in data:
            self._logger.info("60db TTS context closed")
        elif "error" in data:
            err = data["error"]
            self._logger.error("60db TTS error", message=err.get("message"))

    async def close(self) -> None:
        """Close the synthesis context and the WebSocket."""
        if self._ws is None:
            return

        self._connected = False
        try:
            try:
                await self._ws.send(
                    json.dumps({"close_context": {"context_id": self._context_id}})
                )
            except Exception:
                pass

            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass

            await self._ws.close()
        except Exception as e:
            self._logger.error("Error closing 60db TTS", error=str(e))
        finally:
            self._ws = None
            self._context_ready.clear()
