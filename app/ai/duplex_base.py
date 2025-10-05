"""Base protocol and types for AI duplex communication."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator, Dict, Optional, Protocol, runtime_checkable


class AiEventType(Enum):
    """AI event types (simplified for connection management only)."""

    CONNECTED = auto()
    DISCONNECTED = auto()
    ERROR = auto()
    SESSION_UPDATED = auto()

    # Optional debug/logging events
    TRANSCRIPT_PARTIAL = auto()
    TRANSCRIPT_FINAL = auto()


@dataclass
class AiEvent:
    """AI event data."""

    type: AiEventType
    data: Optional[Dict] = None
    timestamp: float = 0.0
    error: Optional[str] = None


@runtime_checkable
class AiDuplexClient(Protocol):
    """Protocol for AI duplex client implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to AI service.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connection to AI service."""
        ...

    @abstractmethod
    async def send_pcm16_8k(self, frame_20ms: bytes) -> None:
        """Send PCM16 @ 8kHz audio frame to AI.

        Each implementation converts to its required format:
        - OpenAI: PCM16 8kHz → PCM16 24kHz
        - Deepgram: PCM16 8kHz → mulaw 8kHz
        - Mock: PCM16 8kHz (passthrough)

        Args:
            frame_20ms: 20ms PCM16 frame @ 8kHz (320 bytes)

        Raises:
            ConnectionError: If not connected
            ValueError: If frame size is invalid
        """
        ...

    @abstractmethod
    async def receive_chunks(self) -> AsyncIterator[bytes]:
        """Iterate over received audio chunks from AI.

        Yields:
            PCM16 audio chunks @ 8kHz (variable size)
            - OpenAI Realtime: Converts G.711 → PCM16 internally (typically 1600-4000 bytes)
            - Deepgram Agent: Returns PCM16 directly (variable size)
            - Mock: Returns PCM16 (320 bytes/20ms frames)

        Raises:
            ConnectionError: If connection is lost
        """
        ...

    @abstractmethod
    async def events(self) -> AsyncIterator[AiEvent]:
        """Iterate over events from AI.

        Yields:
            AI events

        Raises:
            ConnectionError: If connection is lost
        """
        ...

    @abstractmethod
    async def update_session(self, config: Dict) -> None:
        """Update session configuration.

        Args:
            config: Session configuration dictionary

        Raises:
            ConnectionError: If not connected
            ValueError: If configuration is invalid
        """
        ...

    @abstractmethod
    async def ping(self) -> bool:
        """Check connection health.

        Returns:
            True if connection is healthy

        Raises:
            ConnectionError: If connection check fails
        """
        ...

    @abstractmethod
    async def reconnect(self) -> None:
        """Reconnect to AI service.

        Raises:
            ConnectionError: If reconnection fails
        """
        ...


@dataclass
class SessionConfig:
    """Common session configuration."""

    # Audio configuration
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm16"

    # Voice configuration
    voice: Optional[str] = None
    language: str = "en-US"

    # Interaction configuration (VAD/barge-in handled by AI service)
    enable_vad: bool = True  # Informational only - AI service controls VAD
    silence_threshold_ms: int = 500

    # Model configuration
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # Custom instructions
    system_prompt: Optional[str] = None
    initial_context: Optional[str] = None


class AiDuplexBase:
    """Base class for AI duplex clients with common functionality."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20) -> None:
        """Initialize base client.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_ms: Frame duration in milliseconds
        """
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._frame_size = (sample_rate * frame_ms * 2) // 1000  # PCM16 = 2 bytes per sample
        self._connected = False

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        """Get expected frame size in bytes."""
        return self._frame_size

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def validate_frame(self, frame: bytes) -> None:
        """Validate audio frame size.

        Args:
            frame: Audio frame to validate

        Raises:
            ValueError: If frame size is invalid
        """
        if len(frame) != self._frame_size:
            raise ValueError(
                f"Invalid frame size: expected {self._frame_size}, got {len(frame)}"
            )

    def create_session_config(self, **kwargs: any) -> SessionConfig:
        """Create session configuration.

        Args:
            **kwargs: Configuration parameters

        Returns:
            Session configuration object
        """
        return SessionConfig(
            sample_rate=self._sample_rate,
            **kwargs
        )