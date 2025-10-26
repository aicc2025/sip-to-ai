"""Shared test fixtures and configuration."""

import pytest
import pytest_asyncio
from typing import AsyncGenerator

from app.utils.ring_buffer import RingBuffer, StreamBuffer
from tests.mock_ai_client import MockDuplexClient


@pytest.fixture
def sample_rate() -> int:
    """Sample rate for tests."""
    return 16000


@pytest.fixture
def frame_ms() -> int:
    """Frame duration in milliseconds."""
    return 20


@pytest.fixture
def frame_size(sample_rate: int, frame_ms: int) -> int:
    """Frame size in bytes for PCM16."""
    return (sample_rate * frame_ms * 2) // 1000


@pytest.fixture
def g711_frame_8k() -> bytes:
    """Sample G.711 frame at 8kHz."""
    return bytes([128] * 160)  # 20ms at 8kHz


@pytest.fixture
def pcm16_frame_16k() -> bytes:
    """Sample PCM16 frame at 16kHz."""
    return bytes([0, 0] * 320)  # 20ms at 16kHz, 640 bytes


@pytest_asyncio.fixture
async def ring_buffer(frame_size: int) -> AsyncGenerator[RingBuffer, None]:
    """Create ring buffer for testing."""
    buffer = RingBuffer(capacity=10, frame_size=frame_size)
    yield buffer


@pytest_asyncio.fixture
async def stream_buffer() -> AsyncGenerator[StreamBuffer, None]:
    """Create stream buffer for testing."""
    buffer = StreamBuffer(capacity=10)
    yield buffer
    await buffer.close()


@pytest_asyncio.fixture
async def mock_ai_client(sample_rate: int, frame_ms: int) -> AsyncGenerator[MockDuplexClient, None]:
    """Create mock AI client for testing."""
    client = MockDuplexClient(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        echo_delay_ms=100
    )
    yield client
    await client.close()