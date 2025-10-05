"""Tests for ring buffer implementation."""

import asyncio
import pytest

from app.core.ring_buffer import RingBuffer, StreamBuffer


class TestRingBuffer:
    """Test ring buffer functionality."""

    @pytest.mark.asyncio
    async def test_basic_operations(self, ring_buffer: RingBuffer, frame_size: int) -> None:
        """Test basic push/pop operations."""
        frame = b'\x00' * frame_size

        # Test empty buffer
        assert await ring_buffer.is_empty()
        assert not await ring_buffer.is_full()
        assert await ring_buffer.size() == 0

        # Test push
        result = await ring_buffer.push(frame)
        assert result is True
        assert await ring_buffer.size() == 1
        assert not await ring_buffer.is_empty()

        # Test pop
        popped = await ring_buffer.pop()
        assert popped == frame
        assert await ring_buffer.is_empty()

    @pytest.mark.asyncio
    async def test_capacity_limits(self, frame_size: int) -> None:
        """Test buffer capacity limits."""
        capacity = 3
        buffer = RingBuffer(capacity=capacity, frame_size=frame_size)
        frame = b'\x00' * frame_size

        # Fill buffer
        for i in range(capacity):
            await buffer.push(frame)

        assert await buffer.size() == capacity
        assert await buffer.is_full()

        # Test overflow (should drop oldest)
        new_frame = b'\x01' * frame_size
        await buffer.push(new_frame)
        assert await buffer.size() == capacity

        # First popped should be the dropped frame (FIFO)
        popped = await buffer.pop()
        assert popped == frame

    @pytest.mark.asyncio
    async def test_water_level(self, ring_buffer: RingBuffer, frame_size: int) -> None:
        """Test water level calculation."""
        frame = b'\x00' * frame_size

        # Empty buffer
        assert await ring_buffer.get_water_level() == 0.0

        # Half full
        for _ in range(5):
            await ring_buffer.push(frame)
        assert await ring_buffer.get_water_level() == 0.5

        # Full
        for _ in range(5):
            await ring_buffer.push(frame)
        assert await ring_buffer.get_water_level() == 1.0

    @pytest.mark.asyncio
    async def test_peek(self, ring_buffer: RingBuffer, frame_size: int) -> None:
        """Test peek functionality."""
        frame = b'\x00' * frame_size

        # Empty buffer
        assert await ring_buffer.peek() is None

        # Add frame
        await ring_buffer.push(frame)

        # Peek should not remove
        peeked = await ring_buffer.peek()
        assert peeked == frame
        assert await ring_buffer.size() == 1

        # Pop should still work
        popped = await ring_buffer.pop()
        assert popped == frame

    @pytest.mark.asyncio
    async def test_clear(self, ring_buffer: RingBuffer, frame_size: int) -> None:
        """Test buffer clearing."""
        frame = b'\x00' * frame_size

        # Add frames
        for _ in range(5):
            await ring_buffer.push(frame)

        # Clear
        cleared = await ring_buffer.clear()
        assert cleared == 5
        assert await ring_buffer.is_empty()

    @pytest.mark.asyncio
    async def test_invalid_frame_size(self, ring_buffer: RingBuffer, frame_size: int) -> None:
        """Test validation of frame size."""
        wrong_frame = b'\x00' * (frame_size - 1)

        with pytest.raises(ValueError, match="Frame size"):
            await ring_buffer.push(wrong_frame)

    @pytest.mark.asyncio
    async def test_concurrent_access(self, frame_size: int) -> None:
        """Test concurrent push/pop operations."""
        buffer = RingBuffer(capacity=100, frame_size=frame_size)
        frame = b'\x00' * frame_size

        async def producer() -> None:
            for i in range(50):
                await buffer.push(frame)
                await asyncio.sleep(0.001)

        async def consumer() -> None:
            for i in range(50):
                await buffer.pop()
                await asyncio.sleep(0.001)

        # Run concurrently
        async with asyncio.TaskGroup() as tg:
            tg.create_task(producer())
            tg.create_task(consumer())

        # Buffer should be empty
        assert await buffer.is_empty()


class TestStreamBuffer:
    """Test stream buffer functionality."""

    @pytest.mark.asyncio
    async def test_basic_streaming(self, stream_buffer: StreamBuffer) -> None:
        """Test basic send/receive operations."""
        data = b'test data'

        # Send data
        await stream_buffer.send(data)

        # Receive data
        received = await stream_buffer.receive()
        assert received == data

    @pytest.mark.asyncio
    async def test_nowait_operations(self, stream_buffer: StreamBuffer) -> None:
        """Test non-blocking operations."""
        data = b'test data'

        # Empty buffer should raise QueueEmpty
        with pytest.raises(asyncio.QueueEmpty):
            stream_buffer.receive_nowait()

        # Send and receive nowait
        stream_buffer.send_nowait(data)
        received = stream_buffer.receive_nowait()
        assert received == data

    @pytest.mark.asyncio
    async def test_buffer_capacity(self) -> None:
        """Test buffer capacity limits."""
        buffer = StreamBuffer(capacity=2)
        data = b'test'

        # Fill buffer
        buffer.send_nowait(data)
        buffer.send_nowait(data)

        # Should be full now
        with pytest.raises(asyncio.QueueFull):
            buffer.send_nowait(data)

        await buffer.close()

    @pytest.mark.asyncio
    async def test_producer_consumer(self, stream_buffer: StreamBuffer) -> None:
        """Test producer/consumer pattern."""
        messages = [f"message_{i}".encode() for i in range(10)]
        received = []

        async def producer() -> None:
            for msg in messages:
                await stream_buffer.send(msg)

        async def consumer() -> None:
            for _ in range(len(messages)):
                msg = await stream_buffer.receive()
                received.append(msg)

        # Run concurrently
        async with asyncio.TaskGroup() as tg:
            tg.create_task(producer())
            tg.create_task(consumer())

        assert received == messages