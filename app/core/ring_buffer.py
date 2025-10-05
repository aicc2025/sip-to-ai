"""Lock-free ring buffer for audio data."""

import asyncio
from collections import deque
from typing import Optional

import numpy as np


class RingBuffer:
    """Thread-safe ring buffer for audio frames."""

    def __init__(self, capacity: int, frame_size: int) -> None:
        """Initialize ring buffer.

        Args:
            capacity: Maximum number of frames to buffer
            frame_size: Size of each frame in bytes

        Raises:
            ValueError: If capacity or frame_size is <= 0
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if frame_size <= 0:
            raise ValueError(f"Frame size must be positive, got {frame_size}")

        self._capacity = capacity
        self._frame_size = frame_size
        self._buffer: deque[bytes] = deque(maxlen=capacity)
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> int:
        """Maximum number of frames."""
        return self._capacity

    @property
    def frame_size(self) -> int:
        """Size of each frame in bytes."""
        return self._frame_size

    async def size(self) -> int:
        """Current number of frames in buffer."""
        async with self._lock:
            return len(self._buffer)

    async def is_empty(self) -> bool:
        """Check if buffer is empty."""
        async with self._lock:
            return len(self._buffer) == 0

    async def is_full(self) -> bool:
        """Check if buffer is full."""
        async with self._lock:
            return len(self._buffer) == self._capacity

    async def push(self, frame: bytes) -> bool:
        """Push a frame to the buffer.

        Args:
            frame: Audio frame data

        Returns:
            True if pushed successfully, False if buffer was full

        Raises:
            ValueError: If frame size doesn't match buffer frame size
        """
        if len(frame) != self._frame_size:
            raise ValueError(f"Frame size {len(frame)} doesn't match buffer frame size {self._frame_size}")

        async with self._lock:
            if len(self._buffer) >= self._capacity:
                # Drop oldest frame when full
                self._buffer.popleft()
            self._buffer.append(frame)
            return True

    async def pop(self) -> Optional[bytes]:
        """Pop a frame from the buffer.

        Returns:
            Frame data if available, None if empty
        """
        async with self._lock:
            if self._buffer:
                return self._buffer.popleft()
            return None

    async def peek(self) -> Optional[bytes]:
        """Peek at the next frame without removing it.

        Returns:
            Frame data if available, None if empty
        """
        async with self._lock:
            if self._buffer:
                return self._buffer[0]
            return None

    async def clear(self) -> int:
        """Clear all frames from buffer.

        Returns:
            Number of frames cleared
        """
        async with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            return count

    async def get_water_level(self) -> float:
        """Get buffer water level as percentage.

        Returns:
            Water level from 0.0 (empty) to 1.0 (full)
        """
        async with self._lock:
            return len(self._buffer) / self._capacity


class StreamBuffer:
    """Asyncio Queue-based buffer for async communication."""

    def __init__(self, capacity: int) -> None:
        """Initialize stream buffer.

        Args:
            capacity: Maximum number of items to buffer
        """
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=capacity)
        self._capacity = capacity
        self._closed = False

    def send_nowait(self, item: bytes) -> None:
        """Send item without waiting.

        Args:
            item: Data to send

        Raises:
            asyncio.QueueFull: If buffer is full
        """
        if self._closed:
            return
        self._queue.put_nowait(item)

    async def send(self, item: bytes) -> None:
        """Send item, waiting if necessary.

        Args:
            item: Data to send
        """
        if self._closed:
            return
        await self._queue.put(item)

    async def receive(self) -> bytes:
        """Receive item, waiting if necessary.

        Returns:
            Received data
        """
        return await self._queue.get()

    def receive_nowait(self) -> bytes:
        """Receive item without waiting.

        Returns:
            Received data

        Raises:
            asyncio.QueueEmpty: If buffer is empty
        """
        return self._queue.get_nowait()

    async def close(self) -> None:
        """Close the buffer."""
        self._closed = True
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break