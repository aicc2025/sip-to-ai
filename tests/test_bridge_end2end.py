"""End-to-end tests for the complete bridge system."""

import asyncio
import time
import pytest
import numpy as np

from app.bridge import AudioAdapter, CallSession
from tests.mock_ai_client import MockDuplexClient


class TestAudioAdapter:
    """Test audio adapter functionality."""

    @pytest.mark.asyncio
    async def test_pcm16_passthrough_uplink(self) -> None:
        """Test PCM16 passthrough for uplink (PJSUA2 -> AI)."""
        adapter = AudioAdapter(
            uplink_capacity=10,
            downlink_capacity=10
        )

        # Generate test PCM16 frame from PJSUA2 (20ms at 8kHz = 320 bytes)
        pcm16_frame = bytes([0] * 320)  # PCM16 silence

        # Process uplink
        adapter.on_rx_pcm16_8k(pcm16_frame)

        # Should have uplink audio available
        try:
            uplink_audio = await asyncio.wait_for(
                adapter.get_uplink_audio(),
                timeout=1.0
            )
            # Should be 320 bytes (PCM16 passthrough)
            assert len(uplink_audio) == 320
        except asyncio.TimeoutError:
            pytest.fail("No uplink audio received")

        await adapter.close()

    @pytest.mark.asyncio
    async def test_buffer_water_levels(self) -> None:
        """Test buffer water level management."""
        adapter = AudioAdapter(
            uplink_capacity=5,    # Small buffers for testing
            downlink_capacity=5
        )

        pcm16_frame = bytes([0] * 320)

        # Fill uplink buffer
        for i in range(8):  # More than capacity
            adapter.on_rx_pcm16_8k(pcm16_frame)

        # Should not have more than capacity
        # Try to get frames - should not block indefinitely
        frames_received = 0
        start_time = time.time()

        while time.time() - start_time < 1.0:  # 1 second timeout
            try:
                await asyncio.wait_for(adapter.get_uplink_audio(), timeout=0.1)
                frames_received += 1
                if frames_received >= 5:  # Don't wait for more than capacity
                    break
            except asyncio.TimeoutError:
                break

        assert frames_received <= 5, f"Received {frames_received} frames, expected ≤5"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_adapter_stats(self) -> None:
        """Test adapter statistics collection."""
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=10)

        # Get stats
        stats = adapter.get_stats()
        assert "frames_received" in stats
        assert "frames_sent" in stats
        assert "mode" in stats
        assert stats["mode"] == "pcm16_passthrough"

        await adapter.close()


class TestCallSession:
    """Test complete call session functionality."""

    @pytest.mark.asyncio
    async def test_full_bridge_startup_shutdown(self) -> None:
        """Test full bridge startup and shutdown."""
        # Create components
        ai_client = MockDuplexClient(sample_rate=8000, frame_ms=20)

        audio_adapter = AudioAdapter(
            uplink_capacity=10,
            downlink_capacity=10
        )

        session = CallSession(
            audio_adapter=audio_adapter,
            ai_client=ai_client
        )

        # Start bridge
        start_time = time.time()
        await session.start()

        # Should start without error
        startup_time = time.time() - start_time
        assert startup_time < 5.0, f"Startup took {startup_time:.1f}s, too slow"

        # Let it run briefly
        await asyncio.sleep(0.5)

        # Stop bridge
        stop_time = time.time()
        await session.stop()

        shutdown_time = time.time() - stop_time
        assert shutdown_time < 2.0, f"Shutdown took {shutdown_time:.1f}s, too slow"