"""Tests for codec and resampler functionality."""

import numpy as np
import pytest

from app.core.codec import Codec, convert_g711_to_pcm16, convert_pcm16_to_g711
from app.core.resampler import Resampler, AdaptiveResampler


class TestCodec:
    """Test audio codec functionality."""

    def test_g711_frame_size(self) -> None:
        """Test G.711 frame size conversion."""
        # 20ms frame at 8kHz = 160 samples = 320 bytes PCM16 = 160 bytes G.711
        pcm16_8k = b'\x00' * 320  # 160 samples * 2 bytes

        # Convert to μ-law
        ulaw = convert_pcm16_to_g711(pcm16_8k, "ulaw")
        assert len(ulaw) == 160

        # Convert back
        pcm_back = convert_g711_to_pcm16(ulaw, "ulaw")
        assert len(pcm_back) == 320

    def test_invalid_encoding(self) -> None:
        """Test invalid encoding handling."""
        data = b'\x00' * 160

        with pytest.raises(ValueError, match="Unsupported encoding"):
            convert_g711_to_pcm16(data, "invalid")  # type: ignore

        with pytest.raises(ValueError, match="Unsupported encoding"):
            convert_pcm16_to_g711(data, "invalid")  # type: ignore

    def test_empty_data(self) -> None:
        """Test handling of empty data."""
        empty = b''

        ulaw_result = convert_pcm16_to_g711(empty, "ulaw")
        assert ulaw_result == empty

        pcm_result = convert_g711_to_pcm16(empty, "ulaw")
        assert pcm_result == empty


class TestResampler:
    """Test audio resampler functionality."""

    def test_8k_to_16k_upsample(self) -> None:
        """Test upsampling from 8kHz to 16kHz."""
        resampler = Resampler(source_rate=8000, target_rate=16000)

        # 20ms at 8kHz = 160 samples = 320 bytes
        input_data = np.zeros(160, dtype=np.int16).tobytes()

        output_data = resampler.resample(input_data)

        # Should be approximately 2x the length
        assert 620 <= len(output_data) <= 660  # Allow some variation due to filtering

    def test_16k_to_8k_downsample(self) -> None:
        """Test downsampling from 16kHz to 8kHz."""
        resampler = Resampler(source_rate=16000, target_rate=8000)

        # 20ms at 16kHz = 320 samples = 640 bytes
        input_data = np.zeros(320, dtype=np.int16).tobytes()

        output_data = resampler.resample(input_data)

        # Should be approximately half the length
        assert 300 <= len(output_data) <= 340  # Allow some variation

    def test_same_rate(self) -> None:
        """Test resampling with same source and target rate."""
        resampler = Resampler(source_rate=16000, target_rate=16000)

        input_data = np.zeros(320, dtype=np.int16).tobytes()
        output_data = resampler.resample(input_data)

        # Should be same length
        assert len(output_data) == len(input_data)

    def test_empty_input(self) -> None:
        """Test resampling with empty input."""
        resampler = Resampler(source_rate=8000, target_rate=16000)

        output = resampler.resample(b'')
        assert output == b''

    def test_calculate_output_size(self) -> None:
        """Test output size calculation."""
        resampler = Resampler(source_rate=8000, target_rate=16000)

        # 320 bytes input (160 samples) should give ~640 bytes output
        expected = resampler.calculate_output_size(320)
        assert 620 <= expected <= 660

    def test_invalid_parameters(self) -> None:
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            Resampler(source_rate=0, target_rate=16000)

        with pytest.raises(ValueError):
            Resampler(source_rate=8000, target_rate=-1)

        with pytest.raises(ValueError):
            Resampler(source_rate=8000, target_rate=16000, channels=0)

    def test_signal_preservation(self) -> None:
        """Test that resampling preserves signal characteristics."""
        resampler = Resampler(source_rate=8000, target_rate=16000)

        # Generate a sine wave at 1kHz
        t = np.arange(160) / 8000  # 20ms at 8kHz
        signal = (np.sin(2 * np.pi * 1000 * t) * 16000).astype(np.int16)
        input_data = signal.tobytes()

        output_data = resampler.resample(input_data)
        output_signal = np.frombuffer(output_data, dtype=np.int16)

        # Output should have roughly twice as many samples
        assert 300 <= len(output_signal) <= 350

        # Should still be a sine wave (check for reasonable amplitude)
        assert np.max(np.abs(output_signal)) > 10000


class TestAdaptiveResampler:
    """Test adaptive resampler with drift compensation."""

    def test_reset_compensation(self) -> None:
        """Test resetting drift compensation."""
        resampler = AdaptiveResampler(source_rate=8000, target_rate=16000)

        # Apply compensation
        resampler.compensate_drift(1)

        # Reset
        resampler.reset_compensation()

        # Should behave like normal resampler
        input_data = np.zeros(160, dtype=np.int16).tobytes()
        output1 = resampler.resample(input_data)
        output2 = resampler.resample_with_compensation(input_data)

        assert len(output1) == len(output2)

    def test_drift_limits(self) -> None:
        """Test drift compensation limits."""
        resampler = AdaptiveResampler(
            source_rate=8000,
            target_rate=16000,
            max_drift_ppm=500
        )

        # Apply excessive drift - should be clamped
        resampler.compensate_drift(1000)  # Very large drift

        # Should still work without error
        input_data = np.zeros(160, dtype=np.int16).tobytes()
        output = resampler.resample_with_compensation(input_data)
        assert len(output) > 0