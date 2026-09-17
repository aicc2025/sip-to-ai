"""Tests for codec functionality."""

import pytest

from app.utils.codec import convert_g711_to_pcm16, convert_pcm16_to_g711


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


def test_resample_pcm16_downsample_short_input_keeps_ratio() -> None:
    """Inputs shorter than the anti-aliasing filter must not grow to the filter length."""
    from app.utils.codec import resample_pcm16

    assert len(resample_pcm16(b"\x10\x00" * 3, 24000, 8000)) == 2
    assert len(resample_pcm16(b"\x10\x00" * 6, 24000, 8000)) == 4
