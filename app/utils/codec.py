"""Audio codec for μ-law/A-law to PCM16 conversion."""

import numpy as np
from typing import Literal


class Codec:
    """G.711 μ-law/A-law codec with vectorized operations."""

    # μ-law constants
    ULAW_MAX = 0x7FFF  # Maximum linear value
    ULAW_BIAS = 0x84  # Bias for linear code

    # A-law constants
    ALAW_MAX = 0xFFF
    ALAW_AMI_MASK = 0x55

    @staticmethod
    def ulaw_to_pcm16(ulaw_data: bytes) -> bytes:
        """Convert μ-law to 16-bit PCM.

        Args:
            ulaw_data: μ-law encoded audio data

        Returns:
            PCM16 encoded audio data
        """
        # Create lookup table for performance
        if not hasattr(Codec, '_ulaw_table'):
            Codec._ulaw_table = Codec._create_ulaw_table()

        # Vectorized conversion using numpy
        ulaw_array = np.frombuffer(ulaw_data, dtype=np.uint8)
        pcm_array = Codec._ulaw_table[ulaw_array]
        return pcm_array.tobytes()

    @staticmethod
    def pcm16_to_ulaw(pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to μ-law.

        Args:
            pcm_data: PCM16 encoded audio data

        Returns:
            μ-law encoded audio data
        """
        # Create inverse lookup table for performance
        if not hasattr(Codec, '_pcm_to_ulaw_table'):
            Codec._pcm_to_ulaw_table = Codec._create_pcm_to_ulaw_table()

        # Vectorized conversion
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)

        # Clip to valid range
        pcm_array = np.clip(pcm_array, -32768, 32767)

        # Use lookup for common values, compute for others
        ulaw_array = np.zeros(len(pcm_array), dtype=np.uint8)
        for i, sample in enumerate(pcm_array):
            ulaw_array[i] = Codec._encode_ulaw_sample(sample)

        return ulaw_array.tobytes()

    @staticmethod
    def alaw_to_pcm16(alaw_data: bytes) -> bytes:
        """Convert A-law to 16-bit PCM.

        Args:
            alaw_data: A-law encoded audio data

        Returns:
            PCM16 encoded audio data
        """
        # Create lookup table for performance
        if not hasattr(Codec, '_alaw_table'):
            Codec._alaw_table = Codec._create_alaw_table()

        # Vectorized conversion
        alaw_array = np.frombuffer(alaw_data, dtype=np.uint8)
        pcm_array = Codec._alaw_table[alaw_array]
        return pcm_array.tobytes()

    @staticmethod
    def pcm16_to_alaw(pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to A-law.

        Args:
            pcm_data: PCM16 encoded audio data

        Returns:
            A-law encoded audio data
        """
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
        alaw_array = np.zeros(len(pcm_array), dtype=np.uint8)

        for i, sample in enumerate(pcm_array):
            alaw_array[i] = Codec._encode_alaw_sample(sample)

        return alaw_array.tobytes()

    @staticmethod
    def _create_ulaw_table() -> np.ndarray:
        """Create μ-law to PCM16 lookup table."""
        table = np.zeros(256, dtype=np.int16)
        for i in range(256):
            # Complement to obtain normal u-law value
            ulaw = ~i & 0xFF

            # Extract sign, exponent, and mantissa
            sign = ulaw & 0x80
            exponent = (ulaw >> 4) & 0x07
            mantissa = ulaw & 0x0F

            # Compute sample
            sample = mantissa << (exponent + 3)
            sample += Codec.ULAW_BIAS << exponent
            sample -= Codec.ULAW_BIAS

            # Apply sign
            if sign == 0:
                sample = -sample

            table[i] = sample

        return table

    @staticmethod
    def _create_pcm_to_ulaw_table() -> dict[int, int]:
        """Create PCM16 to μ-law lookup table for common values."""
        table = {}
        for pcm in range(-32768, 32768):
            table[pcm] = Codec._encode_ulaw_sample(pcm)
        return table

    @staticmethod
    def _encode_ulaw_sample(sample: int) -> int:
        """Encode a single PCM16 sample to μ-law."""
        # Get sign
        if sample < 0:
            sign = 0x80
            sample = -sample
        else:
            sign = 0

        # Clip
        if sample > Codec.ULAW_MAX:
            sample = Codec.ULAW_MAX

        # Add bias
        sample += Codec.ULAW_BIAS

        # Find exponent
        exponent = 7
        mask = 0x4000
        while (sample & mask) == 0 and exponent > 0:
            exponent -= 1
            mask >>= 1

        # Extract mantissa
        mantissa = (sample >> (exponent + 3)) & 0x0F

        # Combine
        ulaw = sign | (exponent << 4) | mantissa

        # Complement
        return ~ulaw & 0xFF

    @staticmethod
    def _create_alaw_table() -> np.ndarray:
        """Create A-law to PCM16 lookup table."""
        table = np.zeros(256, dtype=np.int16)
        for i in range(256):
            # XOR with AMI mask
            alaw = i ^ Codec.ALAW_AMI_MASK

            # Extract sign and magnitude
            sign = alaw & 0x80
            magnitude = alaw & 0x7F

            # Decode magnitude
            if magnitude < 16:
                sample = magnitude << 4
            else:
                exponent = (magnitude >> 4) & 0x07
                mantissa = magnitude & 0x0F
                sample = (mantissa << 4) + 0x108
                sample <<= exponent - 1

            # Apply sign
            if sign:
                sample = -sample

            table[i] = sample

        return table

    @staticmethod
    def _encode_alaw_sample(sample: int) -> int:
        """Encode a single PCM16 sample to A-law."""
        # Get sign
        if sample < 0:
            sign = 0x80
            sample = -sample
        else:
            sign = 0

        # Clip
        if sample > Codec.ALAW_MAX:
            sample = Codec.ALAW_MAX

        # Encode magnitude
        if sample < 256:
            alaw = sample >> 4
        else:
            # Find exponent
            exponent = 7
            mask = 0x4000
            while (sample & mask) == 0 and exponent > 1:
                exponent -= 1
                mask >>= 1

            # Extract mantissa
            mantissa = (sample >> (exponent + 3)) & 0x0F
            alaw = (exponent << 4) | mantissa

        # Apply sign and XOR with AMI mask
        return (sign | alaw) ^ Codec.ALAW_AMI_MASK


def convert_g711_to_pcm16(data: bytes, encoding: Literal["ulaw", "alaw"]) -> bytes:
    """Convert G.711 encoded audio to PCM16.

    Args:
        data: G.711 encoded audio data
        encoding: Encoding type ("ulaw" or "alaw")

    Returns:
        PCM16 encoded audio data

    Raises:
        ValueError: If encoding type is not supported
    """
    if encoding == "ulaw":
        return Codec.ulaw_to_pcm16(data)
    elif encoding == "alaw":
        return Codec.alaw_to_pcm16(data)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


def convert_pcm16_to_g711(data: bytes, encoding: Literal["ulaw", "alaw"]) -> bytes:
    """Convert PCM16 audio to G.711 encoding.

    Args:
        data: PCM16 encoded audio data
        encoding: Target encoding type ("ulaw" or "alaw")

    Returns:
        G.711 encoded audio data

    Raises:
        ValueError: If encoding type is not supported
    """
    if encoding == "ulaw":
        return Codec.pcm16_to_ulaw(data)
    elif encoding == "alaw":
        return Codec.pcm16_to_alaw(data)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")