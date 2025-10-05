"""Audio resampler for sample rate conversion."""

import numpy as np
import soxr
from typing import Optional


class Resampler:
    """Audio resampler using soxr for high-quality sample rate conversion."""

    def __init__(
        self,
        source_rate: int,
        target_rate: int,
        channels: int = 1,
        quality: str = "HQ"
    ) -> None:
        """Initialize resampler.

        Args:
            source_rate: Source sample rate in Hz
            target_rate: Target sample rate in Hz
            channels: Number of audio channels (default: 1 for mono)
            quality: Resampling quality ("LQ", "MQ", "HQ", "VHQ")

        Raises:
            ValueError: If rates or channels are invalid
        """
        if source_rate <= 0:
            raise ValueError(f"Source rate must be positive, got {source_rate}")
        if target_rate <= 0:
            raise ValueError(f"Target rate must be positive, got {target_rate}")
        if channels <= 0:
            raise ValueError(f"Channels must be positive, got {channels}")

        self._source_rate = source_rate
        self._target_rate = target_rate
        self._channels = channels
        self._ratio = target_rate / source_rate

        # Configure soxr resampler
        quality_map = {
            "LQ": soxr.LQ,
            "MQ": soxr.MQ,
            "HQ": soxr.HQ,
            "VHQ": soxr.VHQ,
        }
        self._quality = quality_map.get(quality, soxr.HQ)

    @property
    def source_rate(self) -> int:
        """Source sample rate."""
        return self._source_rate

    @property
    def target_rate(self) -> int:
        """Target sample rate."""
        return self._target_rate

    @property
    def ratio(self) -> float:
        """Resampling ratio (target/source)."""
        return self._ratio

    def resample(self, audio_data: bytes) -> bytes:
        """Resample audio data.

        Args:
            audio_data: PCM16 audio data at source rate

        Returns:
            Resampled PCM16 audio data at target rate
        """
        if len(audio_data) == 0:
            return b""

        # Convert bytes to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)

        # Reshape for channels if needed
        if self._channels > 1:
            samples = samples.reshape(-1, self._channels)

        # Resample using soxr
        resampled = soxr.resample(
            samples,
            self._source_rate,
            self._target_rate,
            quality=self._quality
        )

        # Convert back to int16
        resampled = np.round(resampled).astype(np.int16)

        return resampled.tobytes()

    def calculate_output_size(self, input_size: int) -> int:
        """Calculate expected output size after resampling.

        Args:
            input_size: Input size in bytes

        Returns:
            Expected output size in bytes
        """
        # Each sample is 2 bytes (int16)
        input_samples = input_size // (2 * self._channels)
        output_samples = int(np.ceil(input_samples * self._ratio))
        return output_samples * 2 * self._channels


class AdaptiveResampler(Resampler):
    """Adaptive resampler with drift compensation."""

    def __init__(
        self,
        source_rate: int,
        target_rate: int,
        channels: int = 1,
        quality: str = "HQ",
        max_drift_ppm: int = 1000
    ) -> None:
        """Initialize adaptive resampler.

        Args:
            source_rate: Source sample rate in Hz
            target_rate: Target sample rate in Hz
            channels: Number of audio channels
            quality: Resampling quality
            max_drift_ppm: Maximum drift in parts per million

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(source_rate, target_rate, channels, quality)
        self._max_drift_ppm = max_drift_ppm
        self._drift_compensation = 0.0
        self._accumulated_error = 0.0

    def compensate_drift(self, drift_frames: int) -> None:
        """Compensate for clock drift.

        Args:
            drift_frames: Number of frames to compensate (+/- 1 typically)
        """
        # Calculate drift in PPM
        drift_ppm = (drift_frames / self._target_rate) * 1_000_000

        # Clamp to maximum allowed drift
        drift_ppm = np.clip(drift_ppm, -self._max_drift_ppm, self._max_drift_ppm)

        # Update compensation
        self._drift_compensation = drift_ppm / 1_000_000

    def resample_with_compensation(self, audio_data: bytes) -> bytes:
        """Resample with drift compensation.

        Args:
            audio_data: PCM16 audio data at source rate

        Returns:
            Resampled PCM16 audio data with drift compensation
        """
        if self._drift_compensation == 0.0:
            return self.resample(audio_data)

        # Apply drift compensation to the effective rate
        adjusted_ratio = self._ratio * (1.0 + self._drift_compensation)

        # Convert to samples
        samples = np.frombuffer(audio_data, dtype=np.int16)
        if self._channels > 1:
            samples = samples.reshape(-1, self._channels)

        # Calculate target length with compensation
        input_samples = len(samples) // self._channels if self._channels > 1 else len(samples)
        target_samples = int(input_samples * adjusted_ratio)

        # Resample with adjusted rate
        resampled = soxr.resample(
            samples,
            self._source_rate,
            self._target_rate * (1.0 + self._drift_compensation),
            quality=self._quality
        )

        # Track accumulated error for fine-grained compensation
        expected_samples = int(input_samples * self._ratio)
        actual_samples = len(resampled) // self._channels if self._channels > 1 else len(resampled)
        self._accumulated_error += (expected_samples - actual_samples)

        # Apply correction if error exceeds threshold
        if abs(self._accumulated_error) >= 1:
            correction = int(self._accumulated_error)
            if correction > 0:
                # Add samples (repeat last)
                padding = np.repeat(resampled[-1:], correction, axis=0)
                resampled = np.concatenate([resampled, padding])
            else:
                # Remove samples
                resampled = resampled[:correction]
            self._accumulated_error -= correction

        # Convert back to int16
        resampled = np.round(resampled).astype(np.int16)
        return resampled.tobytes()

    def reset_compensation(self) -> None:
        """Reset drift compensation."""
        self._drift_compensation = 0.0
        self._accumulated_error = 0.0


def create_resampler(
    source_rate: int,
    target_rate: int,
    adaptive: bool = False
) -> Resampler:
    """Factory function to create appropriate resampler.

    Args:
        source_rate: Source sample rate
        target_rate: Target sample rate
        adaptive: Whether to create adaptive resampler

    Returns:
        Resampler instance
    """
    if adaptive:
        return AdaptiveResampler(source_rate, target_rate)
    else:
        return Resampler(source_rate, target_rate)