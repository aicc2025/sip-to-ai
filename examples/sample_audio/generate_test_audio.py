#!/usr/bin/env python3
"""Generate test audio files for development and testing."""

import numpy as np
import wave
from pathlib import Path


def generate_tone(frequency: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a pure tone.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio samples as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    return (tone * 32767).astype(np.int16)


def generate_speech_pattern(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a speech-like pattern with varying frequencies.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate

    Returns:
        Audio samples simulating speech patterns
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Base frequencies for speech-like pattern
    formants = [350, 900, 2100]  # Typical vowel formants
    speech = np.zeros_like(t)

    for freq in formants:
        # Add frequency with random modulation
        modulation = 1 + 0.1 * np.sin(2 * np.pi * 10 * t)  # 10 Hz modulation
        component = np.sin(2 * np.pi * freq * t * modulation)
        speech += component / len(formants)

    # Apply envelope (speech-like amplitude variation)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz envelope
    speech *= envelope

    # Add some noise for realism
    noise = np.random.normal(0, 0.05, speech.shape)
    speech += noise

    return (speech * 16000).astype(np.int16)


def save_wav(filename: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Save audio to WAV file.

    Args:
        filename: Output filename
        audio: Audio samples
        sample_rate: Sample rate
    """
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    print(f"Generated: {filename} ({len(audio)/sample_rate:.1f}s)")


def main() -> None:
    """Generate test audio files."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    print("Generating test audio files...")

    # 1. Pure tones for testing
    tone_440 = generate_tone(440, 2.0)  # A4 note, 2 seconds
    save_wav(str(output_dir / "tone_440hz_2s.wav"), tone_440)

    tone_1000 = generate_tone(1000, 1.0)  # 1kHz test tone
    save_wav(str(output_dir / "tone_1000hz_1s.wav"), tone_1000)

    # 2. Speech-like pattern
    speech = generate_speech_pattern(5.0)  # 5 seconds
    save_wav(str(output_dir / "speech_pattern_5s.wav"), speech)

    # 3. Multi-tone signal for frequency response testing
    multi_tone = np.zeros(int(16000 * 3))  # 3 seconds
    for i, freq in enumerate([200, 500, 1000, 2000, 4000]):
        start = int(i * 16000 * 0.6)
        end = start + int(16000 * 0.5)  # 0.5s per tone
        if end <= len(multi_tone):
            tone = generate_tone(freq, 0.5)
            multi_tone[start:end] = tone

    save_wav(str(output_dir / "multi_tone_test.wav"), multi_tone.astype(np.int16))

    # 4. Silence for testing
    silence = np.zeros(int(16000 * 1), dtype=np.int16)  # 1 second
    save_wav(str(output_dir / "silence_1s.wav"), silence)

    # 5. Noise for testing noise handling
    noise = np.random.normal(0, 1000, int(16000 * 2)).astype(np.int16)
    save_wav(str(output_dir / "white_noise_2s.wav"), noise)

    # 6. Sweep tone for frequency response
    t = np.linspace(0, 3, int(16000 * 3), False)
    freq_sweep = np.sin(2 * np.pi * (100 + 1900 * t / 3) * t)  # 100Hz to 2kHz in 3s
    save_wav(str(output_dir / "frequency_sweep.wav"), (freq_sweep * 16000).astype(np.int16))

    # 7. Generate 8kHz versions for SIP testing
    print("\nGenerating 8kHz versions for SIP testing...")

    # Downsample to 8kHz
    for wav_file in output_dir.glob("*.wav"):
        if "8k" not in wav_file.name:
            # Load 16kHz file
            with wave.open(str(wav_file), 'r') as wav:
                frames = wav.readframes(wav.getnframes())
                audio_16k = np.frombuffer(frames, dtype=np.int16)

            # Simple downsampling (every other sample)
            audio_8k = audio_16k[::2]

            # Save 8kHz version
            output_8k = output_dir / f"{wav_file.stem}_8k.wav"
            save_wav(str(output_8k), audio_8k, sample_rate=8000)

    print(f"\nTest audio files generated in: {output_dir}")
    print("\nFiles created:")
    for wav_file in sorted(output_dir.glob("*.wav")):
        print(f"  - {wav_file.name}")

    print("\nUsage examples:")
    print("  # Test tone for audio path verification")
    print("  ffplay tone_440hz_2s.wav")
    print("")
    print("  # Speech pattern for AI testing")
    print("  ffplay speech_pattern_5s.wav")
    print("")
    print("  # 8kHz files for SIP endpoint testing")
    print("  ffplay tone_440hz_2s_8k.wav")


if __name__ == "__main__":
    main()