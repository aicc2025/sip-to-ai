"""Audio processing constants."""


class AudioConstants:
    """Audio format constants for SIP and AI processing."""

    # Sample rate
    SAMPLE_RATE = 8000  # 8kHz for SIP/G.711/mulaw

    # Frame timing
    FRAME_MS = 20  # 20ms frame duration

    # Frame sizes
    PCM16_FRAME_SIZE = 320  # PCM16 @ 8kHz: (8000 * 20 * 2) / 1000 = 320 bytes
    G711_FRAME_SIZE = 160   # G.711 @ 8kHz: (8000 * 20 * 1) / 1000 = 160 bytes

    # Common frames
    SILENCE_FRAME = b'\x00' * PCM16_FRAME_SIZE  # 20ms silence @ 8kHz PCM16

    # Logging intervals
    LOG_INTERVAL_FRAMES = 50   # Log every 50 frames (1 second @ 20ms)
    LOG_INTERVAL_STATS = 100   # Stats every 100 frames (2 seconds @ 20ms)
