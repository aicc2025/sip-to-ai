"""Audio bridge between SIP and AI services.

This module provides the bridging layer between SIP telephony and AI voice services:
- AudioAdapter: Audio format adapter (PCM16 passthrough, codec conversion)
- CallSession: AI connection lifecycle manager with health monitoring
"""

__all__ = [
    "AudioAdapter",
    "CallSession",
]

from app.bridge.audio_adapter import AudioAdapter
from app.bridge.call_session import CallSession
