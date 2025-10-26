"""Pure asyncio SIP+RTP protocol stack.

This module provides a pure Python asyncio implementation of SIP and RTP protocols,
inspired by pyVoIP but designed for modern asyncio with TaskGroup support.

Key features:
- No GIL issues (pure Python asyncio)
- Structured concurrency with asyncio.TaskGroup
- Minimal implementation (INVITE/ACK/BYE + G.711 only)
- Python 3.12+ only
"""

__all__ = [
    "RTPSession",
    "G711Codec",
    "AsyncCall",
    "AsyncSIPServer",
    "RTPAudioBridge",
]

# Import order matters due to forward references
from app.sip_async.rtp_session import G711Codec, RTPSession
from app.sip_async.audio_bridge import RTPAudioBridge
from app.sip_async.async_call import AsyncCall
from app.sip_async.async_sip_server import AsyncSIPServer
