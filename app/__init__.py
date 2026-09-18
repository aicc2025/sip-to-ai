"""SIP-to-AI: bidirectional audio bridge between SIP/PJSUA2 and AI realtime voice services.

Package layout:
- app.ai: AI voice service clients (OpenAI Realtime, Gemini Live, Grok Voice, Deepgram, 60db TTS)
- app.bridge: bridging layer between SIP telephony and the AI clients
- app.sip_async: pure asyncio SIP+RTP protocol stack
- app.utils: shared helpers (codec conversion, ring buffers, constants, agent config)
"""
