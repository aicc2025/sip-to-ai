"""AI voice service clients.

All providers implement the duplex interface defined in app.ai.duplex_base:
- duplex_base: shared base class, event types and the downlink audio queue
- openai_realtime: OpenAI Realtime API (and compatible gateways)
- gemini_live: Google Gemini Live API, with transparent session resumption
- grok_voice: xAI Grok Voice
- deepgram_agent: Deepgram Voice Agent (STT + LLM + turn-taking)
- sixtydb_tts: 60db text-to-speech, used as the voice for the Deepgram agent
"""
