"""Configuration module for SIP-to-AI bridge."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AudioConfig:
    """Audio processing configuration.

    Note: Fixed audio parameters for telephony compatibility:
    - Frame duration: 20ms (telephony standard)
    - Sample rate: 8kHz (PJSUA2 and telephony standard)
    - PJSUA2: PCM16 @ 8kHz (320 bytes/frame)
    - AI clients: Handle conversion internally (PCM16 → their format)
    """

    # Fixed audio parameters (not configurable)
    FRAME_MS: int = 20      # Fixed: 20ms telephony standard
    SAMPLE_RATE: int = 8000 # Fixed: 8kHz telephony standard

    # Buffer configuration (configurable)
    uplink_buf_frames: int = 100
    downlink_buf_frames: int = 200

    @property
    def frame_ms(self) -> int:
        """Frame duration in milliseconds."""
        return self.FRAME_MS

    @property
    def sip_sr(self) -> int:
        """SIP sample rate (compatibility alias)."""
        return self.SAMPLE_RATE

    @property
    def ai_sr(self) -> int:
        """AI sample rate (compatibility alias)."""
        return self.SAMPLE_RATE

    @property
    def frame_size_sip(self) -> int:
        """Samples per frame at SIP sample rate."""
        return (self.sip_sr * self.frame_ms) // 1000

    @property
    def frame_size_ai(self) -> int:
        """Samples per frame at AI sample rate."""
        return (self.ai_sr * self.frame_ms) // 1000

    @property
    def bytes_per_frame_sip_pcm16(self) -> int:
        """Bytes per frame for SIP in PCM16 format (PJSUA2 interface).
        
        Returns:
            320 bytes for 20ms @ 8kHz PCM16
        """
        return self.frame_size_sip * 2
    
    @property
    def bytes_per_frame_sip(self) -> int:
        """Bytes per frame for SIP (PCM16). Alias for bytes_per_frame_sip_pcm16."""
        return self.bytes_per_frame_sip_pcm16

    @property
    def bytes_per_frame_ai_g711(self) -> int:
        """Bytes per frame for AI in G.711 format (OpenAI Realtime).
        
        Returns:
            160 bytes for 20ms @ 8kHz G.711
        """
        return self.frame_size_ai
    
    @property
    def bytes_per_frame_ai(self) -> int:
        """Bytes per frame for AI (G.711). Alias for bytes_per_frame_ai_g711."""
        return self.bytes_per_frame_ai_g711


@dataclass(frozen=True)
class AIConfig:
    """AI service configuration.

    Note: VAD, barge-in, and turn-taking are handled by AI services (OpenAI/Deepgram).
    No client-side configuration needed.
    """

    vendor: Literal["openai", "deepgram"] = "openai"

    # Agent Prompt Configuration (shared across vendors)
    # NOTE: Path can be relative (resolved from project root) or absolute
    agent_prompt_file: str = ""  # REQUIRED: Path to YAML file with agent prompts (greeting + instructions)

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_ws_endpoint: str = "wss://api.openai.com/v1/realtime"
    openai_model: str = "gpt-4-turbo-realtime"

    # Deepgram Configuration
    deepgram_api_key: str = ""
    deepgram_listen_model: str = "nova-2"  # STT model (nova-2, nova-3)
    deepgram_speak_model: str = "aura-asteria-en"  # TTS voice
    deepgram_llm_model: str = "gpt-4o-mini"  # LLM model for agent


@dataclass(frozen=True)
class SIPConfig:
    """SIP/PJSUA2 configuration for userless account (receive-only mode).

    Note: This configuration is for receiving incoming SIP INVITE requests only.
    No registration to SIP server required.
    Typically used in internal network, TLS not required.
    """

    domain: str = "localhost"  # SIP domain/IP for URI (useful for multi-IP servers)
    transport_type: Literal["udp", "tcp"] = "udp"
    port: int = 5060


@dataclass(frozen=True)
class SystemConfig:
    """System-wide configuration."""

    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    health_check_interval_sec: int = 30
    reconnect_delay_sec: int = 5
    max_reconnect_attempts: int = 3
    ai_connection_timeout_sec: int = 10  # Timeout for AI WebSocket connection


class Config:
    """Main configuration class."""

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        self.audio = AudioConfig(
            uplink_buf_frames=int(os.getenv("UPLINK_BUF_FRAMES", "100")),
            downlink_buf_frames=int(os.getenv("DOWNLINK_BUF_FRAMES", "200")),
        )

        ai_vendor = os.getenv("AI_VENDOR", "mock").lower()
        if ai_vendor not in ["mock", "openai", "deepgram"]:
            ai_vendor = "mock"

        self.ai = AIConfig(
            vendor=ai_vendor,  # type: ignore
            agent_prompt_file=os.getenv("AGENT_PROMPT_FILE", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_ws_endpoint=os.getenv("OPENAI_WS_ENDPOINT", "wss://api.openai.com/v1/realtime"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-realtime"),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            deepgram_listen_model=os.getenv("DEEPGRAM_LISTEN_MODEL", "nova-2"),
            deepgram_speak_model=os.getenv("DEEPGRAM_SPEAK_MODEL", "aura-asteria-en"),
            deepgram_llm_model=os.getenv("DEEPGRAM_LLM_MODEL", "gpt-4o-mini"),
        )

        self.sip = SIPConfig(
            domain=os.getenv("SIP_DOMAIN", "localhost"),
            transport_type=os.getenv("SIP_TRANSPORT_TYPE", "udp"),  # type: ignore
            port=int(os.getenv("SIP_PORT", "5060")),
        )

        self.system = SystemConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),  # type: ignore
            health_check_interval_sec=int(os.getenv("HEALTH_CHECK_INTERVAL_SEC", "30")),
            reconnect_delay_sec=int(os.getenv("RECONNECT_DELAY_SEC", "5")),
            max_reconnect_attempts=int(os.getenv("MAX_RECONNECT_ATTEMPTS", "3")),
            ai_connection_timeout_sec=int(os.getenv("AI_CONNECTION_TIMEOUT_SEC", "10")),
        )

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment."""
        return cls()


config = Config.load()