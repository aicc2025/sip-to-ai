"""Main application entry point for SIP-to-AI bridge."""

import asyncio
import logging
import signal
import sys
from typing import Optional

import structlog

from pathlib import Path

from app.ai.deepgram_agent import DeepgramAgentClient
from app.ai.duplex_base import AiDuplexClient
from app.ai.openai_realtime import OpenAIRealtimeClient
from app.bridge import AudioAdapter, CallSession
from app.config import config
from app.utils.agent_config import AgentConfig
from app.sip_async import AsyncCall, AsyncSIPServer


def setup_logging() -> None:
    """Configure structured logging with file output."""
    from pathlib import Path
    from datetime import datetime

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sip-to-ai_{timestamp}.log"

    # Configure Python standard logging with both console and file handlers
    log_level = getattr(logging, config.system.log_level.upper(), logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[console_handler, file_handler]
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if config.system.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Log the log file location
    logger = structlog.get_logger(__name__)
    logger.info(f"Logging to file: {log_file}")


def _load_agent_config(logger: structlog.BoundLogger) -> tuple[str, Optional[str]]:
    """Load agent configuration from YAML file.

    Args:
        logger: Logger instance

    Returns:
        Tuple of (instructions, greeting)
    """
    # Default values if no config file
    if not config.ai.agent_prompt_file:
        return "You are a helpful assistant.", None

    # Resolve file path relative to project root if not absolute
    yaml_path = Path(config.ai.agent_prompt_file)
    if not yaml_path.is_absolute():
        project_root = Path(__file__).parent.parent
        yaml_path = project_root / yaml_path

    logger.info(
        "Loading agent prompts from YAML",
        file_path=config.ai.agent_prompt_file,
        resolved_path=str(yaml_path),
        exists=yaml_path.exists()
    )

    agent_config = AgentConfig.from_yaml(yaml_path)
    return agent_config.instructions, agent_config.greeting


def create_ai_client() -> AiDuplexClient:
    """Create AI client based on configuration.

    Returns:
        AI duplex client instance

    Raises:
        ValueError: If vendor is not supported
    """
    vendor = config.ai.vendor
    logger = structlog.get_logger(__name__)

    if vendor == "openai":
        if not config.ai.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        # Load agent configuration (optional for OpenAI)
        instructions, greeting = _load_agent_config(logger)

        logger.info(
            "Using OpenAI Realtime client",
            model=config.ai.openai_model,
            has_greeting=greeting is not None,
            instructions_length=len(instructions),
            greeting_preview=greeting[:50] if greeting else None
        )

        client = OpenAIRealtimeClient(
            api_key=config.ai.openai_api_key,
            model=config.ai.openai_model,
            instructions=instructions,
            greeting=greeting
        )
        logger.info("OpenAI client instance created")
        return client

    elif vendor == "deepgram":
        if not config.ai.deepgram_api_key:
            raise ValueError("Deepgram API key not configured")

        # FAIL-FIRST: YAML prompt file is REQUIRED for Deepgram
        if not config.ai.agent_prompt_file:
            raise ValueError(
                "Agent prompt file is required. "
                "Set AGENT_PROMPT_FILE=agent_config.yaml"
            )

        # Load agent configuration (required for Deepgram)
        instructions, greeting = _load_agent_config(logger)

        logger.info(
            "Using Deepgram Voice Agent client",
            prompt_file=config.ai.agent_prompt_file,
            instructions_length=len(instructions),
            has_greeting=greeting is not None,
            greeting_preview=greeting[:50] if greeting else None,
            instructions_preview=instructions[:100] if instructions else None
        )

        return DeepgramAgentClient(
            api_key=config.ai.deepgram_api_key,
            sample_rate=config.audio.sip_sr,  # Use SIP sample rate (8kHz)
            frame_ms=config.audio.frame_ms,
            audio_format="mulaw",  # Deepgram uses mulaw (same as g711_ulaw)
            listen_model=config.ai.deepgram_listen_model,
            speak_model=config.ai.deepgram_speak_model,
            llm_model=config.ai.deepgram_llm_model,
            instructions=instructions,
            greeting=greeting
        )

    else:
        raise ValueError(f"Unsupported AI vendor: {vendor}")


async def run_real_mode() -> None:
    """Run in real mode with actual SIP and AI services.

    Each incoming call will create its own AI client and bridge.
    """
    logger = structlog.get_logger(__name__)
    logger.info("Starting SIP-to-AI Bridge (Pure Asyncio)")

    async def on_incoming_call(call: AsyncCall) -> None:
        """Handle incoming call - setup AudioAdapter and AI session.

        Args:
            call: AsyncCall instance
        """
        logger.info(
            "Incoming call - setting up resources",
            call_id=call.call_id
        )

        try:
            # Create AudioAdapter for this call
            audio_adapter = AudioAdapter(
                uplink_capacity=config.audio.uplink_buf_frames,
                downlink_capacity=config.audio.downlink_buf_frames
            )

            # Create AI client for this call
            ai_client = create_ai_client()

            # Create CallSession
            call_session = CallSession(
                audio_adapter=audio_adapter,
                ai_client=ai_client
            )

            # Setup call with these components
            await call.setup(audio_adapter, call_session)

            logger.info(
                "Call resources created",
                call_id=call.call_id,
                ai_vendor=config.ai.vendor
            )

        except Exception as e:
            logger.error(
                "Failed to setup call resources",
                call_id=call.call_id,
                error=str(e),
                exc_info=True
            )
            raise

    # Create and run SIP server
    sip_server = AsyncSIPServer(
        host=config.sip.domain,
        port=config.sip.port,
        call_callback=on_incoming_call
    )

    logger.info(
        "SIP server ready - waiting for INVITE requests",
        host=config.sip.domain,
        port=config.sip.port,
        ai_vendor=config.ai.vendor
    )

    try:
        await sip_server.run()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await sip_server.stop()


async def main() -> None:
    """Main application entry point - starts SIP endpoint with AI bridge."""
    logger = structlog.get_logger(__name__)

    logger.info(
        "SIP-to-AI Bridge starting",
        version="0.1.0",
        ai_vendor=config.ai.vendor
    )

    # Setup signal handlers
    def signal_handler(sig: int, frame: any) -> None:
        logger.info(f"Received signal {sig}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Always run with SIP and AI services
    await run_real_mode()


def cli() -> None:
    """CLI entry point."""
    import argparse

    # Setup logging BEFORE anything else
    setup_logging()

    parser = argparse.ArgumentParser(
        description="SIP-to-AI Bridge: Bidirectional audio bridge between SIP and AI services"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    args = parser.parse_args()

    logger = structlog.get_logger(__name__)
    logger.info("Starting SIP-to-AI Bridge")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()