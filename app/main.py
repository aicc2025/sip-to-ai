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
from app.ai.gemini_live import GeminiLiveClient
from app.ai.grok_voice import GrokVoiceClient
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
            endpoint=config.ai.openai_ws_endpoint,
            audio_format=config.ai.openai_audio_format,
            noise_reduction=config.ai.openai_noise_reduction,
            voice=config.ai.openai_voice or None,
            has_project=bool(config.ai.openai_project),
            has_organization=bool(config.ai.openai_organization),
            has_greeting=greeting is not None,
            instructions_length=len(instructions),
            greeting_preview=greeting[:50] if greeting else None
        )

        client = OpenAIRealtimeClient(
            api_key=config.ai.openai_api_key,
            model=config.ai.openai_model,
            voice=config.ai.openai_voice,
            ws_endpoint=config.ai.openai_ws_endpoint,
            project=config.ai.openai_project,
            organization=config.ai.openai_organization,
            instructions=instructions,
            greeting=greeting,
            audio_format=config.ai.openai_audio_format,
            noise_reduction=config.ai.openai_noise_reduction
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

        # Validate 60db voice configuration if selected as the speak provider.
        if config.ai.speak_provider == "60db" and not config.ai.sixtydb_api_key:
            raise ValueError(
                "SPEAK_PROVIDER=60db requires SIXTYDB_API_KEY to be set"
            )

        logger.info(
            "Using Deepgram Voice Agent client",
            prompt_file=config.ai.agent_prompt_file,
            instructions_length=len(instructions),
            has_greeting=greeting is not None,
            greeting_preview=greeting[:50] if greeting else None,
            instructions_preview=instructions[:100] if instructions else None,
            speak_provider=config.ai.speak_provider,
            sixtydb_voice_id=config.ai.sixtydb_voice_id if config.ai.speak_provider == "60db" else None,
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
            greeting=greeting,
            speak_provider=config.ai.speak_provider,
            sixtydb_api_key=config.ai.sixtydb_api_key,
            sixtydb_voice_id=config.ai.sixtydb_voice_id,
        )

    elif vendor == "gemini":
        if not config.ai.gemini_api_key:
            raise ValueError("Gemini API key not configured")

        # Load agent configuration (optional for Gemini)
        instructions, greeting = _load_agent_config(logger)

        logger.info(
            "Using Gemini Live client",
            model=config.ai.gemini_model,
            voice=config.ai.gemini_voice,
            has_greeting=greeting is not None,
            instructions_length=len(instructions),
            greeting_preview=greeting[:50] if greeting else None
        )

        return GeminiLiveClient(
            api_key=config.ai.gemini_api_key,
            model=config.ai.gemini_model,
            voice=config.ai.gemini_voice,
            instructions=instructions,
            greeting=greeting
        )

    elif vendor == "grok":
        if not config.ai.grok_api_key:
            raise ValueError("Grok API key not configured")

        instructions, greeting = _load_agent_config(logger)

        logger.info(
            "Using Grok Voice client",
            model=config.ai.grok_model,
            voice=config.ai.grok_voice,
            has_greeting=greeting is not None,
            instructions_length=len(instructions),
            greeting_preview=greeting[:50] if greeting else None,
        )

        return GrokVoiceClient(
            api_key=config.ai.grok_api_key,
            model=config.ai.grok_model,
            voice=config.ai.grok_voice,
            instructions=instructions,
            greeting=greeting,
            ws_endpoint=config.ai.grok_ws_endpoint,
        )

    else:
        raise ValueError(f"Unsupported AI vendor: {vendor}")


async def run_real_mode(stop_event: Optional[asyncio.Event] = None) -> None:
    """Run in real mode with actual SIP and AI services.

    Each incoming call will create its own AI client and bridge.

    Args:
        stop_event: Set to shut down gracefully (active calls are hung up
            before the SIP transport is closed)
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
        call_callback=on_incoming_call,
        ai_connect_timeout=float(config.system.ai_connection_timeout_sec)
    )

    logger.info(
        "SIP server ready - waiting for INVITE requests",
        host=config.sip.domain,
        port=config.sip.port,
        ai_vendor=config.ai.vendor
    )

    stop_event = stop_event or asyncio.Event()
    server_task = asyncio.create_task(sip_server.run(), name="sip-server")
    stop_task = asyncio.create_task(stop_event.wait(), name="shutdown-wait")

    try:
        done, _ = await asyncio.wait(
            {server_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done:
            logger.info("Shutting down - hanging up active calls")
    finally:
        stop_task.cancel()
        # Hangs up every active call (BYE / 503) before closing the transport
        await sip_server.stop()
        if not server_task.done():
            server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


async def main() -> None:
    """Main application entry point - starts SIP endpoint with AI bridge."""
    logger = structlog.get_logger(__name__)

    logger.info(
        "SIP-to-AI Bridge starting",
        version="0.1.0",
        ai_vendor=config.ai.vendor
    )

    # Graceful shutdown on SIGINT/SIGTERM: the handler only sets an event, so
    # calls are hung up (BYE sent) from the event loop before exit. A second
    # signal cancels the shutdown and exits immediately.
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    main_task = asyncio.current_task()

    def signal_handler(sig: signal.Signals) -> None:
        if stop_event.is_set():
            logger.warning("Received second signal, exiting without waiting", signal=sig.name)
            if main_task is not None:
                main_task.cancel()
            return
        logger.info("Received signal, shutting down gracefully", signal=sig.name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler, sig)

    try:
        # Always run with SIP and AI services
        await run_real_mode(stop_event)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


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
        logger.info("Shutdown complete")
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown complete")
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
