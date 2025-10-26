"""Async call abstraction with TaskGroup.

Manages a single SIP call with RTP session, audio bridge, and AI integration.
"""

import asyncio
from typing import TYPE_CHECKING, Optional

import structlog

from app.sip_async.audio_bridge import RTPAudioBridge
from app.sip_async.rtp_session import RTPSession
from app.sip_async.sdp import build_sdp, extract_remote_rtp_info, parse_sdp
from app.sip_async.sip_protocol import SIPDialog, SIPMessage, SIPMethod

if TYPE_CHECKING:
    from app.bridge import AudioAdapter, CallSession
    from app.sip_async.async_sip_server import AsyncSIPServer

logger = structlog.get_logger(__name__)


class AsyncCall:
    """Async SIP call with TaskGroup lifecycle management."""

    def __init__(
        self,
        invite: SIPMessage,
        sip_server: 'AsyncSIPServer',
        local_ip: str
    ):
        """Initialize call from INVITE request.

        Args:
            invite: SIP INVITE message
            sip_server: Parent SIP server
            local_ip: Local IP address for SDP
        """
        self.invite = invite
        self.sip = sip_server
        self.local_ip = local_ip

        # Parse SDP from INVITE
        sdp = parse_sdp(invite.body)
        remote_ip, remote_port = extract_remote_rtp_info(sdp)

        if not remote_ip or not remote_port:
            raise ValueError("Cannot extract RTP info from INVITE SDP")

        self.remote_rtp_addr = (remote_ip, remote_port)

        # Allocate local RTP port
        self.local_rtp_port = self.sip.allocate_rtp_port()

        # Create SIP dialog
        local_uri = f"{self.local_ip}:{self.sip.port}"
        self.dialog = SIPDialog.from_invite(invite, local_uri)

        # Call ID for logging
        self.call_id = self.dialog.call_id

        # Components (set by caller)
        self.rtp_session: Optional[RTPSession] = None
        self.audio_adapter: Optional['AudioAdapter'] = None
        self.audio_bridge: Optional[RTPAudioBridge] = None
        self.call_session: Optional['CallSession'] = None  # AI session

        self._running = False
        self._session_task: Optional[asyncio.Task] = None

    async def setup(
        self,
        audio_adapter: 'AudioAdapter',
        call_session: 'CallSession'
    ) -> None:
        """Setup call components.

        Args:
            audio_adapter: Audio adapter for AI integration
            call_session: AI call session
        """
        # Create RTP session
        self.rtp_session = RTPSession(
            local_port=self.local_rtp_port,
            remote_addr=self.remote_rtp_addr
        )

        # Create audio bridge
        self.audio_adapter = audio_adapter
        self.audio_bridge = RTPAudioBridge(self.rtp_session, audio_adapter)

        # Store AI session
        self.call_session = call_session

        logger.info(
            "Call setup complete",
            call_id=self.call_id,
            local_rtp_port=self.local_rtp_port,
            remote_rtp_addr=self.remote_rtp_addr
        )

    async def accept(self) -> None:
        """Accept call and send 200 OK."""
        # Build SDP answer
        sdp_body = build_sdp(
            local_ip=self.local_ip,
            local_port=self.local_rtp_port
        )

        # Build 200 OK response
        response = self._build_200_ok(sdp_body)

        # Send via SIP server
        await self.sip.send_message(response, self.invite.remote_addr)

        logger.info("Call accepted - 200 OK sent", call_id=self.call_id)

    def _build_200_ok(self, sdp_body: str) -> bytes:
        """Build 200 OK response with proper headers from INVITE."""
        lines = [
            "SIP/2.0 200 OK",
        ]

        # Copy Via headers from INVITE
        if "Via" in self.invite.headers:
            for via in self.invite.headers["Via"]:
                via_line = f"{via['type']} {via['address'][0]}:{via['address'][1]}"
                # Add parameters
                for k, v in via.items():
                    if k not in ('type', 'address'):
                        if v is not None:
                            via_line += f";{k}={v}"
                        else:
                            via_line += f";{k}"
                lines.append(f"Via: {via_line}")

        # From header (copy from INVITE)
        if "From" in self.invite.headers:
            from_hdr = self.invite.headers["From"]
            from_line = f"<sip:{from_hdr['address']}>"
            if from_hdr.get('tag'):
                from_line += f";tag={from_hdr['tag']}"
            lines.append(f"From: {from_line}")

        # To header (add our tag)
        if "To" in self.invite.headers:
            to_hdr = self.invite.headers["To"]
            to_line = f"<sip:{to_hdr['address']}>;tag={self.dialog.local_tag}"
            lines.append(f"To: {to_line}")

        # Call-ID (copy from INVITE)
        lines.append(f"Call-ID: {self.call_id}")

        # CSeq (copy from INVITE)
        if "CSeq" in self.invite.headers:
            cseq = self.invite.headers["CSeq"]
            lines.append(f"CSeq: {cseq['number']} {cseq['method']}")

        # Contact
        lines.append(f"Contact: <sip:{self.local_ip}:{self.sip.port}>")

        # Content headers
        lines.append("Content-Type: application/sdp")
        lines.append(f"Content-Length: {len(sdp_body)}")
        lines.append("")  # Empty line before body
        lines.append(sdp_body)

        return '\r\n'.join(lines).encode('utf-8')

    async def run(self) -> None:
        """Run call with TaskGroup (manages all call tasks)."""
        if not self.rtp_session or not self.audio_bridge or not self.call_session:
            raise RuntimeError("Call not fully setup - call setup() first")

        self._running = True

        logger.info("Call starting", call_id=self.call_id)

        try:
            async with asyncio.TaskGroup() as tg:
                # RTP session
                tg.create_task(
                    self.rtp_session.run(),
                    name=f"rtp-{self.call_id[:8]}"
                )

                # Audio bridge
                tg.create_task(
                    self.audio_bridge.run(),
                    name=f"bridge-{self.call_id[:8]}"
                )

                # AI session
                self._session_task = tg.create_task(
                    self.call_session.start(),
                    name=f"session-{self.call_id[:8]}"
                )

                logger.info("Call TaskGroup started", call_id=self.call_id)

        except* asyncio.CancelledError:
            # Normal cancellation during shutdown
            logger.debug("Call tasks cancelled (normal shutdown)", call_id=self.call_id)

        except* Exception as eg:
            # Unexpected exceptions
            logger.error(
                "Call TaskGroup exceptions",
                call_id=self.call_id,
                count=len(eg.exceptions)
            )
            for exc in eg.exceptions:
                logger.error(
                    f"Exception: {type(exc).__name__}: {exc}",
                    call_id=self.call_id,
                    exc_info=exc
                )
        finally:
            self._running = False
            logger.info("Call ended", call_id=self.call_id)

    async def hangup(self) -> None:
        """Hangup call (send BYE)."""
        self._running = False

        # Build and send BYE
        bye_msg = self.dialog.build_request(
            method=SIPMethod.BYE,
            request_uri=f"sip:{self.dialog.remote_uri}"
        )

        await self.sip.send_message(bye_msg, self.invite.remote_addr)

        logger.info("BYE sent", call_id=self.call_id)

    async def stop(self) -> None:
        """Stop all call tasks."""
        self._running = False

        # Stop components
        if self.rtp_session:
            await self.rtp_session.stop()

        if self.audio_bridge:
            await self.audio_bridge.stop()

        if self.call_session:
            await self.call_session.stop()

        logger.info("Call stopped", call_id=self.call_id)
