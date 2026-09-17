"""Async call abstraction with TaskGroup.

Manages a single SIP call with RTP session, audio bridge, and AI integration.
"""

import asyncio
import contextlib
import random
from typing import TYPE_CHECKING, Optional

import structlog

from app.ai.duplex_base import describe_error
from app.sip_async.audio_bridge import RTPAudioBridge
from app.sip_async.rtp_session import PortBindError, RTPConfig, RTPSession
from app.sip_async.sdp import (
    build_sdp,
    extract_remote_rtp_info,
    get_answer_direction,
    parse_sdp,
    select_codec,
)
from app.sip_async.sip_protocol import SIPDialog, SIPMessage, SIPMethod

if TYPE_CHECKING:
    from app.bridge import AudioAdapter, CallSession
    from app.sip_async.async_sip_server import AsyncSIPServer

logger = structlog.get_logger(__name__)


class CallTerminated(Exception):
    """Raised inside the call TaskGroup to end the call and cancel its tasks."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# INVITE server transaction states of the call
STATE_EARLY = "early"  # provisional responses sent, AI session connecting
STATE_ANSWERED = "answered"  # 200 OK sent, waiting for ACK
STATE_CONFIRMED = "confirmed"  # ACK received
STATE_TERMINATED = "terminated"  # final error response sent (487/503)

# SIP timers (RFC 3261 17.1.1.1)
SIP_T1 = 0.5
SIP_T2 = 4.0
SIP_TIMER_H = 64 * SIP_T1  # 32s: give up waiting for ACK

# Media directions of our answer that allow sending RTP
_SENDING_DIRECTIONS = ("sendrecv", "sendonly")


class AsyncCall:
    """Async SIP call with TaskGroup lifecycle management.

    Call flow: INVITE -> 100 Trying + 180 Ringing -> AI session connects ->
    200 OK (retransmitted until ACK) -> media. If the AI session cannot be
    connected within ai_connect_timeout the INVITE is rejected with 503 and no
    dialog is established. CANCEL before 200 OK is answered with 487.
    """

    # Maximum time stop() waits for the call tasks to finish
    STOP_TIMEOUT = 5.0
    # Default maximum time to connect the AI session before answering
    AI_CONNECT_TIMEOUT = 10.0

    def __init__(
        self,
        invite: SIPMessage,
        sip_server: 'AsyncSIPServer',
        local_ip: str,
        local_rtp_port: int,
        ai_connect_timeout: Optional[float] = None
    ):
        """Initialize call from INVITE request.

        Args:
            invite: SIP INVITE message
            sip_server: Parent SIP server
            local_ip: Local IP address for SDP
            local_rtp_port: Allocated RTP port
            ai_connect_timeout: Maximum seconds to connect the AI session
                before answering (defaults to AI_CONNECT_TIMEOUT)

        Raises:
            ValueError: If the SDP has no RTP address or no supported codec
        """
        self.invite = invite
        self.sip = sip_server
        self.local_ip = local_ip
        self.local_rtp_port = local_rtp_port

        # Parse SDP from INVITE
        sdp = parse_sdp(invite.body)
        remote_ip, remote_port = extract_remote_rtp_info(sdp)

        if not remote_ip or not remote_port:
            raise ValueError("Cannot extract RTP info from INVITE SDP")

        payload_type = select_codec(sdp)
        if payload_type is None:
            raise ValueError("No supported audio codec (PCMU/PCMA) in INVITE SDP")

        self.remote_rtp_addr = (remote_ip, remote_port)
        self.payload_type = payload_type
        self.media_direction = get_answer_direction(sdp)
        self.send_enabled = self._sending_allowed(self.media_direction, remote_ip)
        self.ai_connect_timeout = ai_connect_timeout or self.AI_CONNECT_TIMEOUT

        # SDP answer origin (same session id for the whole dialog)
        self._sdp_session_id = random.randint(100000, 999999)
        self._sdp_version = self._sdp_session_id

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

        # Teardown state
        self._stop_requested = asyncio.Event()
        self._finished = asyncio.Event()
        self._run_started = False
        self._cleaned_up = False
        self._bye_sent = False

        # INVITE transaction state and responses resent on retransmission
        self.state = STATE_EARLY
        self._last_provisional: Optional[bytes] = None
        self._final_response: Optional[bytes] = None
        self._last_200_ok: Optional[bytes] = None
        self._ack_received = asyncio.Event()
        self._error_retransmit_task: Optional[asyncio.Task[bool]] = None

    @staticmethod
    def _sending_allowed(direction: str, remote_ip: Optional[str]) -> bool:
        """Return True if RTP may be sent for an answer direction and remote address."""
        return direction in _SENDING_DIRECTIONS and remote_ip != "0.0.0.0"

    @property
    def invite_branch(self) -> str:
        """Branch parameter of the top Via of the initial INVITE."""
        vias = self.invite.headers.get("Via") or [{}]
        return str(vias[0].get("branch", ""))

    @property
    def invite_cseq(self) -> int:
        """CSeq number of the initial INVITE."""
        return int((self.invite.headers.get("CSeq") or {}).get("number", 0))

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
        # Create RTP session with the negotiated codec
        self.rtp_session = RTPSession(
            local_port=self.local_rtp_port,
            remote_addr=self.remote_rtp_addr,
            config=RTPConfig(payload_type=self.payload_type)
        )
        self.rtp_session.set_send_enabled(self.send_enabled)

        # Create audio bridge
        self.audio_adapter = audio_adapter
        self.audio_bridge = RTPAudioBridge(self.rtp_session, audio_adapter)

        # Store AI session
        self.call_session = call_session

        logger.info(
            "Call setup complete",
            call_id=self.call_id,
            local_rtp_port=self.local_rtp_port,
            remote_rtp_addr=self.remote_rtp_addr,
            payload_type=self.payload_type
        )

    def _build_sdp_answer(self) -> str:
        """Build the SDP answer: single negotiated codec, current local port."""
        return build_sdp(
            local_ip=self.local_ip,
            local_port=self.local_rtp_port,
            session_id=self._sdp_session_id,
            session_version=self._sdp_version,
            payload_types=[self.payload_type],
            direction=self.media_direction
        )

    async def send_provisional(self) -> None:
        """Send 100 Trying and 180 Ringing for the initial INVITE."""
        trying = self._build_response(self.invite, 100, "Trying", with_tag=False)
        await self.sip.send_message(trying, self.invite.remote_addr)

        ringing = self._build_response(self.invite, 180, "Ringing")
        self._last_provisional = ringing
        await self.sip.send_message(ringing, self.invite.remote_addr)

        logger.info("Call ringing - 180 Ringing sent", call_id=self.call_id)

    async def accept(self) -> None:
        """Accept call and send 200 OK (retransmitted by _wait_ack until ACK)."""
        # Build SDP answer
        sdp_body = self._build_sdp_answer()

        # Build 200 OK response
        response = self._build_200_ok(sdp_body)
        self._last_200_ok = response
        self._final_response = response
        self.state = STATE_ANSWERED

        # Send via SIP server
        await self.sip.send_message(response, self.invite.remote_addr)

        logger.info(
            "Call accepted - 200 OK sent",
            call_id=self.call_id,
            payload_type=self.payload_type
        )

    async def reject(self, status_code: int, status_text: str) -> None:
        """Send a final error response to the initial INVITE (early state only).

        The response is retransmitted (T1 doubling up to T2) until ACK or for
        at most 32s.

        Args:
            status_code: SIP status code (e.g. 487, 503)
            status_text: Reason phrase
        """
        if self.state != STATE_EARLY:
            return
        self.state = STATE_TERMINATED
        response = self._build_response(self.invite, status_code, status_text)
        self._final_response = response
        await self.sip.send_message(response, self.invite.remote_addr)
        logger.info(
            "INVITE rejected - final response sent",
            call_id=self.call_id,
            status=status_code,
            reason=status_text
        )
        self._error_retransmit_task = asyncio.create_task(
            self._retransmit_until_ack(response),
            name=f"invite-error-retx-{self.call_id[:8]}"
        )

    async def _retransmit_until_ack(self, response: bytes) -> bool:
        """Retransmit a final response until ACK (T1 doubling up to T2) or Timer H.

        Returns:
            True if ACK was received, False after Timer H (32s)
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SIP_TIMER_H
        interval = SIP_T1
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                async with asyncio.timeout(min(interval, remaining)):
                    await self._ack_received.wait()
                return True
            except TimeoutError:
                pass
            if loop.time() >= deadline:
                return False
            await self.sip.send_message(response, self.invite.remote_addr)
            logger.debug("Final response retransmitted (no ACK yet)", call_id=self.call_id, interval=interval)
            interval = min(interval * 2, SIP_T2)

    def on_ack(self) -> None:
        """Handle an ACK for this call (stops response retransmissions)."""
        if not self._ack_received.is_set():
            self._ack_received.set()
            logger.debug("ACK received", call_id=self.call_id, state=self.state)
        if self.state == STATE_ANSWERED:
            self.state = STATE_CONFIRMED

    async def resend_last_response(self, addr: tuple) -> None:
        """Resend the last response for a retransmitted initial INVITE.

        Args:
            addr: Source address of the retransmission
        """
        response = self._final_response or self._last_provisional
        if response is None:
            return
        await self.sip.send_message(response, addr)
        logger.debug("Retransmitted INVITE - last response resent", call_id=self.call_id, state=self.state)

    async def resend_200_ok(self, addr: tuple) -> None:
        """Resend the last response for a retransmitted initial INVITE."""
        await self.resend_last_response(addr)

    async def cancel(self) -> bool:
        """Handle CANCEL of the initial INVITE.

        Returns:
            True if the call was not answered yet and has been terminated
            (487 sent), False if the INVITE already had a final response
        """
        if self.state != STATE_EARLY:
            return False
        logger.info("Call cancelled by caller before answer", call_id=self.call_id)
        await self.reject(487, "Request Terminated")
        await self.stop()
        return True

    async def abort(self, reason: str) -> None:
        """End the call from our side (setup failure or server shutdown).

        An unanswered call is rejected with 503; an answered call gets a BYE.

        Args:
            reason: Reason for logging
        """
        logger.info("Call aborted", call_id=self.call_id, reason=reason, state=self.state)
        if self.state == STATE_EARLY:
            await self.reject(503, "Service Unavailable")
        elif self.state in (STATE_ANSWERED, STATE_CONFIRMED):
            try:
                await self.hangup()
            except Exception as e:
                logger.error("Failed to send BYE", call_id=self.call_id, error=str(e))
        await self.stop()

    def cancel_retransmissions(self) -> None:
        """Stop pending final-response retransmissions (server shutdown)."""
        if self._error_retransmit_task and not self._error_retransmit_task.done():
            self._error_retransmit_task.cancel()

    async def handle_reinvite(self, reinvite: SIPMessage, addr: tuple) -> None:
        """Handle an in-dialog INVITE (re-INVITE) for this call.

        Answers with the same local port and codec. The remote RTP address,
        codec and media direction are updated if the new offer changed them.
        No new RTP session or AI session is created.

        Args:
            reinvite: re-INVITE message
            addr: Source address
        """
        changed = False

        if self.state == STATE_EARLY or self.state == STATE_TERMINATED:
            # No confirmed dialog yet: a second INVITE transaction is pending
            await self.sip.send_message(
                self._build_response(reinvite, 491, "Request Pending"),
                addr
            )
            return

        if reinvite.body.strip():
            sdp = parse_sdp(reinvite.body)
            remote_ip, remote_port = extract_remote_rtp_info(sdp)
            payload_type = select_codec(sdp)

            if payload_type is None:
                logger.warning("re-INVITE without supported codec - rejecting", call_id=self.call_id)
                await self.sip.send_message(
                    self._build_response(reinvite, 488, "Not Acceptable Here"),
                    addr
                )
                return

            # c=0.0.0.0 (RFC 2543 hold) keeps the current remote address
            if remote_ip and remote_port and remote_ip != "0.0.0.0":
                new_addr = (remote_ip, remote_port)
                if new_addr != self.remote_rtp_addr:
                    self.remote_rtp_addr = new_addr
                    changed = True

            if payload_type != self.payload_type:
                self.payload_type = payload_type
                changed = True

            direction = get_answer_direction(sdp)
            if direction != self.media_direction:
                self.media_direction = direction
                changed = True

            # Hold (recvonly/inactive answer or c=0.0.0.0): stop sending RTP
            self.send_enabled = self._sending_allowed(direction, remote_ip)
            if self.rtp_session:
                self.rtp_session.set_send_enabled(self.send_enabled)

            if changed:
                self._sdp_version += 1
                if self.rtp_session:
                    self.rtp_session.update_remote(self.remote_rtp_addr, self.payload_type)

        response = self._build_response(reinvite, 200, "OK", self._build_sdp_answer())
        await self.sip.send_message(response, addr)

        logger.info(
            "re-INVITE answered - 200 OK sent",
            call_id=self.call_id,
            local_rtp_port=self.local_rtp_port,
            remote_rtp_addr=self.remote_rtp_addr,
            payload_type=self.payload_type,
            direction=self.media_direction,
            send_enabled=self.send_enabled,
            changed=changed
        )

    def _build_200_ok(self, sdp_body: str) -> bytes:
        """Build 200 OK response with proper headers from INVITE."""
        return self._build_response(self.invite, 200, "OK", sdp_body)

    def _build_response(
        self,
        request: SIPMessage,
        status_code: int,
        status_text: str,
        sdp_body: str = "",
        with_tag: bool = True
    ) -> bytes:
        """Build a response to an INVITE of this dialog (initial or re-INVITE).

        with_tag=False builds a response without To tag and Contact (100 Trying).
        """
        lines = [
            f"SIP/2.0 {status_code} {status_text}",
        ]

        # Copy Via headers from request
        if "Via" in request.headers:
            for via in request.headers["Via"]:
                via_line = f"{via['type']} {via['address'][0]}:{via['address'][1]}"
                # Add parameters
                for k, v in via.items():
                    if k not in ('type', 'address'):
                        if v is not None:
                            via_line += f";{k}={v}"
                        else:
                            via_line += f";{k}"
                lines.append(f"Via: {via_line}")

        # From header (copy from request)
        if "From" in request.headers:
            from_hdr = request.headers["From"]
            from_line = f"<sip:{from_hdr['address']}>"
            if from_hdr.get('tag'):
                from_line += f";tag={from_hdr['tag']}"
            lines.append(f"From: {from_line}")

        # To header (add our tag)
        if "To" in request.headers:
            to_hdr = request.headers["To"]
            to_line = f"<sip:{to_hdr['address']}>"
            if with_tag:
                to_line += f";tag={self.dialog.local_tag}"
            lines.append(f"To: {to_line}")

        # Call-ID (copy from request)
        lines.append(f"Call-ID: {self.call_id}")

        # CSeq (copy from request)
        if "CSeq" in request.headers:
            cseq = request.headers["CSeq"]
            lines.append(f"CSeq: {cseq['number']} {cseq['method']}")

        # Contact
        if with_tag:
            lines.append(f"Contact: <sip:{self.local_ip}:{self.sip.port}>")

        # Content headers
        if sdp_body:
            lines.append("Content-Type: application/sdp")
        lines.append(f"Content-Length: {len(sdp_body)}")
        lines.append("")  # Empty line before body
        lines.append(sdp_body)

        return '\r\n'.join(lines).encode('utf-8')

    async def run(self) -> None:
        """Connect the AI session, answer the call and run all call tasks.

        The AI session is connected before 200 OK; a connect failure or
        timeout rejects the INVITE with 503. Media runs in a TaskGroup with
        retry logic for RTP port binding failures. The call ends when
        stop() is called (BYE), when the AI session ends (AI disconnect or
        failure) or when a task fails. On exit all components are stopped, the
        call is removed from the server, and BYE is sent to the caller unless
        the call was ended by the caller.
        """
        if not self.rtp_session or not self.audio_bridge or not self.call_session:
            raise RuntimeError("Call not fully setup - call setup() first")

        if self._stop_requested.is_set():
            # Caller hung up before the call started
            await self._cleanup()
            return

        self._running = True
        self._run_started = True
        hangup_reason: Optional[str] = None

        logger.info("Call starting", call_id=self.call_id)

        # Retry port binding up to 3 times
        max_retries = 3
        should_continue = True

        try:
            if not await self._connect_ai():
                return

            if self._stop_requested.is_set():
                return

            await self.accept()

            for attempt in range(max_retries):
                if not should_continue:
                    break

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

                        # AI session (ends the call when the AI session ends)
                        self._session_task = tg.create_task(
                            self._run_ai_session(),
                            name=f"session-{self.call_id[:8]}"
                        )

                        # Stop watcher (ends the call on stop()/BYE)
                        tg.create_task(
                            self._wait_stop_requested(),
                            name=f"stop-{self.call_id[:8]}"
                        )

                        # 200 OK retransmission until ACK (ends the call without ACK)
                        tg.create_task(
                            self._wait_ack(),
                            name=f"ack-{self.call_id[:8]}"
                        )

                        logger.info("Call TaskGroup started", call_id=self.call_id)

                    # Tasks completed normally
                    should_continue = False

                except* PortBindError as eg:
                    # RTP port binding failed
                    port_error = eg.exceptions[0]
                    logger.warning(
                        "RTP port bind failed, retrying with new port",
                        call_id=self.call_id,
                        failed_port=port_error.port,
                        attempt=attempt + 1,
                        max_retries=max_retries
                    )

                    if attempt < max_retries - 1:
                        # Release failed port and allocate new one
                        await self.sip.release_rtp_port(port_error.port)
                        new_port = await self.sip.allocate_rtp_port()

                        # Update RTP session with new port
                        self.rtp_session.update_port(new_port)
                        self.local_rtp_port = new_port

                        logger.info(
                            "Retrying with new RTP port",
                            call_id=self.call_id,
                            new_port=new_port
                        )
                    else:
                        # Max retries exceeded
                        logger.error(
                            "Max RTP port bind retries exceeded",
                            call_id=self.call_id,
                            max_retries=max_retries
                        )
                        should_continue = False
                        hangup_reason = "rtp port bind failed"
                        raise

                except* CallTerminated as eg:
                    reason = eg.exceptions[0].reason
                    logger.info("Call terminating", call_id=self.call_id, reason=reason)
                    if not self._stop_requested.is_set():
                        hangup_reason = reason
                    should_continue = False

                except* asyncio.CancelledError:
                    # Normal cancellation during shutdown
                    logger.debug("Call tasks cancelled (normal shutdown)", call_id=self.call_id)
                    should_continue = False

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
                    if not self._stop_requested.is_set():
                        hangup_reason = "call failure"
                    should_continue = False

        finally:
            self._running = False
            if hangup_reason and not self._stop_requested.is_set():
                try:
                    await self.hangup()
                except Exception as e:
                    logger.error("Failed to send BYE", call_id=self.call_id, error=str(e))
            await self._cleanup()
            self._finished.set()
            logger.info("Call ended", call_id=self.call_id)

    async def _connect_ai(self) -> bool:
        """Connect the AI session before answering.

        Waits for the connection, a stop request (CANCEL/BYE/shutdown) or
        ai_connect_timeout, whichever comes first. On failure or timeout the
        INVITE is rejected with 503 Service Unavailable.

        Returns:
            True if the AI session is connected and the call can be answered
        """
        assert self.call_session is not None
        logger.info(
            "Connecting AI session before answering",
            call_id=self.call_id,
            timeout=self.ai_connect_timeout
        )

        connect_task = asyncio.create_task(
            self.call_session.start(),
            name=f"ai-connect-{self.call_id[:8]}"
        )
        stop_task = asyncio.create_task(
            self._stop_requested.wait(),
            name=f"ai-connect-stop-{self.call_id[:8]}"
        )
        try:
            done, _ = await asyncio.wait(
                {connect_task, stop_task},
                timeout=self.ai_connect_timeout,
                return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            connect_task.cancel()
            raise
        finally:
            stop_task.cancel()

        if self._stop_requested.is_set():
            if not connect_task.done():
                connect_task.cancel()
            with contextlib.suppress(BaseException):
                await connect_task
            logger.info("Call ended while AI session was connecting", call_id=self.call_id)
            return False

        if connect_task in done:
            error = connect_task.exception()
            if error is None:
                logger.info("AI session connected - answering call", call_id=self.call_id)
                return True
            logger.error(
                "AI connection failed - rejecting call with 503",
                call_id=self.call_id,
                error=describe_error(error)
            )
        else:
            connect_task.cancel()
            with contextlib.suppress(BaseException):
                await connect_task
            logger.error(
                "AI connection timed out - rejecting call with 503",
                call_id=self.call_id,
                timeout=self.ai_connect_timeout
            )

        await self.reject(503, "Service Unavailable")
        return False

    async def _run_ai_session(self) -> None:
        """End the call when the (already connected) AI session ends."""
        assert self.call_session is not None
        await self.call_session.wait_ended()
        raise CallTerminated("ai session ended")

    async def _wait_ack(self) -> None:
        """Retransmit 200 OK until ACK; end the call if no ACK arrives within 32s."""
        if self._last_200_ok is None:
            return
        if not await self._retransmit_until_ack(self._last_200_ok):
            logger.warning(
                "No ACK for 200 OK within 32s - ending call",
                call_id=self.call_id
            )
            raise CallTerminated("no ACK for 200 OK")

    async def _wait_stop_requested(self) -> None:
        """End the call TaskGroup once stop() has been requested."""
        await self._stop_requested.wait()
        raise CallTerminated("stop requested")

    async def _cleanup(self) -> None:
        """Stop all call components and remove the call from the server (once)."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self.audio_bridge:
            await self.audio_bridge.stop()

        if self.call_session:
            await self.call_session.stop()

        if self.rtp_session:
            await self.rtp_session.stop()

        remove_call = getattr(self.sip, "remove_call", None)
        if remove_call is not None:
            await remove_call(self)

        logger.info("Call stopped", call_id=self.call_id)

    async def hangup(self) -> None:
        """Hangup call (send BYE)."""
        self._running = False

        if self._bye_sent:
            return
        self._bye_sent = True

        # Build and send BYE (to the caller's Contact if known)
        remote_target = self.invite.headers.get("Contact") or self.dialog.remote_uri
        bye_msg = self.dialog.build_request(
            method=SIPMethod.BYE,
            request_uri=f"sip:{remote_target}"
        )

        await self.sip.send_message(bye_msg, self.invite.remote_addr)

        logger.info("BYE sent", call_id=self.call_id)

    async def stop(self) -> None:
        """Stop all call tasks (caller hung up or server shutdown).

        Waits until the bridge, AI session and RTP session are stopped and the
        call is removed from the server.
        """
        self._running = False
        self._stop_requested.set()

        if not self._run_started:
            # run() never started: stop components directly
            await self._cleanup()
            return

        if not self._finished.is_set():
            try:
                async with asyncio.timeout(self.STOP_TIMEOUT):
                    await self._finished.wait()
            except TimeoutError:
                logger.warning("Call did not stop in time, forcing cleanup", call_id=self.call_id)
                await self._cleanup()
