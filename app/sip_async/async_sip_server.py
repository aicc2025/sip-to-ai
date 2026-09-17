"""Async SIP server with TaskGroup.

Main entry point for SIP functionality - listens for INVITE requests
and creates AsyncCall instances.
"""

import asyncio
import random
import time
import uuid
from typing import Callable, Optional

import structlog

from app.sip_async.async_call import SIP_TIMER_H, AsyncCall
from app.sip_async.sip_protocol import SIPMessage, SIPMethod, SIPMessageType, SIPProtocol

logger = structlog.get_logger(__name__)

# Methods advertised in Allow (OPTIONS 200, 405/501 responses)
ALLOWED_METHODS = "INVITE, ACK, BYE, CANCEL, OPTIONS"
# Standard SIP methods that are recognized but not supported (405); any other
# method is unknown (501)
KNOWN_UNSUPPORTED_METHODS = frozenset({
    "REGISTER", "PRACK", "SUBSCRIBE", "NOTIFY", "PUBLISH", "INFO", "REFER", "MESSAGE", "UPDATE",
})
# How long an ended call is remembered (retransmitted BYE/INVITE, late ACK)
RECENTLY_ENDED_TTL = SIP_TIMER_H


class AsyncSIPServer:
    """Async SIP server using TaskGroup.

    Listens for incoming INVITE requests and creates AsyncCall instances.
    """

    def __init__(
        self,
        host: str,
        port: int,
        call_callback: Optional[Callable[[AsyncCall], None]] = None,
        ai_connect_timeout: Optional[float] = None
    ):
        """Initialize SIP server.

        Args:
            host: Local IP address to bind
            port: SIP port (usually 5060)
            call_callback: Optional callback when call is created
            ai_connect_timeout: Maximum seconds a call waits for its AI session
                before the INVITE is rejected with 503
        """
        self.host = host
        self.port = port
        self.call_callback = call_callback
        self.ai_connect_timeout = ai_connect_timeout

        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[SIPProtocol] = None

        # Active calls
        self.active_calls: dict[str, AsyncCall] = {}
        self._calls_lock = asyncio.Lock()
        # Call-IDs whose initial INVITE is being processed (retransmission guard)
        self._pending_invites: set[str] = set()
        # Recently ended calls: Call-ID -> (expiry monotonic time, call)
        self._recently_ended: dict[str, tuple[float, AsyncCall]] = {}

        # RTP port pool (10000-20000)
        self._rtp_port_min = 10000
        self._rtp_port_max = 20000
        self._allocated_ports: set[int] = set()
        self._port_lock = asyncio.Lock()

        self._running = False
        self._stopped = False

    async def start(self) -> None:
        """Start SIP server (create UDP endpoint)."""
        loop = asyncio.get_running_loop()

        transport, protocol = await loop.create_datagram_endpoint(
            lambda: SIPProtocol(self),
            local_addr=(self.host, self.port)
        )

        self.transport = transport  # type: ignore
        self.protocol = protocol  # type: ignore
        self._running = True

        logger.info(
            "SIP server started",
            host=self.host,
            port=self.port
        )

    async def run(self) -> None:
        """Run SIP server (monitor active calls)."""
        await self.start()

        logger.info("SIP server running - waiting for INVITE requests")

        try:
            # Keep running and monitor active calls
            while self._running:
                await asyncio.sleep(1)

                # Clean up completed calls (thread-safe)
                async with self._calls_lock:
                    # Create snapshot to avoid dict modification during iteration
                    ended_calls = [
                        call_id for call_id, call in list(self.active_calls.items())
                        if call._cleaned_up
                    ]

                    # Collect calls to release ports
                    calls_to_release = []
                    for call_id in ended_calls:
                        call = self.active_calls.pop(call_id)
                        calls_to_release.append(call)
                        self._remember_ended(call)

                        logger.info(
                            "Call removed from active calls",
                            call_id=call_id,
                            active_count=len(self.active_calls)
                        )

                # Free RTP ports outside the calls_lock (avoid nested locks)
                for call in calls_to_release:
                    await self.release_rtp_port(call.local_rtp_port)

                self._prune_recently_ended()

        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop SIP server (idempotent).

        Every active call is ended first: answered calls get a BYE, calls
        still connecting their AI session get 503. The transport is closed
        after all calls have stopped.
        """
        if self._stopped:
            return
        self._stopped = True
        self._running = False

        calls = list(self.active_calls.values())
        if calls:
            logger.info("Hanging up active calls before shutdown", count=len(calls))
            results = await asyncio.gather(
                *(call.abort("server shutdown") for call in calls),
                return_exceptions=True
            )
            for call, result in zip(calls, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error("Error ending call during shutdown", call_id=call.call_id, error=str(result))

        for _, call in self._recently_ended.values():
            call.cancel_retransmissions()
        self._recently_ended.clear()

        if self.transport:
            self.transport.close()

        logger.info("SIP server stopped")

    def _remember_ended(self, call: AsyncCall) -> None:
        """Remember an ended call for retransmissions (BYE, INVITE, ACK)."""
        self._recently_ended[call.call_id] = (time.monotonic() + RECENTLY_ENDED_TTL, call)

    def _prune_recently_ended(self) -> None:
        """Forget ended calls older than RECENTLY_ENDED_TTL."""
        now = time.monotonic()
        for call_id in [cid for cid, (expiry, _) in self._recently_ended.items() if expiry <= now]:
            del self._recently_ended[call_id]

    def _recently_ended_call(self, call_id: str) -> Optional[AsyncCall]:
        """Return a recently ended call by Call-ID, if still remembered."""
        self._prune_recently_ended()
        entry = self._recently_ended.get(call_id)
        return entry[1] if entry else None

    async def allocate_rtp_port(self) -> int:
        """Allocate RTP port from pool (thread-safe).

        Returns:
            Available RTP port number
        """
        async with self._port_lock:
            # Try random ports first
            for _ in range(100):
                port = random.randint(self._rtp_port_min, self._rtp_port_max)
                if port not in self._allocated_ports:
                    self._allocated_ports.add(port)
                    return port

            # Fallback: linear search
            for port in range(self._rtp_port_min, self._rtp_port_max):
                if port not in self._allocated_ports:
                    self._allocated_ports.add(port)
                    return port

            raise RuntimeError("No RTP ports available")

    async def remove_call(self, call: AsyncCall) -> None:
        """Remove an ended call and release its RTP port.

        Args:
            call: Ended call
        """
        async with self._calls_lock:
            removed = self.active_calls.get(call.call_id) is call
            if removed:
                del self.active_calls[call.call_id]
        self._remember_ended(call)

        await self.release_rtp_port(call.local_rtp_port)

        if removed:
            logger.info(
                "Call removed from active calls",
                call_id=call.call_id,
                active_count=len(self.active_calls)
            )

    async def release_rtp_port(self, port: int) -> None:
        """Release RTP port back to pool (thread-safe).

        Args:
            port: RTP port to release
        """
        async with self._port_lock:
            if port in self._allocated_ports:
                self._allocated_ports.remove(port)

    async def handle_message(self, msg: SIPMessage, addr: tuple) -> None:
        """Handle incoming SIP message.

        Args:
            msg: Parsed SIP message
            addr: Source address
        """
        try:
            if msg.message_type == SIPMessageType.REQUEST:
                await self._handle_request(msg, addr)
            elif msg.message_type == SIPMessageType.RESPONSE:
                await self._handle_response(msg, addr)

        except Exception as e:
            logger.error(
                "Error handling SIP message",
                error=str(e),
                method=msg.method_name,
                status=msg.status_code,
                exc_info=True
            )

    async def _handle_request(self, msg: SIPMessage, addr: tuple) -> None:
        """Handle SIP request.

        Args:
            msg: SIP request message
            addr: Source address
        """
        if msg.method == SIPMethod.INVITE:
            await self._handle_invite(msg, addr)

        elif msg.method == SIPMethod.ACK:
            self._handle_ack(msg)

        elif msg.method == SIPMethod.BYE:
            await self._handle_bye(msg, addr)

        elif msg.method == SIPMethod.CANCEL:
            await self._handle_cancel(msg, addr)

        elif msg.method == SIPMethod.OPTIONS:
            logger.debug("Received OPTIONS", call_id=msg.headers.get("Call-ID"))
            await self._send_simple_response(
                msg, addr, 200, "OK",
                extra_headers=[f"Allow: {ALLOWED_METHODS}", "Accept: application/sdp"]
            )

        else:
            method = msg.method_name
            if method.upper() in KNOWN_UNSUPPORTED_METHODS:
                status_code, status_text = 405, "Method Not Allowed"
            else:
                status_code, status_text = 501, "Not Implemented"
            logger.info(
                "Unsupported SIP method",
                method=method,
                status=status_code,
                call_id=msg.headers.get("Call-ID")
            )
            await self._send_simple_response(
                msg, addr, status_code, status_text,
                extra_headers=[f"Allow: {ALLOWED_METHODS}"]
            )

    def _handle_ack(self, ack: SIPMessage) -> None:
        """Handle ACK: stops 200 OK / error response retransmission of the call."""
        call_id = ack.headers.get("Call-ID", "")
        logger.debug("Received ACK", call_id=call_id)
        call = self.active_calls.get(call_id) or self._recently_ended_call(call_id)
        if call is not None:
            call.on_ack()

    async def _handle_cancel(self, cancel: SIPMessage, addr: tuple) -> None:
        """Handle CANCEL of a pending initial INVITE.

        200 OK answers the CANCEL; if the INVITE is not answered yet it gets
        487 Request Terminated and the call is torn down (no dialog, no BYE).
        A CANCEL that matches no INVITE transaction gets 481.

        Args:
            cancel: CANCEL message
            addr: Source address
        """
        call_id = cancel.headers.get("Call-ID", "")
        call = self.active_calls.get(call_id) or self._recently_ended_call(call_id)

        vias = cancel.headers.get("Via") or [{}]
        branch = vias[0].get("branch", "")
        cseq = (cancel.headers.get("CSeq") or {}).get("number")
        matches = (
            call is not None
            and (not branch or not call.invite_branch or branch == call.invite_branch)
            and (cseq is None or cseq == call.invite_cseq)
        )
        if not matches:
            logger.info("CANCEL for unknown transaction - 481", call_id=call_id)
            await self._send_simple_response(cancel, addr, 481, "Call/Transaction Does Not Exist")
            return

        assert call is not None
        logger.info("Received CANCEL", call_id=call_id, state=call.state)
        await self._send_simple_response(cancel, addr, 200, "OK", to_tag=call.dialog.local_tag)
        await call.cancel()

    async def _handle_invite(self, invite: SIPMessage, addr: tuple) -> None:
        """Handle INVITE request.

        - New Call-ID: create a call.
        - Active Call-ID with a To tag: re-INVITE, answered by the existing call.
        - Active or in-progress Call-ID without a To tag: retransmitted initial
          INVITE, the 200 OK is resent (no new call).

        Args:
            invite: INVITE message
            addr: Source address
        """
        call_id = invite.headers.get("Call-ID", "")

        # Checked synchronously (no await) so concurrent retransmissions of the
        # same INVITE can never create a second call
        existing = self.active_calls.get(call_id)
        if existing is not None:
            to_tag = invite.headers.get("To", {}).get("tag")
            if to_tag:
                logger.info("Incoming re-INVITE", call_id=call_id, from_addr=addr)
                await existing.handle_reinvite(invite, addr)
            else:
                logger.info("Retransmitted INVITE - resending last response", call_id=call_id, state=existing.state)
                await existing.resend_last_response(addr)
            return

        ended = self._recently_ended_call(call_id)
        if ended is not None and not invite.headers.get("To", {}).get("tag"):
            logger.info("Retransmitted INVITE for ended call - resending final response", call_id=call_id)
            await ended.resend_last_response(addr)
            return

        if invite.headers.get("To", {}).get("tag"):
            # In-dialog INVITE for a dialog that does not exist (anymore)
            logger.info("re-INVITE for unknown call - 481", call_id=call_id)
            await self._send_simple_response(invite, addr, 481, "Call/Transaction Does Not Exist")
            return

        if call_id in self._pending_invites:
            logger.info("Retransmitted INVITE ignored (call being set up)", call_id=call_id)
            return

        logger.info(
            "Incoming INVITE",
            call_id=call_id,
            from_addr=addr,
            from_uri=invite.headers.get("From", {}).get("address")
        )

        self._pending_invites.add(call_id)
        local_rtp_port = None
        try:
            # Allocate RTP port (thread-safe)
            local_rtp_port = await self.allocate_rtp_port()

            # Create call
            try:
                call = AsyncCall(
                    invite=invite,
                    sip_server=self,
                    local_ip=self.host,
                    local_rtp_port=local_rtp_port,
                    ai_connect_timeout=self.ai_connect_timeout
                )
            except ValueError as e:
                logger.warning("Rejecting INVITE", call_id=call_id, error=str(e))
                await self.release_rtp_port(local_rtp_port)
                local_rtp_port = None
                await self._send_simple_response(invite, addr, 488, "Not Acceptable Here")
                return

            # Store call (thread-safe)
            async with self._calls_lock:
                self.active_calls[call_id] = call

            # 100 Trying + 180 Ringing; 200 OK is sent by call.run() once the
            # AI session is connected (503 if it cannot connect)
            await call.send_provisional()

            # Notify callback (callback will setup AudioAdapter and CallSession)
            if self.call_callback:
                # Run callback in background task with exception handling
                task = asyncio.create_task(
                    self._run_call_callback(call),
                    name=f"call-{call_id[:8]}"
                )
                task.add_done_callback(self._handle_call_task_done)

        except Exception as e:
            logger.error(
                "Failed to handle INVITE",
                call_id=call_id,
                error=str(e),
                exc_info=True
            )

            # Release port if allocated
            if local_rtp_port is not None and call_id not in self.active_calls:
                await self.release_rtp_port(local_rtp_port)
                await self._send_simple_response(invite, addr, 500, "Server Internal Error")

        finally:
            self._pending_invites.discard(call_id)

    async def _run_call_callback(self, call: AsyncCall) -> None:
        """Run call callback and start call.

        Args:
            call: AsyncCall instance
        """
        try:
            # Callback should setup AudioAdapter and CallSession
            if self.call_callback:
                result = self.call_callback(call)
                # Handle async callbacks
                if asyncio.iscoroutine(result):
                    await result

            # Start call (runs in background)
            await call.run()

        except Exception as e:
            logger.error(
                "Call callback/run error",
                call_id=call.call_id,
                error=str(e),
                exc_info=True
            )
            # Call cannot proceed: reject (503) or hang up, release its resources
            if not call._stop_requested.is_set():
                await call.abort("call setup failed")
            else:
                await call.stop()

    def _handle_call_task_done(self, task: asyncio.Task) -> None:
        """Handle call task completion and check for exceptions.

        Args:
            task: Completed task
        """
        try:
            # Check if task raised an exception
            task.result()
        except asyncio.CancelledError:
            # Task was cancelled - this is normal during shutdown
            pass
        except Exception as e:
            # Unexpected exception - log it
            logger.error(
                "Unhandled exception in call task",
                error=str(e),
                exc_info=e,
                task_name=task.get_name()
            )

    @staticmethod
    def _format_sip_header(header_name: str, header_value: any) -> list[str]:
        """Format parsed SIP header value to proper SIP format.

        Args:
            header_name: Header name (e.g., "Via", "From")
            header_value: Parsed header value (dict/list from SIPMessage parser)

        Returns:
            List of formatted header lines (Via can have multiple)
        """
        if header_name == "Via":
            # Via is a list of dicts: [{"type": "SIP/2.0/UDP", "address": ("host", "port"), "branch": "..."}]
            lines = []
            for via in header_value:
                via_line = f"{via['type']} {via['address'][0]}:{via['address'][1]}"
                # Add parameters (branch, rport, etc.)
                for k, v in via.items():
                    if k not in ('type', 'address'):
                        if v is not None:
                            via_line += f";{k}={v}"
                        else:
                            via_line += f";{k}"
                lines.append(f"Via: {via_line}")
            return lines

        elif header_name in ("From", "To"):
            # From/To are dicts: {"raw": "...", "tag": "...", "address": "user@host", "display_name": "..."}
            # Use raw value if available, otherwise reconstruct
            if "raw" in header_value and header_value["raw"]:
                line = header_value["raw"]
            else:
                line = f"<sip:{header_value['address']}>"

            # Add tag if present
            if header_value.get('tag'):
                line += f";tag={header_value['tag']}"

            return [f"{header_name}: {line}"]

        elif header_name == "CSeq":
            # CSeq is a dict: {"number": 123, "method": "BYE"}
            return [f"CSeq: {header_value['number']} {header_value['method']}"]

        elif header_name == "Call-ID":
            # Call-ID is a string
            return [f"Call-ID: {header_value}"]

        else:
            # Fallback for unknown headers
            return [f"{header_name}: {header_value}"]

    async def _handle_bye(self, bye: SIPMessage, addr: tuple) -> None:
        """Handle BYE request (hangup).

        Args:
            bye: BYE message
            addr: Source address
        """
        call_id = bye.headers.get("Call-ID", "")

        logger.info("Received BYE", call_id=call_id)

        # Find call (thread-safe)
        async with self._calls_lock:
            call = self.active_calls.get(call_id)

        if call is None:
            if self._recently_ended_call(call_id) is not None:
                # Retransmitted BYE (our 200 OK was lost) or BYE crossing ours
                logger.debug("BYE for recently ended call - 200 OK", call_id=call_id)
                await self._send_simple_response(bye, addr, 200, "OK")
            else:
                logger.info("BYE for unknown call - 481", call_id=call_id)
                await self._send_simple_response(bye, addr, 481, "Call/Transaction Does Not Exist")
            return

        # Answer first so the caller does not retransmit while the call stops
        await self._send_simple_response(bye, addr, 200, "OK")
        logger.debug("Sent 200 OK for BYE", call_id=call_id)

        # An unanswered INVITE is terminated with 487
        await call.reject(487, "Request Terminated")

        # stop() tears down the bridge, AI session and RTP session and removes the call
        await call.stop()

    async def _send_simple_response(
        self,
        request: SIPMessage,
        addr: tuple,
        status_code: int,
        status_text: str,
        to_tag: Optional[str] = None,
        extra_headers: Optional[list[str]] = None
    ) -> None:
        """Send a response without body, copying the transaction headers.

        Final responses always carry a To tag (RFC 3261 8.2.6.2): the request's
        tag if present, else to_tag, else a new random tag.

        Args:
            request: SIP request being answered
            addr: Destination address
            status_code: SIP status code
            status_text: SIP reason phrase
            to_tag: To tag to add when the request has none
            extra_headers: Additional header lines (e.g. "Allow: ...")
        """
        # Build response with properly formatted headers (RFC 3261)
        lines = [f"SIP/2.0 {status_code} {status_text}"]

        # Copy required headers from request with proper formatting
        for header in ["Via", "From", "To", "Call-ID", "CSeq"]:
            if header in request.headers:
                value = request.headers[header]
                if header == "To" and status_code > 100 and not value.get("tag"):
                    value = {**value, "tag": to_tag or uuid.uuid4().hex[:8]}
                lines.extend(self._format_sip_header(header, value))

        lines.extend(extra_headers or [])
        lines.append("Content-Length: 0")
        lines.append("")  # Empty line before body
        lines.append("")

        response = '\r\n'.join(lines).encode('utf-8')
        await self.send_message(response, addr)

    async def _handle_response(self, response: SIPMessage, addr: tuple) -> None:
        """Handle SIP response.

        Args:
            response: SIP response message
            addr: Source address
        """
        logger.debug(
            "Received SIP response",
            status=response.status_code,
            status_text=response.status_text
        )
        # We don't initiate calls, so no response handling needed

    async def send_message(self, data: bytes, addr: tuple) -> None:
        """Send SIP message.

        Args:
            data: Raw SIP message bytes
            addr: Destination address
        """
        if self.transport:
            self.transport.sendto(data, addr)
        else:
            logger.error("Cannot send message - transport not ready")
