"""Async SIP server with TaskGroup.

Main entry point for SIP functionality - listens for INVITE requests
and creates AsyncCall instances.
"""

import asyncio
import random
from typing import Callable, Optional

import structlog

from app.sip_async.async_call import AsyncCall
from app.sip_async.sip_protocol import SIPMessage, SIPMethod, SIPMessageType, SIPProtocol

logger = structlog.get_logger(__name__)


class AsyncSIPServer:
    """Async SIP server using TaskGroup.

    Listens for incoming INVITE requests and creates AsyncCall instances.
    """

    def __init__(
        self,
        host: str,
        port: int,
        call_callback: Optional[Callable[[AsyncCall], None]] = None
    ):
        """Initialize SIP server.

        Args:
            host: Local IP address to bind
            port: SIP port (usually 5060)
            call_callback: Optional callback when call is created
        """
        self.host = host
        self.port = port
        self.call_callback = call_callback

        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[SIPProtocol] = None

        # Active calls
        self.active_calls: dict[str, AsyncCall] = {}
        self._calls_lock = asyncio.Lock()

        # RTP port pool (10000-20000)
        self._rtp_port_min = 10000
        self._rtp_port_max = 20000
        self._allocated_ports: set[int] = set()
        self._port_lock = asyncio.Lock()

        self._running = False

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
                        if not call._running
                    ]

                    # Collect calls to release ports
                    calls_to_release = []
                    for call_id in ended_calls:
                        call = self.active_calls.pop(call_id)
                        calls_to_release.append(call)

                        logger.info(
                            "Call removed from active calls",
                            call_id=call_id,
                            active_count=len(self.active_calls)
                        )

                # Free RTP ports outside the calls_lock (avoid nested locks)
                for call in calls_to_release:
                    await self.release_rtp_port(call.local_rtp_port)

        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop SIP server."""
        self._running = False

        # Stop all active calls
        for call in list(self.active_calls.values()):
            await call.stop()

        if self.transport:
            self.transport.close()

        logger.info("SIP server stopped")

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
                method=msg.method,
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
            logger.debug("Received ACK", call_id=msg.headers.get("Call-ID"))
            # ACK is for 200 OK confirmation - no action needed

        elif msg.method == SIPMethod.BYE:
            await self._handle_bye(msg, addr)

        else:
            logger.warning("Unsupported SIP method", method=msg.method)

    async def _handle_invite(self, invite: SIPMessage, addr: tuple) -> None:
        """Handle INVITE request (create call).

        Args:
            invite: INVITE message
            addr: Source address
        """
        call_id = invite.headers.get("Call-ID", "")

        logger.info(
            "Incoming INVITE",
            call_id=call_id,
            from_addr=addr,
            from_uri=invite.headers.get("From", {}).get("address")
        )

        local_rtp_port = None
        try:
            # Allocate RTP port (thread-safe)
            local_rtp_port = await self.allocate_rtp_port()

            # Create call
            call = AsyncCall(
                invite=invite,
                sip_server=self,
                local_ip=self.host,
                local_rtp_port=local_rtp_port
            )

            # Store call (thread-safe)
            async with self._calls_lock:
                self.active_calls[call_id] = call

            # Accept call (send 200 OK)
            await call.accept()

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
            if local_rtp_port is not None:
                await self.release_rtp_port(local_rtp_port)

            # TODO: Send 500 Internal Server Error

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

        # Find and stop call (thread-safe)
        async with self._calls_lock:
            call = self.active_calls.get(call_id)

        if call:
            await call.stop()

        # Build 200 OK with properly formatted headers (RFC 3261)
        lines = ["SIP/2.0 200 OK"]

        # Copy required headers from request with proper formatting
        for header in ["Via", "From", "To", "Call-ID", "CSeq"]:
            if header in bye.headers:
                lines.extend(self._format_sip_header(header, bye.headers[header]))

        lines.append("Content-Length: 0")
        lines.append("")  # Empty line before body

        response = '\r\n'.join(lines).encode('utf-8')
        await self.send_message(response, addr)

        logger.debug("Sent 200 OK for BYE", call_id=call_id)

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
