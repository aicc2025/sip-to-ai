"""SIP protocol implementation (simplified, inspired by pyVoIP).

Minimal SIP implementation supporting only INVITE/ACK/BYE for incoming calls.
"""

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class SIPMethod(Enum):
    """Supported SIP methods."""
    INVITE = "INVITE"
    ACK = "ACK"
    BYE = "BYE"
    CANCEL = "CANCEL"


class SIPMessageType(Enum):
    """SIP message type."""
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"


@dataclass
class SIPMessage:
    """SIP message (request or response).

    Simplified from pyVoIP - only parses essential headers.
    """

    raw: bytes = b""
    message_type: Optional[SIPMessageType] = None

    # Request fields
    method: Optional[SIPMethod] = None
    request_uri: str = ""

    # Response fields
    status_code: int = 0
    status_text: str = ""

    # Headers (simplified)
    headers: dict[str, Any] = field(default_factory=dict)

    # Body (SDP as string)
    body: str = ""

    # Source address
    remote_addr: tuple[str, int] = ("", 0)

    def __post_init__(self):
        """Parse message if raw data provided."""
        if self.raw:
            self.parse(self.raw)

    def parse(self, data: bytes) -> None:
        """Parse SIP message from raw bytes (inspired by pyVoIP).

        Args:
            data: Raw SIP message bytes
        """
        try:
            # Split headers and body
            if b"\r\n\r\n" in data:
                headers_data, body_data = data.split(b"\r\n\r\n", 1)
                self.body = body_data.decode('utf-8', errors='ignore')
            else:
                headers_data = data
                self.body = ""

            # Parse headers
            lines = headers_data.decode('utf-8', errors='ignore').split('\r\n')
            if not lines:
                raise ValueError("Empty SIP message")

            # First line: request or response
            first_line = lines[0]
            if first_line.startswith("SIP/"):
                # Response: SIP/2.0 200 OK
                self.message_type = SIPMessageType.RESPONSE
                parts = first_line.split(' ', 2)
                if len(parts) >= 2:
                    self.status_code = int(parts[1])
                if len(parts) >= 3:
                    self.status_text = parts[2]
            else:
                # Request: INVITE sip:user@host SIP/2.0
                self.message_type = SIPMessageType.REQUEST
                parts = first_line.split(' ')
                if len(parts) >= 2:
                    try:
                        self.method = SIPMethod(parts[0])
                    except ValueError:
                        logger.warning(f"Unsupported SIP method: {parts[0]}")
                    self.request_uri = parts[1]

            # Parse headers
            for line in lines[1:]:
                if ':' not in line:
                    continue
                header, value = line.split(':', 1)
                header = header.strip()
                value = value.strip()

                self._parse_header(header, value)

        except Exception as e:
            logger.error("SIP parse error", error=str(e), data=data[:200])
            raise

    def _parse_header(self, header: str, value: str) -> None:
        """Parse individual header (simplified from pyVoIP)."""

        if header == "Via":
            # Via: SIP/2.0/UDP 192.168.1.100:5060;branch=z9hG4bK...
            if "Via" not in self.headers:
                self.headers["Via"] = []

            # Extract address
            parts = value.split(';')
            via_info = parts[0].split()
            if len(via_info) >= 2:
                address_port = via_info[1]
                if ':' in address_port:
                    addr, port = address_port.rsplit(':', 1)
                else:
                    addr, port = address_port, "5060"

                via_dict = {
                    "type": via_info[0],
                    "address": (addr, port)
                }

                # Extract parameters (branch, etc.)
                for param in parts[1:]:
                    if '=' in param:
                        k, v = param.split('=', 1)
                        via_dict[k.strip()] = v.strip()

                self.headers["Via"].append(via_dict)

        elif header in ("From", "To"):
            # From: "Alice" <sip:alice@atlanta.com>;tag=1928301774
            tag = ""
            if ";tag=" in value:
                value, tag = value.split(";tag=", 1)

            # Extract address
            match = re.search(r'<sip:([^>]+)>', value)
            if match:
                address = match.group(1)
            else:
                # No angle brackets
                address = value.replace("sip:", "")

            # Extract number and host
            if '@' in address:
                number, host = address.split('@', 1)
            else:
                number = ""
                host = address

            # Extract display name
            display_name = re.sub(r'<.*>', '', value).strip().strip('"')

            self.headers[header] = {
                "raw": value,
                "tag": tag,
                "address": address,
                "number": number,
                "host": host,
                "display_name": display_name
            }

        elif header == "Call-ID":
            self.headers[header] = value

        elif header == "CSeq":
            # CSeq: 1 INVITE
            parts = value.split()
            if len(parts) >= 2:
                self.headers[header] = {
                    "number": int(parts[0]),
                    "method": parts[1]
                }

        elif header == "Contact":
            # Contact: <sip:alice@192.168.1.100:5060>
            match = re.search(r'<sip:([^>]+)>', value)
            if match:
                self.headers[header] = match.group(1)
            else:
                self.headers[header] = value.replace("sip:", "")

        elif header == "Content-Type":
            self.headers[header] = value

        elif header == "Content-Length":
            self.headers[header] = int(value)

        else:
            # Store as-is for other headers
            self.headers[header] = value


@dataclass
class SIPDialog:
    """SIP dialog state for a call."""

    call_id: str
    local_tag: str
    remote_tag: str
    cseq: int = 1
    local_uri: str = ""
    remote_uri: str = ""
    contact: str = ""

    @classmethod
    def from_invite(cls, invite: SIPMessage, local_uri: str) -> 'SIPDialog':
        """Create dialog from INVITE request."""
        return cls(
            call_id=invite.headers.get("Call-ID", ""),
            local_tag=str(uuid.uuid4())[:8],  # Generate tag for 200 OK
            remote_tag=invite.headers.get("From", {}).get("tag", ""),
            local_uri=local_uri,
            remote_uri=invite.headers.get("From", {}).get("address", ""),
            contact=local_uri
        )

    def build_response(self, status_code: int, status_text: str, sdp_body: str = "") -> bytes:
        """Build SIP response message.

        Args:
            status_code: SIP status code (e.g., 200)
            status_text: Status text (e.g., "OK")
            sdp_body: Optional SDP body

        Returns:
            Raw SIP response bytes
        """
        lines = [
            f"SIP/2.0 {status_code} {status_text}",
        ]

        # Add headers
        # Via: Copy from request (will be set by caller)
        # From: Copy from request (will be set by caller)
        # To: Add our tag
        # Call-ID: Same as request
        # CSeq: Same as request

        lines.append(f"Call-ID: {self.call_id}")
        lines.append(f"To: <sip:{self.local_uri}>;tag={self.local_tag}")
        lines.append(f"From: <sip:{self.remote_uri}>;tag={self.remote_tag}")
        lines.append(f"Contact: <sip:{self.contact}>")

        if sdp_body:
            lines.append("Content-Type: application/sdp")
            lines.append(f"Content-Length: {len(sdp_body)}")
        else:
            lines.append("Content-Length: 0")

        lines.append("")  # Empty line before body

        if sdp_body:
            lines.append(sdp_body)

        return '\r\n'.join(lines).encode('utf-8')

    def build_request(self, method: SIPMethod, request_uri: str) -> bytes:
        """Build SIP request message.

        Args:
            method: SIP method
            request_uri: Request URI

        Returns:
            Raw SIP request bytes
        """
        lines = [
            f"{method.value} {request_uri} SIP/2.0",
        ]

        # Add headers
        lines.append(f"Call-ID: {self.call_id}")
        lines.append(f"From: <sip:{self.local_uri}>;tag={self.local_tag}")
        lines.append(f"To: <sip:{self.remote_uri}>;tag={self.remote_tag}")
        lines.append(f"CSeq: {self.cseq} {method.value}")
        lines.append(f"Contact: <sip:{self.contact}>")
        lines.append("Content-Length: 0")
        lines.append("")

        self.cseq += 1

        return '\r\n'.join(lines).encode('utf-8')


class SIPProtocol(asyncio.DatagramProtocol):
    """Asyncio datagram protocol for SIP."""

    def __init__(self, server: 'AsyncSIPServer'):
        """Initialize SIP protocol.

        Args:
            server: Parent SIP server
        """
        self.server = server
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        """Called when UDP socket is ready."""
        self.transport = transport  # type: ignore
        self.server.transport = transport  # type: ignore
        logger.info("SIP protocol connection made")

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        """Called when SIP message is received.

        Args:
            data: Raw SIP message
            addr: Source address
        """
        try:
            msg = SIPMessage(raw=data, remote_addr=addr)

            # Handle in async context with exception tracking
            task = asyncio.create_task(
                self.server.handle_message(msg, addr),
                name=f"sip-msg-{msg.method or 'response'}"
            )
            task.add_done_callback(self._handle_message_task_done)

        except Exception as e:
            logger.error("SIP message parse error", error=str(e), addr=addr)

    def _handle_message_task_done(self, task: asyncio.Task) -> None:
        """Handle message task completion and check for exceptions.

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
            # Unexpected exception - log it (double safety check)
            logger.error(
                "Unhandled exception in message handler task",
                error=str(e),
                exc_info=e,
                task_name=task.get_name()
            )

    def error_received(self, exc: Exception) -> None:
        """Called when socket error occurs."""
        logger.error("SIP protocol error", error=str(exc))
