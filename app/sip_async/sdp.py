"""SDP (Session Description Protocol) parsing and generation.

Simplified SDP implementation for audio-only sessions with G.711 codec.
Inspired by pyVoIP but streamlined for our use case.
"""

from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Media direction attributes (RFC 3264)
MEDIA_DIRECTIONS = ("sendrecv", "sendonly", "recvonly", "inactive")

# Direction to answer for each offered direction (RFC 3264 section 6.1)
_ANSWER_DIRECTIONS = {
    "sendrecv": "sendrecv",
    "sendonly": "recvonly",
    "recvonly": "sendonly",
    "inactive": "inactive",
}


@dataclass
class SDPMedia:
    """SDP media description (m= line)."""
    media_type: str  # "audio" or "video"
    port: int
    proto: str  # "RTP/AVP"
    formats: list[int] = field(default_factory=list)  # Payload types

    # Optional attributes
    rtpmap: dict[int, str] = field(default_factory=dict)  # {PT: "codec/rate"}
    connection: Optional[str] = None  # c= line override
    direction: Optional[str] = None  # a=sendrecv/sendonly/recvonly/inactive


@dataclass
class SDPSession:
    """SDP session description."""

    # Session-level fields
    version: int = 0
    origin_username: str = "sip-to-ai"
    session_id: int = 0
    session_version: int = 0
    network_type: str = "IN"
    address_type: str = "IP4"
    address: str = "0.0.0.0"

    session_name: str = "SIP-to-AI Session"
    connection: Optional[str] = None  # c= line
    direction: Optional[str] = None  # session-level a=sendrecv/...

    # Time
    time_start: int = 0
    time_stop: int = 0

    # Media descriptions
    media: list[SDPMedia] = field(default_factory=list)


def parse_sdp(sdp_body: str) -> SDPSession:
    """Parse SDP from string (inspired by pyVoIP).

    Args:
        sdp_body: SDP text content

    Returns:
        Parsed SDP session

    Example SDP:
        v=0
        o=- 123456 123456 IN IP4 192.168.1.100
        s=SIP Call
        c=IN IP4 192.168.1.100
        t=0 0
        m=audio 10000 RTP/AVP 0 8 101
        a=rtpmap:0 PCMU/8000
        a=rtpmap:8 PCMA/8000
        a=rtpmap:101 telephone-event/8000
    """
    session = SDPSession()
    current_media: Optional[SDPMedia] = None

    for line in sdp_body.strip().split('\n'):
        line = line.strip()
        if not line or '=' not in line:
            continue

        field_type = line[0]
        field_value = line[2:].strip()

        try:
            if field_type == 'v':
                # Version
                session.version = int(field_value)

            elif field_type == 'o':
                # Origin: o=<username> <sess-id> <sess-version> <nettype> <addrtype> <addr>
                parts = field_value.split()
                if len(parts) >= 6:
                    session.origin_username = parts[0]
                    session.session_id = int(parts[1])
                    session.session_version = int(parts[2])
                    session.network_type = parts[3]
                    session.address_type = parts[4]
                    session.address = parts[5]

            elif field_type == 's':
                # Session name
                session.session_name = field_value

            elif field_type == 'c':
                # Connection: c=IN IP4 192.168.1.100
                if current_media:
                    current_media.connection = field_value
                else:
                    session.connection = field_value

            elif field_type == 't':
                # Time: t=0 0
                parts = field_value.split()
                if len(parts) >= 2:
                    session.time_start = int(parts[0])
                    session.time_stop = int(parts[1])

            elif field_type == 'm':
                # Media: m=audio 10000 RTP/AVP 0 8 101
                parts = field_value.split()
                if len(parts) >= 3:
                    media = SDPMedia(
                        media_type=parts[0],
                        port=int(parts[1]),
                        proto=parts[2],
                        formats=[int(x) for x in parts[3:] if x.isdigit()]
                    )
                    session.media.append(media)
                    current_media = media

            elif field_type == 'a' and field_value in MEDIA_DIRECTIONS:
                # Direction attribute (media-level overrides session-level)
                if current_media:
                    current_media.direction = field_value
                else:
                    session.direction = field_value

            elif field_type == 'a' and current_media:
                # Attribute
                if field_value.startswith('rtpmap:'):
                    # a=rtpmap:0 PCMU/8000
                    rtpmap_value = field_value[7:]  # Remove "rtpmap:"
                    if ' ' in rtpmap_value:
                        pt_str, codec_info = rtpmap_value.split(' ', 1)
                        pt = int(pt_str)
                        current_media.rtpmap[pt] = codec_info

        except (ValueError, IndexError) as e:
            logger.warning("SDP parse error", line=line, error=str(e))
            continue

    return session


def build_sdp(
    local_ip: str,
    local_port: int,
    session_id: Optional[int] = None,
    payload_types: Optional[list[int]] = None,
    session_version: Optional[int] = None,
    direction: str = "sendrecv"
) -> str:
    """Build SDP for audio session with G.711.

    Args:
        local_ip: Local IP address
        local_port: Local RTP port
        session_id: Session ID (random if None)
        payload_types: Payload types to offer (defaults to [0, 8] for PCMU/PCMA)
        session_version: o= session version (defaults to session_id)
        direction: Media direction attribute

    Returns:
        SDP string
    """
    import random

    if session_id is None:
        session_id = random.randint(100000, 999999)

    if session_version is None:
        session_version = session_id

    if payload_types is None:
        payload_types = [0, 8]  # PCMU, PCMA

    sdp_lines = [
        "v=0",
        f"o=sip-to-ai {session_id} {session_version} IN IP4 {local_ip}",
        "s=SIP-to-AI Audio Session",
        f"c=IN IP4 {local_ip}",
        "t=0 0",
        f"m=audio {local_port} RTP/AVP {' '.join(map(str, payload_types))}",
    ]

    # Add rtpmap for each payload type
    codec_map = {
        0: "PCMU/8000",
        8: "PCMA/8000",
        101: "telephone-event/8000"
    }

    for pt in payload_types:
        if pt in codec_map:
            sdp_lines.append(f"a=rtpmap:{pt} {codec_map[pt]}")

    # Add direction attribute
    sdp_lines.append(f"a={direction}")

    return '\r\n'.join(sdp_lines) + '\r\n'


def extract_remote_rtp_info(sdp: SDPSession) -> tuple[Optional[str], Optional[int]]:
    """Extract remote RTP address and port from SDP.

    Args:
        sdp: Parsed SDP session

    Returns:
        Tuple of (remote_ip, remote_port) or (None, None) if not found
    """
    # Get connection address (session-level or media-level)
    remote_ip: Optional[str] = None

    if sdp.connection:
        # Parse "IN IP4 192.168.1.100"
        parts = sdp.connection.split()
        if len(parts) >= 3:
            remote_ip = parts[2]

    if not remote_ip:
        remote_ip = sdp.address

    # Get first audio media port
    remote_port: Optional[int] = None
    for media in sdp.media:
        if media.media_type == "audio":
            # Check for media-level connection override
            if media.connection:
                parts = media.connection.split()
                if len(parts) >= 3:
                    remote_ip = parts[2]
            remote_port = media.port
            break

    return remote_ip, remote_port


def get_supported_codecs(sdp: SDPSession) -> list[int]:
    """Get list of supported audio codecs from SDP.

    Args:
        sdp: Parsed SDP session

    Returns:
        Payload types we support, in the order of the offer (the offerer's
        preference, RFC 3264 section 6.1)
    """
    supported_pts = (0, 8)  # PCMU, PCMA
    result: list[int] = []

    for media in sdp.media:
        if media.media_type == "audio":
            for pt in media.formats:
                if pt in supported_pts and pt not in result:
                    result.append(pt)

    return result


def select_codec(sdp: SDPSession) -> Optional[int]:
    """Select the single audio codec to answer with.

    Only G.711 is supported, so telephone-event is never answered (DTMF is
    not handled and RTP with other payload types is ignored).

    Args:
        sdp: Parsed SDP offer

    Returns:
        Payload type (0 = PCMU, 8 = PCMA), or None if no supported codec is offered
    """
    codecs = get_supported_codecs(sdp)
    return codecs[0] if codecs else None


def get_answer_direction(sdp: SDPSession) -> str:
    """Return the media direction to answer with for an SDP offer.

    Args:
        sdp: Parsed SDP offer

    Returns:
        Direction attribute value (e.g. "recvonly" for a "sendonly" hold offer)
    """
    offered = sdp.direction or "sendrecv"
    for media in sdp.media:
        if media.media_type == "audio":
            if media.direction:
                offered = media.direction
            break
    return _ANSWER_DIRECTIONS.get(offered, "sendrecv")
