"""RTP media-path tests (issue #6): inbound decode and malformed-packet safety.

The inbound RTP path (caller -> our socket) is where "caller audio is not
logged" originates if packets never arrive or fail to decode. These tests pin
down the real decode behavior so diagnostics can be added safely.
"""

import pytest

from app.sip_async.rtp_session import G711Codec, RTPPacket, RTPProtocol, RTPSession


def _session() -> RTPSession:
    return RTPSession(local_port=40000, remote_addr=("127.0.0.1", 5004))


class TestInboundRtpDecode:
    @pytest.mark.asyncio
    async def test_ulaw_packet_decoded_to_pcm16_and_queued(self) -> None:
        session = _session()
        protocol = RTPProtocol(session)

        ulaw_payload = b"\xff" * 160  # 20ms G.711 @ 8kHz
        packet = RTPPacket.build(
            payload=ulaw_payload, seq=1, timestamp=0, ssrc=123, pt=0
        )

        protocol.datagram_received(packet, ("127.0.0.1", 5004))

        pcm = session.rx_queue.get_nowait()
        assert pcm == G711Codec().decode_pcmu(ulaw_payload)
        assert len(pcm) == 320  # PCM16 is twice the mu-law size

    @pytest.mark.asyncio
    async def test_malformed_packet_does_not_enqueue_or_raise(self) -> None:
        session = _session()
        protocol = RTPProtocol(session)

        # Too short to be a valid RTP packet (header is 12 bytes).
        protocol.datagram_received(b"\x00\x01", ("127.0.0.1", 5004))

        assert session.rx_queue.empty()
